from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
import copy
from components.epsilon_schedules import DecayThenFlatSchedule
import torch.nn as nn


# This multi-agent controller shares parameters between agents
class ExpMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.input_shape = self._get_input_shape()
        self._build_agents()
        self.agent_output_type = args.AGENT_OUTPUT_TYPE

        if args.USE_INPUT_NOISE:
            self.schedule = DecayThenFlatSchedule(
                args.INPUT_NOISE_DECAY_START, args.INPUT_NOISE_DECAY_FINISH, args.INPUT_NOISE_DECAY_ANNEAL_TIME,
                decay="linear"
            )

            self.input_noise_epsilon = self.schedule.eval(0)
            self.input_noise = args.INPUT_NOISE
            self.input_noise_clip = args.INPUT_NOISE_CLIP

        self.hidden_for_exp = None

        self.mse_loss = nn.MSELoss(size_average=None, reduce=None, reduction='none')

        self.device = th.device(self.args.DEVICE)

    def forward(self, ep_batch, t, train=True):
        batch_size = ep_batch.batch_size
        obs = ep_batch["obs"][:, t]
        obs = obs.view(batch_size, self.n_agents, -1)
        inputs = obs.clone()
        target_y = self._build_inputs(ep_batch, inputs, t)
        state_y = ep_batch["state"][:, t]

        if self.args.USE_INPUT_NOISE:
            # Build inputs for obs-AE
            inputs_noise = th.normal(
                0, self.input_noise,
                size=inputs.shape
            ).clamp(-self.input_noise_clip, self.input_noise_clip).to(device=self.device)
            inputs = ((self.input_noise_epsilon * inputs_noise + inputs).detach() - inputs).detach() + inputs
        agent_inputs = self._build_inputs(ep_batch, inputs, t)

        if self.args.ENCODER_RNN:
            if train:
                self.hidden_for_exp, inputs_decoded = self.exp_ae(agent_inputs, self.hidden_for_exp)
            else:
                self.hidden_for_exp, inputs_decoded = self.target_exp_ae(agent_inputs, self.hidden_for_exp)
        else:
            if train:
                encoded, inputs_decoded = self.exp_ae(agent_inputs)
            else:
                encoded, inputs_decoded = self.target_exp_ae(agent_inputs)

        inputs_decoded = inputs_decoded.view(batch_size, self.n_agents, -1)
        target_y = target_y.view(batch_size, self.n_agents, -1)

        if train:
            state_decoded = self.exp_sd(self.hidden_for_exp)
        else:
            state_decoded = self.target_exp_sd(self.hidden_for_exp)

        return inputs_decoded, target_y, state_decoded, state_y

    def calc_mse_loss(self, inputs_decoded, target_y, state_decoded, state_y):
        exp_prediction_error = self.mse_loss(inputs_decoded, target_y).sum(2).mean(1)
        exp_state_decoding_error = self.mse_loss(state_decoded, state_y).sum(1)
        return exp_prediction_error, exp_state_decoding_error

    def calc_int_reward(self, batch, ts, t_env):
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_noise_epsilon(steps=t_env)
        inputs_decoded, target_y, state_decoded, state_y = self.forward(batch, t=ts, train=False)
        exp_prediction_error, exp_state_decoding_error = self.calc_mse_loss(inputs_decoded, target_y, state_decoded, state_y)
        int_rewards = self.args.BETA*(self.args.RHO * exp_prediction_error + (1 - self.args.RHO) * exp_state_decoding_error)
        return int_rewards.detach()

    def update_noise_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        if self.args.USE_INPUT_NOISE:
            self.input_noise_epsilon = self.schedule.eval(standard)

    def parameters(self):
        return list(self.exp_ae.parameters()) + list(self.exp_sd.parameters())

    def load_state(self):
        for target_param, local_param in zip(self.target_exp_ae.parameters(), self.exp_ae.parameters()):
            target_param.data.copy_(self.args.SOFT_UPDATE_TAU * local_param.data + (1.0 - self.args.SOFT_UPDATE_TAU) * target_param.data)
        for target_param, local_param in zip(self.target_exp_sd.parameters(), self.exp_sd.parameters()):
            target_param.data.copy_(self.args.SOFT_UPDATE_TAU * local_param.data + (1.0 - self.args.SOFT_UPDATE_TAU) * target_param.data)

    def to_device(self):
        self.exp_ae.to(device=self.device)
        self.target_exp_ae.to(device=self.device)
        self.exp_sd.to(device=self.device)
        self.target_exp_sd.to(device=self.device)

    def save_models(self, path, model_name="{}/exp_ae{}.th"):
        th.save(self.exp_ae.state_dict(), model_name.format(path, ''))
        th.save(self.exp_sd.state_dict(), model_name.format(path, '_state_decoder'))

    def load_models(self, path, model_name="{}/exp_ae{}.th"):
        self.exp_ae.load_state_dict(th.load(model_name.format(path, ''), map_location=lambda storage, loc: storage))
        self.exp_sd.load_state_dict(th.load(model_name.format(path, '_state_decoder'), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.exp_ae = agent_REGISTRY[self.args.EXP_AGENT](self.input_shape, self.args)
        self.target_exp_ae = copy.deepcopy(self.exp_ae)
        self.exp_sd = agent_REGISTRY["state_decoder"](self.args)
        self.target_exp_sd = copy.deepcopy(self.exp_sd)

    def init_hidden(self, batch_size):
        self.hidden_for_exp = self.exp_ae.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def _build_inputs(self, batch, input_state, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(input_state)
        if self.args.OBS_AGENT_ID:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        input_shape = self.scheme["obs"]["vshape"]
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents

        return input_shape