from modules.agents import REGISTRY as agent_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class IcmMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.input_shape = self._get_input_shape()
        self._build_agents()
        self.agent_output_type = args.AGENT_OUTPUT_TYPE

        self.device = th.device(self.args.DEVICE)

    def forward(self, ep_batch, t):
        batch_size = ep_batch.batch_size
        obs = ep_batch["obs"][:, t-1]
        next_obs = ep_batch["obs"][:, t]
        action = ep_batch["actions_onehot"][:, t-1]
        obs = obs.view(batch_size, self.n_agents, -1)
        next_obs = next_obs.view(batch_size, self.n_agents, -1)
        inputs = obs
        next_inputs = next_obs

        # Build inputs
        agent_inputs = self._build_inputs(ep_batch, inputs, t)
        agent_next_inputs = self._build_inputs(ep_batch, next_inputs, t)

        agent_inputs = agent_inputs.view(batch_size, self.n_agents, -1)
        agent_next_inputs = agent_next_inputs.view(batch_size, self.n_agents, -1)
        action = action.reshape(batch_size, self.n_agents, -1)
        action_y = action.clone()

        real_next_inputs_feature, pred_next_inputs_feature, pred_action = self.icm(agent_inputs, agent_next_inputs, action)

        action_y = action_y.reshape(batch_size, self.n_agents, -1)
        pred_action = pred_action.reshape(batch_size, self.n_agents, -1)
        return real_next_inputs_feature, pred_next_inputs_feature, pred_action, action_y

    def calc_mse_loss(self, real_next_inputs_feature, pred_next_inputs_feature):
        diff = real_next_inputs_feature - pred_next_inputs_feature
        prediction_error = (diff ** 2).mean(2).mean(1)

        return prediction_error

    def calc_ce_loss(self, pred_action, action_y):
        diff = pred_action - action_y
        cross_entropy_error = (diff ** 2).mean(2).mean(1)
        return cross_entropy_error

    def calc_int_reward(self, batch, ts, t_env):
        real_next_inputs_feature, pred_next_inputs_feature, _, _ = self.forward(batch, t=ts)
        prediction_error = self.calc_mse_loss(real_next_inputs_feature, pred_next_inputs_feature)
        int_rewards = self.args.RHO * prediction_error
        return int_rewards.detach()

    def parameters(self):
        return self.icm.parameters()

    def load_state(self):
        pass

    def to_device(self):
        self.icm.to(device=self.device)

    def save_models(self, path, model_name="{}/icm.th"):
        th.save(self.icm.state_dict(), model_name.format(path))

    def load_models(self, path, model_name="{}/icm.th"):
        self.icm.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def init_hidden(self, batch_size):
        pass

    def update_noise_epsilon(self, steps=0, train_steps=0):
        pass

    def _build_agents(self):
        self.icm = agent_REGISTRY[self.args.EXP_AGENT](self.input_shape, self.args)

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
