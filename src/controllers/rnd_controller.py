from modules.agents import REGISTRY as agent_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class RndMAC:
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
        obs = obs.view(batch_size, self.n_agents, -1)
        inputs = obs

        # Build inputs
        agent_inputs = self._build_inputs(ep_batch, inputs, t)

        predict_features, target_features = self.rnd(agent_inputs)
        
        predict_features = predict_features.view(batch_size, self.n_agents, -1)
        target_features = target_features.view(batch_size, self.n_agents, -1)

        return predict_features, target_features

    def calc_mse_loss(self, predict_features, target_features):
        obs_diff = predict_features - target_features
        obs_prediction_error = (obs_diff ** 2).mean(2).mean(1)
        return obs_prediction_error

    def calc_int_reward(self, batch, ts, t_env):
        predict_features, target_features = self.forward(batch, t=ts)
        obs_prediction_error = self.calc_mse_loss(predict_features, target_features)
        int_rewards = self.args.RHO * obs_prediction_error
        return int_rewards.detach()

    def parameters(self):
        return self.rnd.parameters()

    def load_state(self):
        pass

    def to_device(self):
        self.rnd.to(device=self.device)

    def save_models(self, path, model_name="{}/rnd.th"):
        th.save(self.rnd.state_dict(), model_name.format(path))

    def load_models(self, path, model_name="{}/rnd.th"):
        self.rnd.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def init_hidden(self, batch_size):
        pass

    def update_noise_epsilon(self, steps=0, train_steps=0):
        pass

    def _build_agents(self):
        self.rnd = agent_REGISTRY[self.args.EXP_AGENT](self.input_shape, self.args)

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
