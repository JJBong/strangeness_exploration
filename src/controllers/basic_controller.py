from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.input_shape = self._get_input_shape()
        self._build_agents()
        self.agent_output_type = args.AGENT_OUTPUT_TYPE

        self.action_selector = action_REGISTRY[args.ACTION_SELECTOR](args)

        self.epsilon = args.EPSILON_START

        self.hidden_states = None

        self.device = th.device(self.args.DEVICE)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            # COMA
            if self.args.ALGORITHM == 'coma':
                if getattr(self.args, "mask_before_softmax", True):
                    # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                    reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                    agent_outs[reshaped_avail_actions == 0] = -1e10

                agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
                if not test_mode:
                    # Epsilon floor
                    epsilon_action_num = agent_outs.size(-1)
                    if getattr(self.args, "mask_before_softmax", True):
                        # With probability epsilon, we will pick an available action uniformly
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                    agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                  + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def get_hidden(self):
        return self.hidden_states

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def to_device(self):
        self.agent.to(device=self.device)

    def save_models(self, path, model_name="{}/agent.th"):
        th.save(self.agent.state_dict(), model_name.format(path))

    def save_models_for_hidden(self, path, model_name="{}/hidden_agent.th"):
        th.save(self.agent.state_dict(), model_name.format(path))

    def load_models(self, path, model_name="{}/agent.th"):
        self.agent.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def load_models_for_hidden(self, path, model_name="{}/hidden_agent.th"):
        self.agent.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.AGENT](self.input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.OBS_LAST_ACTION:
            if self.args.ACTION_SPACE == "discrete":
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            elif self.args.ACTION_SPACE == "continuous":
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t-1])
            else:
                raise Exception("ACTION SPACE must be defined ! ")
        if self.args.OBS_AGENT_ID:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if isinstance(self.input_shape, int):
            inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        elif isinstance(self.input_shape, tuple):
            inputs = th.cat([x.reshape(bs*self.n_agents, *self.input_shape) for x in inputs], dim=1)
        else:
            raise NotImplementedError
        return inputs

    def _get_input_shape(self):
        input_shape = self.scheme["obs"]["vshape"]
        if self.args.OBS_LAST_ACTION:
            if self.args.ACTION_SPACE == "discrete":
                input_shape += self.scheme["actions_onehot"]["vshape"][0]
            elif self.args.ACTION_SPACE == "continuous":
                input_shape += self.scheme["actions"]["vshape"]
            else:
                raise Exception("ACTION SPACE must be defined ! ")
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents

        return input_shape

    def gumbel_softmax(self, logits, avail_actions=None, tau: float = 1, hard: bool = False, dim: int = -1,
                       test_mode: bool = False):
        self.epsilon = self.action_selector.epsilon
        gumbels = (
            -th.empty_like(logits, memory_format=th.legacy_contiguous_format).exponential_().log()
        ).detach()  # ~Gumbel(0,1)

        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            # available action (very important!!)
            y_soft_ = y_soft.clone()
            if avail_actions is not None:
                y_soft_[avail_actions == 0] = -float("inf")
            # Straight through.
            index = y_soft_.max(dim, keepdim=True)[1]

            y_hard = th.zeros_like(logits, memory_format=th.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
