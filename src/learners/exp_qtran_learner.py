from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from learners import QTranLearner as QLearner
from utils.rl_utils import overrides


class ExpQLearner(QLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args, hidden_mac=None):
        super().__init__(mac, scheme, logger, args)
        self.n_agents = args.N_AGENTS

        # strangeness index module
        self.exp_mac = exp_mac
        self.exp_mac_params = self.exp_mac.parameters()

        if args.HIDDEN_POLICY:
            # goal action-value function --> self.hidden_mac
            # exploration action-value function --> self.mac
            self.hidden_mac = hidden_mac
            self.params += list(self.hidden_mac.parameters())

        # re-configure optimizer
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.exp_optimizer = RMSprop(params=self.exp_mac_params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.last_target_update_episode = 0

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Exp encoder-decoder training
        exp_prediction_error_list = []
        exp_state_decoding_error_list = []
        intrinsic_reward_list = []

        if self.args.ENCODER_RNN:
            self.exp_mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            if t == 0:
                continue
            inputs_decoded, target_y, state_decoded, state_y = self.exp_mac.forward(batch, t=t)
            exp_prediction_error, exp_state_decoding_error = self.exp_mac.calc_mse_loss(inputs_decoded, target_y,
                                                                                        state_decoded, state_y)
            intrinsic_reward = self.exp_mac.calc_int_reward(batch, t, t_env)
            exp_prediction_error_list.append(exp_prediction_error.mean())
            exp_state_decoding_error_list.append(exp_state_decoding_error.mean())
            intrinsic_reward_list.append(intrinsic_reward)
        exp_prediction_errors = th.stack(exp_prediction_error_list)
        exp_state_decoding_errors = th.stack(exp_state_decoding_error_list)
        intrinsic_rewards = th.stack(intrinsic_reward_list, dim=1).unsqueeze(2).detach()

        exp_prediction_loss = exp_prediction_errors.mean()
        exp_state_decoding_loss = exp_state_decoding_errors.mean()

        self.logger.log_stat("exp_ae_prediction_loss", exp_prediction_loss.item(), t_env)
        self.logger.log_stat("exp_ae_state_decoding_loss", exp_state_decoding_loss.item(), t_env)
        self.logger.log_stat("exp_ae_intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)

        # additional_loss = exp_prediction_loss
        exp_total_loss = exp_prediction_loss + exp_state_decoding_loss
        self.exp_optimizer.zero_grad()
        exp_total_loss.backward()
        th.nn.utils.clip_grad_norm_(parameters=self.exp_mac_params, max_norm=self.args.GRAD_NORM_CLIP)
        self.exp_optimizer.step()

        # if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
        self.exp_mac.load_state()

        additional_loss = None
        # Hidden --------------------------------------------------------------------------------------
        if self.args.HIDDEN_POLICY:
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            # Calculate estimated Q-Values
            mac_out = []
            mac_hidden_states = []
            self.hidden_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.hidden_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
                mac_hidden_states.append(self.hidden_mac.hidden_states)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            mac_hidden_states = th.stack(mac_hidden_states, dim=1)
            mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.N_AGENTS, batch.max_seq_length,
                                                          -1).transpose(1, 2)  # btav

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            target_mac_hidden_states = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                target_mac_hidden_states.append(self.target_mac.hidden_states)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
            target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
            target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.N_AGENTS,
                                                                        batch.max_seq_length, -1).transpose(1, 2)  # btav

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
            mac_out_maxs = mac_out.clone()
            mac_out_maxs[avail_actions == 0] = -9999999

            # Best joint action computed by target agents
            target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            # Best joint-action computed by regular agents
            max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3, keepdim=True)

            if self.args.MIXER == "qtran_base":
                # -- TD Loss --
                # Joint-action Q-Value estimates
                joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1])

                # Need to argmax across the target agents' actions to compute target joint-action Q-Values
                if self.args.DOUBLE_Q:
                    max_actions_current_ = th.zeros(
                        size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS),
                        device=batch.device)
                    max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
                    max_actions_onehot = max_actions_current_onehot
                else:
                    max_actions = th.zeros(
                        size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS),
                        device=batch.device)
                    max_actions_onehot = max_actions.scatter(3, target_max_actions[:, :], 1)
                target_joint_qs, target_vs = self.target_mixer(batch[:, 1:], hidden_states=target_mac_hidden_states[:, 1:],
                                                               actions=max_actions_onehot[:, 1:])

                # Td loss targets
                td_targets = rewards.reshape(-1, 1) + self.args.GAMMA * (1 - terminated.reshape(-1, 1)) * target_joint_qs
                td_error = (joint_qs - td_targets.detach())
                masked_td_error = td_error * mask.reshape(-1, 1)
                td_loss = (masked_td_error ** 2).sum() / mask.sum()
                # -- TD Loss --

                # -- Opt Loss --
                # Argmax across the current agents' actions
                if not self.args.DOUBLE_Q:  # Already computed if we're doing double Q-Learning
                    max_actions_current_ = th.zeros(
                        size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS),
                        device=batch.device)
                    max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
                max_joint_qs, _ = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1], actions=max_actions_current_onehot[:, :-1])  # Don't use the target network and target agent max actions as per author's email

                # max_actions_qvals = th.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
                opt_error = max_actions_qvals[:, :-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
                masked_opt_error = opt_error * mask.reshape(-1, 1)
                opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
                # -- Opt Loss --

                # -- Nopt Loss --
                # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
                nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs  # Don't use target networks here either
                nopt_error = nopt_values.clamp(max=0)
                masked_nopt_error = nopt_error * mask.reshape(-1, 1)
                nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
                # -- Nopt loss --

            elif self.args.MIXER == "qtran_alt":
                raise Exception("Not supported yet.")

            loss = td_loss + self.args.OPT_LOSS * opt_loss + self.args.NOPT_MIN_LOSS * nopt_loss

            additional_loss += loss
        # ---------------------------------------------------------------------------------------------

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss, intrinsic_rewards)

    @overrides(QLearner)
    def _update_targets(self):
        if self.args.HIDDEN_POLICY:
            self.target_mac.load_state(self.hidden_mac)
        else:
            self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()
        if self.args.HIDDEN_POLICY:
            self.hidden_mac.to_device()

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)
        self.hidden_mac.save_models_for_hidden(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
        self.hidden_mac.load_models_for_hidden(path)
