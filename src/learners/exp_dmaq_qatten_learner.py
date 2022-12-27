from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from learners import DMAQ_qattenLearner


class ExpDMAQ_qattenLearner(DMAQ_qattenLearner):
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
            self.params += self.hidden_mac.parameters()

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
            actions_onehot = batch["actions_onehot"][:, :-1]

            # Calculate estimated Q-Values
            mac_out = []
            self.hidden_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.hidden_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            x_mac_out = mac_out.clone().detach()
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            # Max over target Q-Values
            if self.args.DOUBLE_Q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                target_max_qvals = target_mac_out.max(dim=3)[0]
                target_next_actions = cur_max_actions.detach()

                cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,))
                cur_max_actions_onehot = cur_max_actions_onehot.to(device=self.device)
                cur_max_actions = cur_max_actions.to(device=self.device)
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            else:
                # Calculate the Q-Values necessary for the target
                target_mac_out = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
                target_max_qvals = target_mac_out.max(dim=3)[0]

            # Mix
            if self.mixer is not None:
                if self.args.MIXER == "dmaq_qatten":
                    ans_chosen, q_attend_regs, head_entropies = \
                        self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                    ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                          max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv
                else:
                    ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                    ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                    max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv

                if self.args.DOUBLE_Q:
                    if self.args.MIXER == "dmaq_qatten":
                        target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                        target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                             actions=cur_max_actions_onehot,
                                                             max_q_i=target_max_qvals, is_v=False)
                        target_max_qvals = target_chosen + target_adv
                    else:
                        target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                        target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                       actions=cur_max_actions_onehot,
                                                       max_q_i=target_max_qvals, is_v=False)
                        target_max_qvals = target_chosen + target_adv
                else:
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            if self.args.MIXER == "dmaq_qatten":
                loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            else:
                loss = (masked_td_error ** 2).sum() / mask.sum()

            additional_loss += loss
        # ---------------------------------------------------------------------------------------------

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss, intrinsic_rewards)

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
