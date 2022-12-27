import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from learners import MAXQLearner
from utils.rl_utils import overrides


class ExpMAXQLearner(MAXQLearner):
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
            self.hidden_central_mac = copy.deepcopy(self.central_mac)
            self.params += list(self.hidden_central_mac.parameters())

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
            self.hidden_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.hidden_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
            chosen_action_qvals = chosen_action_qvals_agents

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

            # Max over target Q-Values
            if self.args.DOUBLE_Q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
                target_max_agent_qvals = th.gather(target_mac_out[:, :], 3, cur_max_actions[:, :]).squeeze(3)
            else:
                raise Exception("Use double q")

            # Central MAC stuff
            central_mac_out = []
            self.hidden_central_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.hidden_central_mac.forward(batch, t=t)
                central_mac_out.append(agent_outs)
            central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
            central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3,
                                                           index=actions.unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                             self.args.CENTRAL_ACTION_EMBED)).squeeze(
                3)  # Remove the last dim

            central_target_mac_out = []
            self.target_central_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_central_mac.forward(batch, t=t)
                central_target_mac_out.append(target_agent_outs)
            central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
            # Mask out unavailable actions
            central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
            # Use the Qmix max actions
            central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
                                                       cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                 self.args.CENTRAL_ACTION_EMBED)).squeeze(
                3)
            # ---

            # Mix
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:, 1:], batch["state"][:, 1:])

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals

            # Td-error
            td_error = (chosen_action_qvals - (targets.detach()))

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Training central Q
            central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
            central_td_error = (central_chosen_action_qvals - targets.detach())
            central_mask = mask.expand_as(central_td_error)
            central_masked_td_error = central_td_error * central_mask
            central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

            # QMIX loss with weighting
            ws = th.ones_like(td_error) * self.args.W
            if self.args.HYSTERETIC_QMIX:  # OW-QMIX
                ws = th.where(td_error < 0, th.ones_like(td_error) * 1, ws)  # Target is greater than current max
                w_to_use = ws.mean().item()  # For logging
            else:  # CW-QMIX
                is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
                max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
                qtot_larger = targets > max_action_qtot
                ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error) * 1,
                              ws)  # Target is greater than current max
                w_to_use = ws.mean().item()  # Average of ws for logging

            qmix_loss = (ws.detach() * (masked_td_error ** 2)).sum() / mask.sum()

            # The weightings for the different losses aren't used (they are always set to 1)
            loss = self.args.QMIX_LOSS * qmix_loss + self.args.CENTRAL_LOSS * central_loss

            additional_loss += loss
        # ---------------------------------------------------------------------------------------------

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss, intrinsic_rewards)

    @overrides(MAXQLearner)
    def _update_targets(self):
        if self.args.HIDDEN_POLICY:
            self.target_mac.load_state(self.hidden_mac)
        else:
            self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.central_mac is not None:
            if self.args.HIDDEN_POLICY:
                self.target_central_mac.load_state(self.hidden_central_mac)
            else:
                self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()
        if self.args.HIDDEN_POLICY:
            self.hidden_mac.to_device()
            self.hidden_central_mac.to_device()

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
