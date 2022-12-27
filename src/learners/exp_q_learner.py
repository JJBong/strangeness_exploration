import copy

from components.episode_buffer import EpisodeBatch
from learners import QLearner
import torch as th
from torch.optim import RMSprop
from utils.rl_utils import overrides
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


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

            self.hidden_mixer = None
            if args.MIXER is not None:
                if args.MIXER == "vdn":
                    self.hidden_mixer = VDNMixer()
                elif args.MIXER == "qmix":
                    self.hidden_mixer = QMixer(args)
                else:
                    raise ValueError("Mixer {} not recognised.".format(args.MIXER))
                self.params += list(self.hidden_mixer.parameters())

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
            exp_prediction_error, exp_state_decoding_error = self.exp_mac.calc_mse_loss(inputs_decoded, target_y, state_decoded, state_y)
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
            hidden_mac_out = []
            self.hidden_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.hidden_mac.forward(batch, t=t)
                hidden_mac_out.append(agent_outs)
            hidden_mac_out = th.stack(hidden_mac_out, dim=1)  # Concat over time

            # # Pick the Q-Values for the actions taken by each agent
            hidden_chosen_action_qvals = th.gather(hidden_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

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
                mac_out_detach = hidden_mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            # Mix
            if self.mixer is not None:
                hidden_chosen_action_qvals = self.hidden_mixer(hidden_chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals

            # Td-error
            td_error = (hidden_chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            hidden_loss = (masked_td_error ** 2).sum() / mask.sum()

            additional_loss = hidden_loss

            if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
                mask_elems = mask.sum().item()
                self.logger.log_stat("hidden_q_taken_mean", (hidden_chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
        # ---------------------------------------------------------------------------------------------

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss, intrinsic_rewards)

    @overrides(QLearner)
    def _update_targets(self):
        if self.args.HIDDEN_POLICY:
            self.target_mac.load_state(self.hidden_mac)
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.hidden_mixer.state_dict())
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
        if self.args.HIDDEN_POLICY:
            self.hidden_mac.save_models_for_hidden(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
        if self.args.HIDDEN_POLICY:
            self.hidden_mac.load_models_for_hidden(path)
