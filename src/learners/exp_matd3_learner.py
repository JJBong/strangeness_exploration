import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam, RMSprop
from learners.matd3_learner import MATd3Learner
from utils.rl_utils import overrides


class ExpMATd3Learner(MATd3Learner):
    def __init__(self, mac, exp_mac, scheme, logger, args, hidden_mac=None):
        super().__init__(mac, scheme, logger, args)
        self.n_agents = args.N_AGENTS

        self.exp_mac = exp_mac
        self.exp_mac_params = self.exp_mac.parameters()

        if args.HIDDEN_POLICY:
            self.hidden_critic1 = copy.deepcopy(self.critic1)
            self.hidden_critic2 = copy.deepcopy(self.critic2)
            self.critic1_params += list(self.hidden_critic1.parameters())
            self.critic1_params += list(self.hidden_critic2.parameters())

        # re-configure optimizer
        self.critic1_optimizer = Adam(self.critic1_params, lr=args.CRITIC_LR)

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
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

            # -------------------- Critic Update ------------------------------------------------------------
            with th.no_grad():
                # Calculate the Q-Values necessary for the target
                target_critic_inputs = []
                self.target_mac.init_hidden(batch.batch_size)
                # target_mac greedy action selector' epsilon update
                self.target_mac.action_selector.epsilon = self.target_mac.action_selector.schedule.eval(t_env)
                for t in range(batch.max_seq_length):
                    # target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_agent_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)

                    # target noise
                    target_noise = th.normal(
                        0, self.target_noise,
                        size=target_agent_outs.shape
                    ).clamp(-self.target_noise_clip, self.target_noise_clip).to(device=self.device)
                    target_agent_outs = (
                            target_agent_outs + target_noise
                    ).clamp(-self.args.MAX_ACTION, self.args.MAX_ACTION)

                    target_critic_input = self._build_critic_inputs(batch, t=t, actions=target_agent_outs)
                    target_critic_inputs.append(target_critic_input)
                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_critic_inputs = th.stack(target_critic_inputs[1:], dim=1)  # Concat across time
                target_Q1 = self.target_critic1(target_critic_inputs)
                target_Q2 = self.target_critic2(target_critic_inputs)
                target_Q_detached = th.min(target_Q1, target_Q2).squeeze(-1)

            critic_inputs = []
            for t in range(batch.max_seq_length):
                critic_input = self._build_critic_inputs(batch, t=t)
                critic_inputs.append(critic_input)
            critic_inputs = th.stack(critic_inputs[:-1], dim=1)  # Concat across time
            current_Q1 = self.hidden_critic1(critic_inputs)
            current_Q2 = self.hidden_critic2(critic_inputs)

            # Mix
            mixed_qvals1 = self.mixer(current_Q1, batch["state"][:, :-1])
            mixed_qvals2 = self.mixer(current_Q2, batch["state"][:, :-1])
            target_q_vals = self.target_mixer(target_Q_detached, batch["state"][:, 1:])

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.GAMMA * (1 - terminated) * target_q_vals

            # Td-error
            td_error1 = (mixed_qvals1 - targets.detach())
            td_error2 = (mixed_qvals2 - targets.detach())

            mask1 = mask.expand_as(td_error1)
            mask2 = mask.expand_as(td_error2)

            # 0-out the targets that came from padded data
            masked_td_error1 = td_error1 * mask1
            masked_td_error2 = td_error2 * mask2

            # Normal L2 loss, take mean over actual data
            critic_loss1 = (masked_td_error1 ** 2).sum() / mask1.sum()
            critic_loss2 = (masked_td_error2 ** 2).sum() / mask2.sum()

            critic_loss = critic_loss1 + critic_loss2
            additional_loss += critic_loss
        # ---------------------------------------------------------------------------------------------

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss, intrinsic_rewards)

    @overrides(MATd3Learner)
    def _update_targets(self):
        if self.args.HIDDEN_POLICY:
            self.target_mac.load_state(self.hidden_mac)
        else:
            self.target_mac.load_state(self.mac)
        if self.args.HIDDEN_POLICY:
            self.target_critic1.load_state_dict(self.hidden_critic1.state_dict())
            self.target_critic2.load_state_dict(self.hidden_critic2.state_dict())
        else:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()
        if self.args.HIDDEN_POLICY:
            self.hidden_critic1.to(device=self.mac.device)
            self.hidden_critic2.to(device=self.mac.device)

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
