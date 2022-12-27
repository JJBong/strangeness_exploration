import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import Adam
from modules.critics.td3_critic import TD3Critic


class MATd3Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.logger = logger

        self.n_agents = args.N_AGENTS

        self.mac = mac
        self.agent_params = list(mac.parameters())
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.target_noise = 0.2
        self.target_noise_clip = 0.5

        self.last_target_update_episode = 0

        critic_input_shape = self._get_critic_input_shape(scheme)
        self.critic1 = TD3Critic(critic_input_shape, args)
        self.critic2 = TD3Critic(critic_input_shape, args)
        self.critic1_params = list(self.critic1.parameters())
        self.critic2_params = list(self.critic2.parameters())
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        if args.MIXER == "vdn":
            self.mixer = VDNMixer()
        elif args.MIXER == "qmix":
            self.mixer = QMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.MIXER))
        self.mixer_params = list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_optimizer = Adam(self.agent_params, lr=args.AGENT_LR)
        self.critic1_optimizer = Adam(self.critic1_params, lr=args.CRITIC_LR)
        self.critic2_optimizer = Adam(self.critic2_params, lr=args.CRITIC_LR)
        self.mixer_optimizer = Adam(self.mixer_params, lr=args.CRITIC_LR)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

        self.total_it = 0

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.total_it += 1
        with th.autograd.set_detect_anomaly(True):
            self._train(batch, t_env, episode_num)

    def _train(self, batch, t_env, episode_num, additional_loss=None, intrinsic_rewards=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        if intrinsic_rewards is not None:
            rewards += intrinsic_rewards
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
        current_Q1 = self.critic1(critic_inputs)
        current_Q2 = self.critic2(critic_inputs)

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

        td_error = (targets - mixed_qvals1)

        critic_loss = critic_loss1 + critic_loss2

        # Optimise
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.mixer_optimizer.zero_grad()
        if additional_loss is not None:
            critic_loss += additional_loss
        critic_loss.backward()
        critic_grad_norm1 = th.nn.utils.clip_grad_norm_(self.critic1_params, self.args.GRAD_NORM_CLIP)
        critic_grad_norm2 = th.nn.utils.clip_grad_norm_(self.critic2_params, self.args.GRAD_NORM_CLIP)
        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.GRAD_NORM_CLIP)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        self.mixer_optimizer.step()

        # -------------------- Agent Update ------------------------------------------------------------
        if self.total_it % self.args.AGENT_TRAIN_FREQ == 0:
            critic_inputs = []
            _agent_outs = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                # agent_outs = self.mac.forward(batch, t=t)
                agent_outs = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
                _agent_outs.append(agent_outs)
                critic_input = self._build_critic_inputs(batch, t=t, actions=agent_outs)
                critic_inputs.append(critic_input)
            critic_inputs = th.stack(critic_inputs[:-1], dim=1)  # Concat across time
            _agent_outs = th.stack(_agent_outs[:-1], dim=1)  # Concat across time
            Q1 = self.critic1(critic_inputs)
            mixed_qvals1 = self.mixer(Q1, batch["state"][:, :-1])
            agent_loss = -mixed_qvals1.mean()

            self.agent_optimizer.zero_grad()
            agent_loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.GRAD_NORM_CLIP)
            self.agent_optimizer.step()

            self.logger.log_stat("agent_loss", agent_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm1", critic_grad_norm1, t_env)
            self.logger.log_stat("critic_grad_norm2", critic_grad_norm2, t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (mixed_qvals1 * mask).sum().item() / (mask_elems * self.args.N_AGENTS),
                                 t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
            self.log_stats_t = t_env

    def _build_critic_inputs(self, batch, t, actions=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if actions is not None:
            inputs.append(actions)
        else:
            inputs.append(batch["actions"][:, t])
        if self.args.OBS_AGENT_ID:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def _get_critic_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["actions"]["vshape"]
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents
        return input_shape

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        self.mac.to_device()
        self.target_mac.to_device()
        self.critic1.to(device=self.mac.device)
        self.critic2.to(device=self.mac.device)
        self.target_critic1.to(device=self.mac.device)
        self.target_critic2.to(device=self.mac.device)
        self.mixer.to(device=self.mac.device)
        self.target_mixer.to(device=self.mac.device)


    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimizer.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic1_optimizer.state_dict(), "{}/critic1_opt.th".format(path))
        th.save(self.critic2_optimizer.state_dict(), "{}/critic2_opt.th".format(path))
        th.save(self.mixer_optimizer.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mac.load_models(path)
        self.target_critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimizer.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic1_optimizer.load_state_dict(th.load("{}/critic1_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2_optimizer.load_state_dict(th.load("{}/critic2_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimizer.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))