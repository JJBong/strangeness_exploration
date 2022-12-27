import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class SemiEMCQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.ext_mac = copy.deepcopy(mac)
        self.logger = logger

        self.params = list(mac.parameters())
        ext_params = list(self.ext_mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        self.ext_mixer = None
        if args.MIXER is not None:
            if args.MIXER == "vdn":
                self.mixer = VDNMixer()
                self.ext_mixer = VDNMixer()
            elif args.MIXER == "qmix":
                self.mixer = QMixer(args)
                self.ext_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.MIXER))
            self.params += list(self.mixer.parameters())
            ext_params += list(self.ext_mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.target_ext_mixer = copy.deepcopy(self.ext_mixer)

        # make predictor
        self.predictor_mac = copy.deepcopy(mac)
        self.predictor_params = list(self.predictor_mac.parameters())

        self.optimizer = RMSprop(params=self.params + ext_params + self.predictor_params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # ext_mac
        self.target_ext_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self._train(batch, t_env, episode_num)

    def _train(self, batch, t_env, episode_num):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values and Build Predictor' Q-Values
        mac_out = []
        ext_mac_out = []
        predictor_mac_out = []
        self.mac.init_hidden(batch.batch_size)
        self.ext_mac.init_hidden(batch.batch_size)
        self.predictor_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            ext_agent_outs = self.ext_mac.forward(batch, t=t)
            ext_mac_out.append(ext_agent_outs)
            predictor_agent_outs = self.predictor_mac.forward(batch, t=t)
            predictor_mac_out.append(predictor_agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        ext_mac_out = th.stack(ext_mac_out, dim=1)  # Concat over time
        predictor_mac_out = th.stack(predictor_mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        ext_chosen_action_qvals = th.gather(ext_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        predictor_chosen_action_qvals = th.gather(predictor_mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_ext_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.target_ext_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_ext_agent_outs = self.target_ext_mac.forward(batch, t=t)
            target_ext_mac_out.append(target_ext_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        _target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        _target_ext_mac_out = th.stack(target_ext_mac_out[1:], dim=1)  # Concat across time

        # Make targets for target of predictor
        predictor_target_mac_out = th.stack(target_ext_mac_out[:-1], dim=1)  # Concat across time
        predictor_chosen_target_action_qvals = th.gather(predictor_target_mac_out, dim=3, index=actions).squeeze(3)

        # Mask out unavailable actions
        _target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        _target_ext_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.DOUBLE_Q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            ext_mac_out_detach = ext_mac_out.clone().detach()
            ext_mac_out_detach[avail_actions == 0] = -9999999
            ext_cur_max_actions = ext_mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(_target_mac_out, 3, cur_max_actions).squeeze(3)
            target_ext_max_qvals = th.gather(_target_ext_mac_out, 3, ext_cur_max_actions).squeeze(3)
        else:
            target_max_qvals = _target_mac_out.max(dim=3)[0]
            target_ext_max_qvals = _target_ext_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            ext_chosen_action_qvals = self.ext_mixer(ext_chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            target_ext_max_qvals = self.target_ext_mixer(target_ext_max_qvals, batch["state"][:, 1:])

        # Calculate prediction loss and build intrinsic reward with predictor
        predictor_chosen_action_qvals = predictor_chosen_action_qvals.sum(2, keepdim=True)
        predictor_chosen_target_action_qvals = predictor_chosen_target_action_qvals.sum(2, keepdim=True)
        predictor_td_error = predictor_chosen_action_qvals - predictor_chosen_target_action_qvals.detach()
        predictor_td_error = (predictor_td_error ** 2).mean(2, keepdim=True)
        intrinsic_rewards = self.args.RHO * predictor_td_error.clone().detach()

        predictor_loss = predictor_td_error.mean()

        # Calculate 1-step Q-Learning targets
        targets = (rewards + intrinsic_rewards.detach()) + self.args.GAMMA * (1 - terminated) * target_max_qvals

        # Td-error -------------------------------------------
        td_error = (chosen_action_qvals - targets.detach())

        _mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * _mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / _mask.sum()
        # ----------------------------------------------------

        # Ext Calculate 1-step Q-Learning targets
        ext_targets = rewards + self.args.GAMMA * (1 - terminated) * target_ext_max_qvals

        # Ext td-error ---------------------------------------
        ext_td_error = (ext_chosen_action_qvals - ext_targets.detach())

        _mask = mask.expand_as(ext_td_error)

        # 0-out the targets that came from padded data
        ext_masked_td_error = ext_td_error * _mask

        # Normal L2 loss, take mean over actual data
        ext_loss = (ext_masked_td_error ** 2).sum() / _mask.sum()
        # ----------------------------------------------------

        total_loss = loss + ext_loss + predictor_loss

        # Optimise
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        # Ext mac - update target
        for target_param, local_param in zip(self.target_ext_mac.parameters(), self.ext_mac.parameters()):
            target_param.data.copy_(self.args.SOFT_UPDATE_TAU * local_param.data + (1.0 - self.args.SOFT_UPDATE_TAU) * target_param.data)

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("predictor_loss", predictor_loss.item(), t_env)
            self.logger.log_stat("loss", total_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        self.mac.to_device()
        self.ext_mac.to_device()
        self.predictor_mac.to_device()
        self.target_mac.to_device()
        self.target_ext_mac.to_device()
        if self.mixer is not None:
            self.mixer.to(device=self.mac.device)
            self.target_mixer.to(device=self.mac.device)
            self.ext_mixer.to(device=self.mac.device)
            self.target_ext_mixer.to(device=self.mac.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
