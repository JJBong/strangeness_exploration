import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTranBase
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.MIXER == "qtran_base":
            self.mixer = QTranBase(args)
        elif args.MIXER == "qtran_alt":
            raise Exception("Not implemented here!")

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self._train(batch, t_env, episode_num)

    def _train(self, batch: EpisodeBatch, t_env: int, episode_num: int, additional_loss=None, intrinsic_rewards=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        if intrinsic_rewards is not None:
            rewards += intrinsic_rewards
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.N_AGENTS, batch.max_seq_length, -1).transpose(1,2) #btav

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
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.N_AGENTS, batch.max_seq_length, -1).transpose(1,2) #btav

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
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS), device=batch.device)
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS), device=batch.device)
                max_actions_onehot = max_actions.scatter(3, target_max_actions[:, :], 1)
            target_joint_qs, target_vs = self.target_mixer(batch[:, 1:], hidden_states=target_mac_hidden_states[:,1:], actions=max_actions_onehot[:,1:])

            # Td loss targets
            td_targets = rewards.reshape(-1,1) + self.args.GAMMA * (1 - terminated.reshape(-1, 1)) * target_joint_qs
            td_error = (joint_qs - td_targets.detach())
            masked_td_error = td_error * mask.reshape(-1, 1)
            td_loss = (masked_td_error ** 2).sum() / mask.sum()
            # -- TD Loss --

            # -- Opt Loss --
            # Argmax across the current agents' actions
            if not self.args.DOUBLE_Q: # Already computed if we're doing double Q-Learning
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.N_AGENTS, self.args.N_ACTIONS), device=batch.device)
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
            max_joint_qs, _ = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1], actions=max_actions_current_onehot[:,:-1]) # Don't use the target network and target agent max actions as per author's email

            # max_actions_qvals = th.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
            opt_error = max_actions_qvals[:,:-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
            masked_opt_error = opt_error * mask.reshape(-1, 1)
            opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
            # -- Opt Loss --

            # -- Nopt Loss --
            # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
            nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs # Don't use target networks here either
            nopt_error = nopt_values.clamp(max=0)
            masked_nopt_error = nopt_error * mask.reshape(-1, 1)
            nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
            # -- Nopt loss --

        elif self.args.MIXER == "qtran_alt":
            raise Exception("Not supported yet.")

        loss = td_loss + self.args.OPT_LOSS * opt_loss + self.args.NOPT_MIN_LOSS * nopt_loss

        # Optimize
        self.optimizer.zero_grad()
        if additional_loss is not None:
            loss += additional_loss
        loss.backward(retain_graph=True)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if self.args.MIXER == "qtran_base":
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_targets", ((masked_td_error).sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_chosen_qs", (joint_qs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("v_mean", (vs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("agent_indiv_qs", ((chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.N_AGENTS)), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        self.mac.to_device()
        self.target_mac.to_device()
        if self.mixer is not None:
            self.mixer.to(device=self.mac.device)
            self.target_mixer.to(device=self.mac.device)

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
