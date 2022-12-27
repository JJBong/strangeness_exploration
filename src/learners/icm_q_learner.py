from components.episode_buffer import EpisodeBatch
from learners import QLearner
import torch as th
from torch.optim import RMSprop


class IcmQLearner(QLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.n_agents = args.N_AGENTS

        self.exp_mac = exp_mac
        exp_mac_params = self.exp_mac.parameters()
        self.params += list(exp_mac_params)

        # re-configure optimizer
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.last_target_update_episode = 0

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Exp encoder-decoder tranining
        inputs_prediction_error_list = []
        actions_prediction_error_list = []
        if self.args.ENCODER_RNN:
            self.exp_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            real_next_inputs_feature, pred_next_inputs_feature, pred_action, action_y = self.exp_mac.forward(batch, t=t)
            inputs_prediction_error = self.exp_mac.calc_mse_loss(real_next_inputs_feature, pred_next_inputs_feature)
            actions_prediction_error = self.exp_mac.calc_ce_loss(pred_action, action_y)
            inputs_prediction_error_list.append(inputs_prediction_error.mean())
            actions_prediction_error_list.append(actions_prediction_error.mean())
        inputs_prediction_errors = th.stack(inputs_prediction_error_list)
        actions_prediction_errors = th.stack(actions_prediction_error_list)

        inputs_prediction_loss = inputs_prediction_errors.mean()
        actions_prediction_loss = actions_prediction_errors.mean()

        self.logger.log_stat("icm_inputs_prediction_loss", inputs_prediction_loss.item(), t_env)
        self.logger.log_stat("icm_actions_prediction_loss", actions_prediction_loss.item(), t_env)

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self.exp_mac.load_state()

        additional_loss = inputs_prediction_loss + actions_prediction_loss
        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, additional_loss)

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
