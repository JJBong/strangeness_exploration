from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from learners import QLearner


class RNDQLearner(QLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.n_agents = args.N_AGENTS

        self.exp_mac = exp_mac
        self.params += list(self.exp_mac.parameters())
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)
        self.last_target_update_episode = 0

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Exp encoder-decoder tranining
        obs_prediction_error_list = []
        for t in range(batch.max_seq_length):
            if t == 0:
                continue
            predict_features, target_features = self.exp_mac.forward(batch, t=t)
            obs_prediction_error = self.exp_mac.calc_mse_loss(predict_features, target_features)
            obs_prediction_error_list.append(obs_prediction_error)
        obs_prediction_errors = th.stack(obs_prediction_error_list)

        obs_prediction_loss = obs_prediction_errors.mean()

        self.logger.log_stat("rnd_loss", obs_prediction_loss.item(), t_env)

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, obs_prediction_loss)

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
