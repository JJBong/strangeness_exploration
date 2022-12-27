import torch as th
import torch.nn as nn
import numpy as np
from modules.cnn_module import CNNModule


class ICMModule(nn.Module):
    def __init__(self, input_shape, args):
        super(ICMModule, self).__init__()
        self.args = args
        self.obs_dim = args.OBS_SHAPE
        self.input_shape = input_shape
        self.n_agents = args.N_AGENTS
        self.n_actions = args.N_ACTIONS

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.reshaped_input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.feature = nn.Sequential(
            nn.Linear(input_shape, args.FEATURE_HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Linear(args.FEATURE_HIDDEN_DIM, args.FEATURE_HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Linear(args.FEATURE_HIDDEN_DIM, args.FEATURE_HIDDEN_DIM)
        )

        self.feature_z_dim = args.FEATURE_HIDDEN_DIM

        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_z_dim * 2, args.FEATURE_HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Linear(args.FEATURE_HIDDEN_DIM, self.n_actions)
        )

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.n_actions + self.feature_z_dim, self.feature_z_dim),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.n_actions + self.feature_z_dim, self.feature_z_dim),
        )

    def forward(self, inputs, next_inputs, actions):
        if self.use_cnn_model:
            inputs = np.reshape(inputs, (-1, *self.input_shape))
            inputs = self.cnn_module(inputs)
            inputs = inputs.reshape(-1, self.reshaped_input_shape)

        encode_inputs = self.feature(inputs)
        encoded_next_inputs = self.feature(next_inputs)

        pred_action = th.cat((encode_inputs, encoded_next_inputs), 2)
        pred_action = self.inverse_net(pred_action)

        pred_next_inputs_feature_orig = th.cat((encode_inputs, actions), 2)
        pred_next_inputs_feature_orig = self.forward_net_1(pred_next_inputs_feature_orig)

        pred_next_inputs_feature = self.forward_net_2(th.cat((pred_next_inputs_feature_orig, actions), 2))

        real_next_inputs_feature = encoded_next_inputs
        return real_next_inputs_feature, pred_next_inputs_feature, pred_action
