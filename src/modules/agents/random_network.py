import torch.nn as nn
import numpy as np
from torch.nn import init
from modules.cnn_module import CNNModule


class RandomNetworkModel(nn.Module):
    def __init__(self, input_shape, args):
        super(RandomNetworkModel, self).__init__()
        self.args = args
        self.obs_dim = args.OBS_SHAPE
        self.input_shape = input_shape

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.reshaped_input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.predictor = nn.Sequential(
            nn.Linear(input_shape, args.RANDOM_NETWORK_DIM),
            nn.ReLU(),
            nn.Linear(args.RANDOM_NETWORK_DIM, args.RANDOM_NETWORK_DIM),
            nn.ReLU(),
            nn.Linear(args.RANDOM_NETWORK_DIM, args.RANDOM_NETWORK_DIM)
        )

        self.target = nn.Sequential(
            nn.Linear(input_shape, args.RANDOM_NETWORK_DIM),
            nn.ReLU(),
            nn.Linear(args.RANDOM_NETWORK_DIM, args.RANDOM_NETWORK_DIM),
            nn.ReLU(),
            nn.Linear(args.RANDOM_NETWORK_DIM, args.RANDOM_NETWORK_DIM)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        if self.use_cnn_model:
            inputs = np.reshape(inputs, (-1, *self.input_shape))
            inputs = self.cnn_module(inputs)
            inputs = inputs.reshape(-1, self.reshaped_input_shape)
        else:
            inputs = inputs.reshape(-1, self.input_shape)

        predict_feature = self.predictor(inputs)
        target_feature = self.target(inputs)
        return predict_feature, target_feature
