import torch.nn as nn
import numpy as np
from modules.cnn_module import CNNModule


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.reshaped_input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, args.ENCODER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(args.ENCODER_HIDDEN_DIM, int(args.ENCODER_HIDDEN_DIM / 2)),
            nn.ReLU(),
            nn.Linear(int(args.ENCODER_HIDDEN_DIM / 2), int(args.ENCODER_HIDDEN_DIM / 4))
        )

        self.z_embed_dim = int(args.ENCODER_HIDDEN_DIM / 4)

        if args.ENCODER_RNN:
            self.rnn = nn.GRUCell(self.z_embed_dim, self.z_embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_embed_dim, int(args.ENCODER_HIDDEN_DIM / 2)),
            nn.ReLU(),
            nn.Linear(int(args.ENCODER_HIDDEN_DIM / 2), args.ENCODER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(args.ENCODER_HIDDEN_DIM, input_shape),
            # nn.Sigmoid()
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder[-1].weight.new(1, self.z_embed_dim).zero_()

    def forward(self, inputs, hidden=None):
        if self.use_cnn_model:
            inputs = np.reshape(inputs, (-1, *self.input_shape))
            inputs = self.cnn_module(inputs)
            inputs = inputs.reshape(-1, self.reshaped_input_shape)
        else:
            inputs = inputs.reshape(-1, self.input_shape)

        encoded = self.encoder(inputs)

        if self.args.ENCODER_RNN:
            h_in = hidden.reshape(-1, self.z_embed_dim)
            encoded = self.rnn(encoded, h_in)

        decoded = self.decoder(encoded)
        return encoded, decoded
