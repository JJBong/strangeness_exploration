import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.cnn_module import CNNModule


class CentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.reshaped_input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, args.N_ACTIONS * args.CENTRAL_ACTION_EMBED)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_cnn_model:
            inputs = np.reshape(inputs, (-1, *self.input_shape))
            inputs = self.cnn_module(inputs)
            inputs = inputs.reshape(-1, self.reshaped_input_shape)
        else:
            inputs = inputs.reshape(-1, self.input_shape)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        q = q.reshape(-1, self.args.N_ACTIONS, self.args.CENTRAL_ACTION_EMBED)
        return q, h
