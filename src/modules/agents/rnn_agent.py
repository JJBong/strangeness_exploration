import torch.nn as nn
import torch.nn.functional as F
from modules.cnn_module import CNNModule


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args, output_shape=None):
        super(RNNAgent, self).__init__()
        self.args = args

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_cnn_model:
            inputs = self.cnn_module(inputs)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        if self.args.ACTION_SPACE == 'continuous':
            q = F.tanh(q)
        return q, h
