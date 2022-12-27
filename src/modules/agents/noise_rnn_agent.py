import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, args.N_ACTIONS)

        self.noise_fc1 = nn.Linear(args.NOISE_DIM + args.N_AGENTS, args.NOISE_EMBEDDING_DIM)
        self.noise_fc2 = nn.Linear(args.NOISE_EMBEDDING_DIM, args.NOISE_EMBEDDING_DIM)
        self.noise_fc3 = nn.Linear(args.NOISE_EMBEDDING_DIM, args.N_ACTIONS)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.NOISE_DIM + args.N_AGENTS, args.RNN_HIDDEN_DIM * args.N_ACTIONS)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state, noise):
        agent_ids = th.eye(self.args.N_AGENTS, device=inputs.device).repeat(noise.shape[0], 1)
        noise_repeated = noise.repeat(1, self.args.N_AGENTS).reshape(agent_ids.shape[0], -1)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.args.N_ACTIONS, self.args.RNN_HIDDEN_DIM)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, h
