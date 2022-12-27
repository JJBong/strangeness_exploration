import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.cnn_module import CNNModule


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.N_AGENTS
        self.embed_dim = args.MIXING_EMBED_DIM

        self.state_shape = args.STATE_SHAPE

        self.use_cnn_model = False
        if isinstance(args.STATE_SHAPE, tuple):
            self.cnn_module = CNNModule(input_shape=args.STATE_SHAPE, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.state_dim = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True
        else:
            self.state_dim = int(np.prod(args.STATE_SHAPE))

        if getattr(args, "HYPERNET_LAYERS", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "HYPERNET_LAYERS", 1) == 2:
            HYPERNET_EMBED = self.args.HYPERNET_EMBED
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, HYPERNET_EMBED),
                                           nn.ReLU(),
                                           nn.Linear(HYPERNET_EMBED, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, HYPERNET_EMBED),
                                               nn.ReLU(),
                                               nn.Linear(HYPERNET_EMBED, self.embed_dim))
        elif getattr(args, "HYPERNET_LAYERS", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        if self.use_cnn_model:
            states = np.reshape(states, (-1, *self.state_shape))
            states = self.cnn_module(states)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
