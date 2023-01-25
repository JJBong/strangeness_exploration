import torch.nn as nn


class StateDecoder(nn.Module):
    def __init__(self, args):
        super(StateDecoder, self).__init__()
        self.args = args
        self.n_agents = args.N_AGENTS
        self.state_shape = args.STATE_SHAPE
        self.z_embed_dim = int(args.ENCODER_HIDDEN_DIM / 4)

        self.encoder = nn.Sequential(
            nn.Linear(self.z_embed_dim*self.n_agents, int(args.ENCODER_HIDDEN_DIM / 2)),
            nn.ReLU(),
            nn.Linear(int(args.ENCODER_HIDDEN_DIM / 2), args.ENCODER_HIDDEN_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.ENCODER_HIDDEN_DIM, self.state_shape)#, nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.reshape(-1, self.z_embed_dim*self.n_agents)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded