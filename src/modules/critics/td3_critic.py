import torch.nn as nn
import torch.nn.functional as F


class TD3Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(TD3Critic, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.CRITIC_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.CRITIC_HIDDEN_DIM, args.CRITIC_HIDDEN_DIM)
        self.fc3 = nn.Linear(args.CRITIC_HIDDEN_DIM, 1)

    def forward(self, inputs):
        q = F.relu(self.fc1(inputs))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class DuelingTD3Critic(TD3Critic):
    def __init__(self, input_shape, args, output_shape=None):
        super().__init__(input_shape, args)

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.value_fc = nn.Linear(args.CRITIC_HIDDEN_DIM, args.CRITIC_HIDDEN_DIM)
        self.value = self.fc3

        self.adv_fc = nn.Linear(args.CRITIC_HIDDEN_DIM, args.CRITIC_HIDDEN_DIM)
        self.adv = nn.Linear(args.CRITIC_HIDDEN_DIM, output_shape)

    def forward(self, inputs, actions):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))

        v = self.value_fc(x)
        v = self.value(v)

        a = self.adv_fc(x)
        a = self.adv(a)

        a_mean = a.mean(-1, keepdim=True)
        a *= actions
        a -= a_mean
        a = a.sum(-1, keepdim=True)
        q = v + a
        return q, a
