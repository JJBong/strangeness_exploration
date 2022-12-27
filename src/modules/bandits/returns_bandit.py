# Categorical policy for discrete z

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(args.state_shape, 256)
        self.affine2 = nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, args.noise_dim)
        self.output_scale = self.args.bandit_reward_scaling

    def forward(self, x):
        x = x.view(-1, self.args.state_shape)
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.relu(x)
        returns = self.affine3(x)
        return returns * self.output_scale


class ReturnsBandit:
    def __init__(self, args, logger):
        self.args = args
        self.lr = args.LR
        self.logger = logger
        self.noise_dim = self.args.NOISE_DIM
        # size of state vector
        self.state_shape = self.args.STATE_SHAPE
        self.net = Net(args)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr)

        self.buffer = deque(maxlen=self.args.BANDIT_BUFFER)
        self.epsilon_floor = args.BANDIT_EPSILON

        self.uniform_noise = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1/self.args.NOISE_DIM for _ in range(self.args.NOISE_DIM)]).repeat(self.args.BATCH_SIZE_RUN, 1))

        self.device = torch.device(self.args.DEVICE)

    def sample(self, state, test_mode):
        if test_mode:
            return self.uniform_noise.sample()
        else:
            estimated_returns = self.net(state)
            probs = F.softmax(estimated_returns, dim=-1)
            probs_eps = (1 - self.epsilon_floor) * probs + self.epsilon_floor / self.noise_dim
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs_eps)
            action = m.sample().cpu()
            return action

    def update_returns(self, states, actions, returns, test_mode, t):
        if test_mode:
            return

        for s,a,r in zip(states, actions, returns):
            self.buffer.append((s,a,torch.tensor(r, dtype=torch.float)))

        for _ in range(self.args.BANDIT_ITERS):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.BANDIT_BATCH)
            batch_elems = [self.buffer[i] for i in idxs]
            states_ = torch.stack([x[0] for x in batch_elems]).to(states.device)
            actions_ = torch.stack([x[1] for x in batch_elems]).to(states.device)
            returns_ = torch.stack([x[2] for x in batch_elems]).to(states.device)

            if not self.args.BANDIT_USE_STATE:
                states_ = torch.ones_like(states_)

            estimated_returns_all = self.net(states_)
            estimated_returns = (estimated_returns_all * actions_).sum(dim=1)
            loss = (returns_ - estimated_returns).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Log info about the last iteration
        self.logger.log_stat("bandit_loss", loss.item(), t)
        action_distrib = torch.distributions.OneHotCategorical(F.softmax(estimated_returns_all, dim=1))
        mean_entropy = action_distrib.entropy().mean()
        self.logger.log_stat("bandit_entropy", mean_entropy.item(), t)
        mins = estimated_returns_all.min(dim=1)[0].mean().item()
        maxs = estimated_returns_all.max(dim=1)[0].mean().item()
        means = estimated_returns_all.mean().item()
        self.logger.log_stat("min_returns", mins, t)
        self.logger.log_stat("max_returns", maxs, t)
        self.logger.log_stat("mean_returns", means, t)

    def to_device(self):
        self.net.to(self.device)

    def save_model(self, path):
        torch.save(self.net.state_dict(), "{}/returns_bandit_net.th".format(path))
