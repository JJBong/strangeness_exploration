# Categorical policy for discrete z

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(args.STATE_SHAPE, 128)
        self.affine2 = nn.Linear(128, args.NOISE_DIM)

    def forward(self, x):
        x = x.view(-1, self.args.STATE_SHAPE)
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Z_agent:
    def __init__(self, args):
        self.args = args
        self.lr = args.LR
        self.noise_dim = self.args.NOISE_DIM
        # size of state vector
        self.state_shape = self.args.STATE_SHAPE
        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.device = torch.device(self.args.DEVICE)

    def sample(self, state):
        probs = self.policy(state)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
        action = m.sample()
        return action

    def update_returns(self, states, actions, returns, test_mode):
        if test_mode:
            return
        probs = self.policy(states)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
        log_probs = m.log_prob(actions)
        self.optimizer.zero_grad()
        policy_loss = -torch.dot(log_probs, returns)
        policy_loss.backward()
        self.optimizer.step()

    def to_device(self):
        self.policy.to(self.device)


# Max entropy Z agent
class EZ_agent:
    def __init__(self, args, logger):
        self.args = args
        self.lr = args.LR
        self.noise_dim = self.args.NOISE_DIM
        # size of state vector
        self.state_shape = self.args.STATE_SHAPE
        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        # Scaling factor for entropy, would roughly be similar to MI scaling
        self.entropy_scaling = args.ENTROPY_SCALING
        self.uniform_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1/self.args.NOISE_DIM for _ in range(self.args.NOISE_DIM)]).repeat(self.args.BATCH_SIZE_RUN, 1))

        self.buffer = deque(maxlen=self.args.BANDIT_BUFFER)
        self.epsilon_floor = args.BANDIT_EPSILON

        self.logger = logger

        self.device = torch.device(self.args.DEVICE)

    def sample(self, state, test_mode):
        # During testing we just sample uniformly
        if test_mode:
            return self.uniform_distrib.sample()
        else:
            probs = self.policy(state)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
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

            probs = self.policy(states_)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            log_probs = m.log_prob(actions_.to(probs.device))
            self.optimizer.zero_grad()
            policy_loss = -torch.dot(log_probs, torch.tensor(returns_, device=log_probs.device).float()) + self.entropy_scaling * log_probs.sum()
            policy_loss.backward()
            self.optimizer.step()

        mean_entropy = m.entropy().mean()
        self.logger.log_stat("bandit_entropy", mean_entropy.item(), t)

    def to_device(self):
        self.policy.to(self.device)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), "{}/ez_bandit_policy.th".format(path))
