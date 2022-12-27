import numpy as np
from envs.multiagentenv import MultiAgentEnv


# K-step payoff matrix game
# K represents a max length of an episode
class PayOffMatrix(MultiAgentEnv):
    def __init__(self, map_name='k_step', k=64):
        assert map_name in ['k_step']
        self.map_name = map_name
        self.k = k
        self.state_num = 0
        self.info = {'n_episodes': 0, 'ep_length': self.state_num}
        if self.map_name == 'k_step':
            value_list = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            self.n_agents = 2
            self.episode_limit = self.k
            self.path_len = 4
            self.N = 2
            self.n_actions = 2
            self.matrix = np.array(value_list).reshape(4, self.N, self.N)
            self.states = np.eye(self.k)
            self.matrix_index = 0
            self.done = False

    def reset(self):
        self.state_num = 0
        self.matrix_index = 0
        self.done = False
        return [self.states[self.state_num], self.states[self.state_num]], self.states[self.state_num]

    def step(self, actions):
        reward = 0.
        if self.map_name == 'k_step':
            reward = self.matrix[self.matrix_index][actions[0]][actions[1]]

            if self.state_num >= self.episode_limit - 1:
                self.done = True
            else:
                if reward > 0:
                    self.state_num += 1
                    self.matrix_index = self.state_num % self.path_len
                else:
                    self.done = True
            self.info['ep_length'] = self.state_num

        return [reward, self.done, self.info]

    def get_obs(self):
        return [self.states[self.state_num], self.states[self.state_num]]

    def get_state(self):
        return self.states[self.state_num]

    def get_avail_actions(self):
        return np.array([np.ones(self.get_total_actions()) for _ in range(self.n_agents)])

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions())

    def get_state_size(self):
        if self.map_name == 'k_step':
            return len(self.states)

    def get_obs_size(self):
        if self.map_name == 'k_step':
            return len(self.states)

    def get_total_actions(self):
        if self.map_name == 'k_step':
            return self.n_actions

    def get_stats(self):
        stats = {}
        return stats

    def close(self):
        pass