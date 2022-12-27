from envs.multiagentenv import MultiAgentEnv
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.sisl import pursuit_v4
import numpy as np
from functools import reduce
import operator


class PettingZoo(MultiAgentEnv):
    # render_mode: (None, 'human')
    def __init__(self, map_name, max_cycles, use_linear_input=True, seed=None, continuous=False, render_mode=None):
        self.map_name = map_name
        self.seed = seed
        self.episode_limit = max_cycles
        self.use_linear_input = use_linear_input
        if map_name == 'pistonball':
            self.env = pistonball_v6.env(continuous=continuous, render_mode=render_mode, max_cycles=max_cycles)
        elif map_name == 'cooperative_pong':
            self.env = cooperative_pong_v5.env(render_mode=render_mode, max_cycles=max_cycles)
        elif map_name == 'simple_spread':
            self.env = simple_spread_v2.env(continuous_actions=continuous, render_mode=render_mode, max_cycles=max_cycles, local_ratio=0.5, N=5)
        elif map_name == 'pursuit':
            self.env = pursuit_v4.env(render_mode=render_mode, max_cycles=max_cycles, x_size=16, y_size=16, shared_reward=True, n_evaders=30, n_pursuers=8, obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01, catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
        else:
            raise NotImplementedError

        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        if map_name == 'pursuit':
            pass
        else:
            self.state_space = self.env.state_space
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space = self.action_spaces[self.agents[0]]

    def step(self, actions):
        for agent, action in zip(self.agents, actions):
            _, reward, termination, truncation, info = self.env.last()
            action = None if termination or truncation else action
            self.env.step(action)
        done = termination or truncation
        return [reward, done, info]

    def get_obs(self):
        observations = []
        for agent in self.agents:
            observation = self.env.observe(agent)
            if self.map_name != 'simple_spread':
                observation = np.moveaxis(observation, 2, 0) / 255.0
                if self.use_linear_input:
                    observation = rgb2gray(observation)
                    observation = np.reshape(observation, (-1))
            elif self.map_name == 'pursuit':
                observation = np.moveaxis(observation, 2, 0)
                if self.use_linear_input:
                    observation = np.reshape(observation, (-1))
            observations.append(observation)
        return observations

    def get_obs_agent(self, agent_id):
        observation = self.env.observe(self.agents[agent_id])
        if self.map_name != 'simple_spread':
            observation = np.moveaxis(observation, 2, 0) / 255.0
            if self.use_linear_input:
                observation = rgb2gray(observation)
                observation = np.reshape(observation, (-1))
        elif self.map_name == 'pursuit':
            observation = np.moveaxis(observation, 2, 0)
            if self.use_linear_input:
                observation = np.reshape(observation, (-1))
        return observation

    def get_obs_size(self):
        if self.map_name != 'simple_spread':
            observation_shape = list(self.observation_space.shape)
            observation_shape.insert(0, observation_shape.pop(-1))
            observation_shape = tuple(observation_shape)
            if self.use_linear_input:
                observation_shape = reduce(operator.mul, observation_shape[1:], 1)
        else:
            observation_shape = self.observation_space.shape[0]
        return observation_shape

    def get_state(self):
        if self.map_name == 'pursuit':
            all_obs = self.get_obs()
            state = np.concatenate(all_obs, axis=0)
        elif self.map_name != 'simple_spread':
            state = np.moveaxis(self.env.state(), 2, 0) / 255.0
            if self.use_linear_input:
                state = rgb2gray(state)
                state = np.reshape(state, (-1))
        else:
            state = self.env.state()
        return state

    def get_state_size(self):
        if self.map_name == 'pursuit':
            if self.use_linear_input:
                return self.get_obs_size() * self.n_agents
            else:
                all_obs_size = self.get_obs_size()
                all_obs_size[0] *= self.n_agents
                return all_obs_size
        if self.map_name != 'simple_spread':
            state_shape = list(self.state_space.shape)
            state_shape.insert(0, state_shape.pop(-1))
            state_shape = tuple(state_shape)
            if self.use_linear_input:
                state_shape = reduce(operator.mul, state_shape[1:], 1)
        else:
            state_shape = self.state_space.shape[0]
        return state_shape

    def get_avail_actions(self):
        return np.array([np.ones(self.get_total_actions()) for _ in range(self.n_agents)])

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions())

    def get_total_actions(self):
        return self.action_space.n

    def reset(self):
        self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        self.env.seed(self.seed)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit + 1}
        return env_info

    def get_stats(self):
        stats = {}
        return stats


def rgb2gray(rgb):

    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
