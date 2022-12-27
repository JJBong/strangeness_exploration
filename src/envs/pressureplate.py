from envs.multiagentenv import MultiAgentEnv
import numpy as np
import gym
import pressureplate


class PressurePlateWrapper(MultiAgentEnv):
    def __init__(self, map_name, episode_limit=500):
        self.map_name = map_name
        self.episode_limit = episode_limit
        if map_name == 'linear-4p':
            self.env = gym.make('pressureplate-linear-4p-v0')
        elif map_name == 'linear-5p':
            self.env = gym.make('pressureplate-linear-5p-v0')
        elif map_name == 'linear-6p':
            self.env = gym.make('pressureplate-linear-6p-v0')
        else:
            raise NotImplementedError
        all_obs = self.env.reset()
        self.env.close()
        self.n_agents = len(all_obs)
        self.observation_space = all_obs[0].shape[0]
        self.state_space = self.observation_space * self.n_agents
        self.action_space = 5
        self.steps = 0

    def step(self, actions):
        all_obs, rewards, done, info = self.env.step(actions)
        reward = np.sum(rewards) / self.n_agents

        if self.episode_limit - 1 > self.steps:
            done = any(done)
            if done:
                reward += 50.
        else:
            done = True
        self.steps += 1
        return [reward, done, info]

    def get_obs(self):
        all_obs = self.env.get_obs()
        return all_obs

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.observation_space

    def get_state(self):
        all_obs = self.get_obs()
        state = np.concatenate(all_obs, 0)
        return state

    def get_state_size(self):
        return self.state_space

    def get_avail_actions(self):
        return np.array([np.ones(self.get_total_actions()) for _ in range(self.n_agents)])

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions())

    def get_total_actions(self):
        return self.action_space

    def reset(self):
        all_obs = self.env.reset()
        state = np.concatenate(all_obs, 0)
        self.steps = 0
        return all_obs, state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {}
        return stats
