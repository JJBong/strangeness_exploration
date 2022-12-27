from envs.multiagentenv import MultiAgentEnv
import numpy as np
import gym
import robotic_warehouse


class RWarehouse(MultiAgentEnv):
    def __init__(self, map_name, episode_limit=500):
        self.map_name = map_name
        self.episode_limit = episode_limit
        if map_name == 'tiny-2ag-v1':
            self.env = gym.make("rware-tiny-2ag-v1")
            self.n_agents = 2
        elif map_name == 'small-4ag-v1':
            self.env = gym.make("rware-small-4ag-v1")
            self.n_agents = 4
        elif map_name == '6ag-hard-v1':
            self.env = gym.make("rware-medium-6ag-hard-v1")
            self.n_agents = 6
        else:
            raise NotImplementedError
        self.observation_space = self.env.observation_space[0].shape[0]
        self.state_space = self.observation_space * self.n_agents
        self.action_space = self.env.action_space[0].n
        self.steps = 0

    def step(self, actions):
        all_obs, rewards, done, info = self.env.step(actions)
        reward = np.sum(rewards)

        if self.episode_limit - 1 > self.steps:
            done = any(done)
        else:
            done = True
        self.steps += 1
        return [reward, done, info]

    def get_obs(self):
        all_obs = [self.env._make_obs(agent) for agent in self.env.agents]
        return all_obs

    def get_obs_agent(self, agent_id):
        obs = self.env._make_obs(self.env.agents[agent_id])
        return obs

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
