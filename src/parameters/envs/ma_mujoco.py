class MaMujoco:
    ENV = 'ma_mujoco'
    MAP = 'Ant-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "2x4",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    TEST_GREEDY = True
    TEST_NEPISODE = 32
    TEST_INTERVAL = 2500
    LOG_INTERVAL = 2500
    RUNNER_LOG_INTERVAL = 2500
    LEARNER_LOG_INTERVAL = 2500
    TRAIN_STEP_MAX = 1000

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 0.2
    INPUT_NOISE_CLIP = 0.5
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 100000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class MaMujoco2AAnt(MaMujoco):
    MAP = 'Ant-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "2x4",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco4AAnt(MaMujoco):
    MAP = 'Ant-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "4x2",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco2AHalfcheetah(MaMujoco):
    MAP = 'HalfCheetah-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "2x3",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco6AHalfcheetah(MaMujoco):
    MAP = 'HalfCheetah-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "6x1",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco3AHopper(MaMujoco):
    MAP = 'Hopper-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "3x1",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco2AHumanoid(MaMujoco):
    MAP = 'Humanoid-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "9|8",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujoco2AHumanoidStandup(MaMujoco):
    MAP = 'HumanoidStandup-v2'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "9|8",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000

    EXTRINSIC_REWARD_WEIGHT = 0.005


class MaMujocoManyAgentSwimmer(MaMujoco):
    MAP = 'manyagent_swimmer'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "10x2",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujocoCoupledHalfCheetah(MaMujoco):
    MAP = 'coupled_half_cheetah'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "1p1",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000


class MaMujocoManyAgentAnt(MaMujoco):
    MAP = 'manyagent_ant'
    ENV_ARGS = {
        'env_args': {
            "scenario": MAP,
            "agent_conf": "2x3",
            "agent_obsk": 1,
            "episode_limit": 200
        }
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 20000
