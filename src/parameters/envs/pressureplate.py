class PressurePlate:
    ENV = 'pressureplate'
    MAP = 'linear-4p' # 'linear-5p', 'linear-6p'
    EPISODE_LIMIT = 500

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 20000

    TEST_INTERVAL = 1000
    LOG_INTERVAL = 1000
    RUNNER_LOG_INTERVAL = 1000
    LEARNER_LOG_INTERVAL = 1000

    TRAIN_STEP_MAX = 10000

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 1.0
    INPUT_NOISE_CLIP = 2.0
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 30000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class PressurePlateLinear4P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-4p'
    EPISODE_LIMIT = 400

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BUFFER_SIZE = 5000
    BATCH_SIZE = 32
    TRAIN_STEP_MAX = 50000


class PressurePlateLinear5P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-5p'
    EPISODE_LIMIT = 300

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BUFFER_SIZE = 5000
    BATCH_SIZE = 32
    TRAIN_STEP_MAX = 50000


class PressurePlateLinear6P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-6p'
    EPISODE_LIMIT = 500

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BUFFER_SIZE = 5000
    BATCH_SIZE = 32
    TRAIN_STEP_MAX = 50000
