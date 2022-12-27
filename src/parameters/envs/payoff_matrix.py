class PayoffMatrix:
    ENV = 'payoff_matrix'
    MAP = 'k_step'
    K = 64
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 20000

    TEST_INTERVAL = 1000
    LOG_INTERVAL = 1000
    RUNNER_LOG_INTERVAL = 1000
    LEARNER_LOG_INTERVAL = 1000

    TRAIN_STEP_MAX = 100

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 1.0
    INPUT_NOISE_CLIP = 2.0
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 30000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class PayoffMatrix64Step(PayoffMatrix):
    MAP = 'k_step'
    K = 64
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 3000


class PayoffMatrix128Step(PayoffMatrix):
    MAP = 'k_step'
    K = 128
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 3000


class PayoffMatrix256Step(PayoffMatrix):
    MAP = 'k_step'
    K = 256
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 3000
