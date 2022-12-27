class PettingZoo:
    ENV = 'pettingzoo'
    MAP = 'pistonball'
    BATCH_SEQUENCE = False #True
    BATCH_SEQUENCE_SIZE = 10
    MAX_CYCLES = 125
    USE_LINEAR = True
    if not USE_LINEAR:
        OBS_LAST_ACTION = False
        OBS_AGENT_ID = False

    ENV_ARGS = {
        'map_name': MAP,
        'max_cycles': MAX_CYCLES,
        'seed': None,
        'continuous': False,
        'render_mode': None,
        'use_linear_input': USE_LINEAR
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


class PettingZooPistonBall(PettingZoo):
    MAP = 'pistonball'
    MAX_CYCLES = 75
    USE_LINEAR = True
    if not USE_LINEAR:
        OBS_LAST_ACTION = False
        OBS_AGENT_ID = False
    ENV_ARGS = {
        'map_name': MAP,
        'max_cycles': MAX_CYCLES,
        'seed': None,
        'continuous': False,
        'render_mode': None,
        'use_linear_input': USE_LINEAR
    }

    BUFFER_SIZE = 100
    BATCH_SIZE = 4
    IMAGE_FLATTENED_SIZE = 64
    TRAIN_STEP_MAX = 50000


class PettingZooCooperativePong(PettingZoo):
    MAP = 'cooperative_pong'
    MAX_CYCLES = 250
    USE_LINEAR = True
    if not USE_LINEAR:
        OBS_LAST_ACTION = False
        OBS_AGENT_ID = False
    ENV_ARGS = {
        'map_name': MAP,
        'max_cycles': MAX_CYCLES,
        'seed': None,
        'render_mode': None,
        'use_linear_input': USE_LINEAR
    }

    BUFFER_SIZE = 50
    BATCH_SIZE = 4
    IMAGE_FLATTENED_SIZE = 64
    TRAIN_STEP_MAX = 50000


class PettingZooSimpleSpread(PettingZoo):
    MAP = 'simple_spread'
    MAX_CYCLES = 30
    USE_LINEAR = True
    if not USE_LINEAR:
        OBS_LAST_ACTION = False
        OBS_AGENT_ID = False
    ENV_ARGS = {
        'map_name': MAP,
        'max_cycles': MAX_CYCLES,
        'seed': None,
        'render_mode': None,
        'use_linear_input': USE_LINEAR
    }

    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    IMAGE_FLATTENED_SIZE = 64
    TRAIN_STEP_MAX = 50000


class PettingZooPursuit(PettingZoo):
    MAP = 'pursuit'
    MAX_CYCLES = 300
    USE_LINEAR = True
    if not USE_LINEAR:
        OBS_LAST_ACTION = False
        OBS_AGENT_ID = False
    ENV_ARGS = {
        'map_name': MAP,
        'max_cycles': MAX_CYCLES,
        'seed': None,
        'render_mode': None,
        'use_linear_input': USE_LINEAR
    }

    BUFFER_SIZE = 100
    BATCH_SIZE = 4
    IMAGE_FLATTENED_SIZE = 64
    TRAIN_STEP_MAX = 50000