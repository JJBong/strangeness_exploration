class QPLEX:
    ALGORITHM = 'qplex'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"
    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "dmaq_qatten_learner"
    DOUBLE_Q = True
    MIXER = "dmaq"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # params in Q_PLEX
    ADV_HYPERNET_LAYERS = 1
    ADV_HYPERNET_EMBED = 64
    NUM_KERNEL = 4
    IS_MINUS_ONE = True
    WEIGHTED_HEAD = True
    IS_ADV_ATTENTION = True
    IS_STOP_GRADIENT = True
    NONLINEAR = False
    STATE_BIAS = True
