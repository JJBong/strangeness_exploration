class EXPQPLEX:
    ALGORITHM = 'exp_qplex'

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
    LEARNER = "exp_dmaq_qatten_learner"
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

    # exploration parameters
    EXPLORATION = True
    HIDDEN_POLICY = True
    EXP_MAC = "exp_mac"
    EXP_AGENT = "auto_encoder"
    RHO = 0.5
    BETA = 1.0
    ENCODER_HIDDEN_DIM = 64
    FEATURE_HIDDEN_DIM = 64
    ENCODER_RNN = False

    IR_DECAY_START = 1.0
    IR_DECAY_FINISH = 0.01
    IR_DECAY_ANNEAL_TIME = 20000

    Q_INT_REWARD = False
