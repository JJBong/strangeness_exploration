class EXPCWQMIX:
    ALGORITHM = 'exp_cw_qmix'

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
    LEARNER = "exp_max_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # only use in cwqmix
    CENTRAL_LOSS = 1
    QMIX_LOSS = 1
    W = 0.1  # $\alpha$ in the paper
    HYSTERETIC_QMIX = False  # False -> CW-QMIX, True -> OW-QMIX

    CENTRAL_MIXING_EMBED_DIM = 256
    CENTRAL_ACTION_EMBED = 1
    CENTRAL_MAC = "basic_central_mac"
    CENTRAL_AGENT = "central_rnn"
    CENTRAL_RNN_HIDDEN_DIM = 64
    CENTRAL_MIXER = "ff"

    # exploration parameter
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
