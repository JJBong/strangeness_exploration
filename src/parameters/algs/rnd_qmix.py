class RNDQMIX:
    ALGORITHM = 'rnd_qmix'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"
    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    RUNNER = "exp_parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "rnd_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # random network dim
    RANDOM_NETWORK_DIM = 64

    # intrinsic reward weighting
    RHO = 0.5
    BETA = 1.0
    # exploration parameters
    EXPLORATION = True
    EXP_MAC = "rnd_mac"
    EXP_AGENT = "random_network"
    ENCODER_HIDDEN_DIM = 64
