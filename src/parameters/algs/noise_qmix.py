class NOISEQMIX:
    ALGORITHM = 'maven'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"
    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    RUNNER = "noise_parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "noise_q_learner"
    MAC = "noise_mac"
    AGENT = "noise_rnn"
    NOISE_DIM = 2
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    SKIP_CONNECTIONS = False
    HYPER_INITIALIZATION_NONZEROS = 0
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    MI_LOSS = 1
    RNN_DISCRIM = False
    RNN_AGG_SIZE = 32

    DISCRIM_SIZE = 64
    DISCRIM_LAYERS = 3

    NOISE_EMBEDDING_DIM = 32

    NOISE_BANDIT = False
    NOISE_BANDIT_LR = 0.1
    NOISE_BANDIT_EPSILON = 0.2

    MI_INTRINSIC = False
    MI_SCALER = 0.1
    HARD_QS = False

    BANDIT_EPSILON = 0.1
    BANDIT_ITERS = 8
    BANDIT_BATCH = 64
    BANDIT_BUFFER = 512
    BANDIT_REWARD_SCALING = 20
    BANDIT_USE_STATE = True
    BANDIT_POLICY = True
