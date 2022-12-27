class CWQMIX:
    ALGORITHM = 'cw_qmix'

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
    LEARNER = "max_q_learner"
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
