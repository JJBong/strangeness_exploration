class QTRAN:
    ALGORITHM = 'qtran'

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
    LEARNER = "qtran_learner"
    DOUBLE_Q = True
    MIXER = "qtran_base"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    OPT_LOSS = 1
    NOPT_MIN_LOSS = 0.1

    NETWORK_SIZE = "small"

    QTRAN_ARCH = "qtran_paper"
