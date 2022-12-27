class EXPMATD3:
    ALGORITHM = 'exp_matd3'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "multinomial"
    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 500000

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    ACTION_SPACE = "continuous"
    AGENT_OUTPUT_TYPE = "pi_logits"
    MAX_ACTION = 1.0
    MASK_BEFORE_SOFTMAX = True
    LEARNER = "exp_matd3_learner"
    EXPL_NOISE = 1.0
    MIXER = "qmix"
    MIXING_EMBED_DIM = 64
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64
    RNN_HIDDEN_DIM = 64
    CRITIC_HIDDEN_DIM = 64

    # --- RL hyperparameters (for actor-critic) ---
    START_TRAINING_EPISODE = 1000
    AGENT_TRAIN_FREQ = 2
    AGENT_LR = 0.0001
    CRITIC_LR = 0.001

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
