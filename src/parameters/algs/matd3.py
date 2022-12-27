class MATD3:
    ALGORITHM = 'matd3'

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
    LEARNER = "matd3_learner"
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
