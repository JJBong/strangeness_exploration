from .parameters_preamble import *


class MultiParameters:
    param1 = QmixSc3M()
    param1.WANDB = False
    param1.SAVE_MODEL = False
    param1.DEVICE = 'cpu'
    param1.RUNS = 10
    param1.BATCH_SIZE_RUN = 7
    param1.TRAIN_STEP_MAX = 20000
    param1.BUFFER_SIZE = 5000
    param1.TARGET_UPDATE_INTERVAL = 200
    param1.EPSILON_ANNEAL_TIME = int(param1.TRAIN_STEP_MAX / 5)

    param2 = ExpQmixSc3M()
    param2.WANDB = False
    param2.SAVE_MODEL = False
    param2.DEVICE = 'cpu'
    param2.RUNS = 10
    param2.BATCH_SIZE_RUN = 7
    param2.TRAIN_STEP_MAX = 20000
    param2.BUFFER_SIZE = 5000
    param2.TARGET_UPDATE_INTERVAL = 200
    param2.EPSILON_ANNEAL_TIME = int(param2.TRAIN_STEP_MAX / 5)
    param2.HIDDEN_POLICY = True
    param2.RHO = 0.5
    param2.BETA = 1.0

    params_list = [param1, param2]
