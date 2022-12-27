import os
src_path = os.getcwd()
replay_path = os.path.join(src_path, "results", "replays")


class SC2:
    ENV = 'sc2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    TEST_GREEDY = True
    TEST_NEPISODE = 32
    TEST_INTERVAL = 2500
    LOG_INTERVAL = 2500
    RUNNER_LOG_INTERVAL = 2500
    LEARNER_LOG_INTERVAL = 2500
    TRAIN_STEP_MAX = 10000

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 0.2
    INPUT_NOISE_CLIP = 0.5
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 100000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class SC2_3m(SC2):
    MAP = '3m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    TRAIN_STEP_MAX = 30000


class SC2_3s5z(SC2):
    MAP = '3s5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    TRAIN_STEP_MAX = 30000


class SC2_27m_vs_30m(SC2):
    MAP = '27m_vs_30m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "27m_vs_30m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_3s_vs_5z(SC2):
    MAP = '3s_vs_5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s_vs_5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_Corridor(SC2):
    MAP = 'corridor'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "corridor",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_5m_vs_6m(SC2):
    MAP = '5m_vs_6m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "5m_vs_6m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_MMM(SC2):
    MAP = 'MMM'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_MMM2(SC2):
    MAP = 'MMM2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM2",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_2c_vs_64zg(SC2):
    MAP = '2c_vs_64zg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2c_vs_64zg",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_6h_vs_8z(SC2):
    MAP = '6h_vs_8z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "6h_vs_8z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000000

    TRAIN_STEP_MAX = 50000


class SC2_3s5z_vs_3s6z(SC2):
    MAP = '3s5z_vs_3s6z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s5z_vs_3s6z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000000

    TRAIN_STEP_MAX = 50000

class SC2_2s_vs_1sc(SC2):
    MAP = '2s_vs_1sc'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2s_vs_1sc",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 50000