import copy
import os
import random
import time
import torch as th
# from runners.parallel_runner import rollout_episode
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.rl_utils import make_dirs


# @ray.remote(num_cpus=10, num_gpus=1, max_calls=1)
def run(_config, run_number=0):
    args = _config
    args.DEVICE = args.DEVICE

    logger = Logger()

    if args.WANDB:
        logger.set_wandb(args)

    # Run and train
    run_sequential(args, logger)

    # Save stats
    hidden_policy = '-no_hidden' if (not args.HIDDEN_POLICY and args.EXPLORATION) else ''
    save_path = os.path.join(args.LOCAL_RESULTS_PATH, "stats", args.ALGORITHM + hidden_policy, args.ENV)
    make_dirs(save_path)
    stats_file_name = args.MAP+'_run:{}'.format(run_number)
    save_path = os.path.join(save_path, stats_file_name)
    logger.save_stats(save_path)
    if args.WANDB:
        logger.wandb_close()

    return None


def evaluate_sequential(args, runner):

    for _ in range(args.TEST_NEPISODE):
        runner.run(test_mode=True)

    if args.SAVE_REPLAY:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    model_save_path = os.path.join(args.LOCAL_RESULTS_PATH, "models", args.ALGORITHM, args.ENV, args.MAP)

    # Init runner so we can get env info
    runner = r_REGISTRY[args.RUNNER](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.N_AGENTS = env_info["n_agents"]
    args.N_ACTIONS = env_info["n_actions"]
    args.STATE_SHAPE = env_info["state_shape"]
    args.OBS_SHAPE = env_info["obs_shape"]

    if args.ACTION_SPACE == 'discrete' and not(args.ALGORITHM == "maven"):
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.N_AGENTS
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.N_ACTIONS)])
        }
    elif args.ACTION_SPACE == 'continuous' and not(args.ALGORITHM == "maven"):
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": env_info["n_actions"], "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.N_AGENTS
        }
        preprocess = {}
    elif args.ALGORITHM == "maven":
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "noise": {"vshape": (args.NOISE_DIM,)}
        }
        groups = {
            "agents": args.N_AGENTS
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.N_ACTIONS)])
        }
    else:
        raise Exception("redefine scheme")

    buffer = ReplayBuffer(scheme, groups, args.BUFFER_SIZE, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu" if args.BUFFER_CPU_ONLY else args.DEVICE)
    # Setup multiagent controller here
    # Give runner the scheme
    # Learner
    if args.EXPLORATION:
        mac = mac_REGISTRY[args.MAC](buffer.scheme, groups, args)
        exp_mac = mac_REGISTRY[args.EXP_MAC](buffer.scheme, args)
        if args.HIDDEN_POLICY:
            hidden_mac = mac_REGISTRY[args.MAC](buffer.scheme, groups, args)
            # hidden_mac = copy.deepcopy(mac)
            learner = le_REGISTRY[args.LEARNER](mac, exp_mac, buffer.scheme, logger, args, hidden_mac)
        else:
            learner = le_REGISTRY[args.LEARNER](mac, exp_mac, buffer.scheme, logger, args)
    else:
        mac = mac_REGISTRY[args.MAC](buffer.scheme, groups, args)
        learner = le_REGISTRY[args.LEARNER](mac, buffer.scheme, logger, args)

    if args.RUNNER.find('exp') >= 0:
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, exp_mac=exp_mac)
    else:
        if args.HIDDEN_POLICY:
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, hidden_mac=hidden_mac)
        else:
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    learner.to_device()

    if args.LOAD_MODEL:
        logger.log_info("Loading model from {}".format(model_save_path))
        learner.load_models(model_save_path)

        if args.EVALUATE or args.SAVE_REPLAY:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.TEST_INTERVAL - 1
    last_log_T = 0
    last_test_return_for_model_saving = -99999
    train_iteration = 0

    start_time = time.time()
    last_time = start_time

    logger.log_info("Beginning training for {} training timesteps".format(args.TRAIN_STEP_MAX))

    # while runner.t_env <= args.T_MAX:
    while train_iteration <= args.TRAIN_STEP_MAX:

        # Run for a whole episode at a time
        # futures = [runner.run.remote(runner, test_mode=False) for i in range(3)]
        # episode_batchs = ray.get(futures)
        episode_batch = runner.run(test_mode=False)
        # episode_batch = ray.get(rollout_episode.remote(runner, test_mode=False))
        # episode_batch = runner.run.remote(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        # for episode_batch in episode_batchs:
        #     buffer.insert_episode_batch(episode_batch)

        # Update epsilon in epsilon-greedy action selector
        if args.EPSILON_UPDATE_STANDARD == "train_steps":
            mac.action_selector.update_epsilon(train_steps=train_iteration)
            if args.EXPLORATION:
                exp_mac.update_noise_epsilon(train_steps=train_iteration)

        if buffer.can_sample(args.BATCH_SIZE) and episode >= args.START_TRAINING_EPISODE:
            for train_epoch in range(int(args.BATCH_SIZE_RUN / 2.0 + 1)):
                # logger.log_info("train_epoch - {}".format(train_epoch))
                episode_sample = buffer.sample(args.BATCH_SIZE)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if args.BATCH_SEQUENCE:
                    assert max_ep_t > args.BATCH_SEQUENCE_SIZE
                    start_batch_idx = random.randint(0, max_ep_t - 1 - args.BATCH_SEQUENCE_SIZE)
                    end_batch_idx = start_batch_idx + args.BATCH_SEQUENCE_SIZE
                    episode_sample = episode_sample[:, start_batch_idx:end_batch_idx]

                if episode_sample.device != args.DEVICE:
                    episode_sample.to(args.DEVICE)

                learner.train(episode_sample, runner.t_env, episode)
                train_iteration += 1
                logger.log_stat("train_iteration", train_iteration, runner.t_env)

                episode_sample.to('cpu')
                del episode_sample
                # cuda memory clear
                if 'cuda' in args.DEVICE:
                    th.cuda.empty_cache()

        # Execute test runs once in a while
        n_test_runs = max(1, args.TEST_NEPISODE // runner.batch_size)
        if (runner.t_env - last_test_T) / args.TEST_INTERVAL >= 1.0:

            logger.log_info("t_env: {}; train_steps {} / {}".format(runner.t_env, train_iteration, args.TRAIN_STEP_MAX))
            logger.log_info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, train_iteration, args.TRAIN_STEP_MAX), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

            current_test_return_for_model_saving = runner.test_return_for_model_saving
            if args.SAVE_MODEL and current_test_return_for_model_saving >= last_test_return_for_model_saving:
                last_test_return_for_model_saving = current_test_return_for_model_saving
                make_dirs(model_save_path)
                logger.log_info("Saving models to {}".format(model_save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(model_save_path)

        episode += args.BATCH_SIZE_RUN

        if (runner.t_env - last_log_T) >= args.LOG_INTERVAL:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.log_info("Finished Training")


def args_sanity_check(config, _log):

    # set DEVICE flags
    # config["DEVICE"] = True # Use cuda whenever possible!
    if config["DEVICE"] == 'cuda' and not th.cuda.is_available():
        config["DEVICE"] = 'cpu'
        _log.warning("CUDA flag CUDA was switched OFF automatically because no DEVICE devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
