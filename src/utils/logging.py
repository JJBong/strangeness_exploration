import time
from collections import defaultdict
import torch as th
import wandb
import logging
import json
import numpy as np
import jsonpickle
import pickle


def flatten(obj):
    return json.loads(jsonpickle.encode(obj, keys=True))


class Logger:
    def __init__(self):
        logging.basicConfig()
        logger = logging.getLogger()
        logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(level=logging.INFO)

        self.console_logger = logger
        self.logging_info = {}
        self.stats = defaultdict(lambda: [])
        self.stats_for_saving = {}

    # def set_wandb(self, _wandb):
    def set_wandb(self, args):
        # if _wandb is not None:
        #     self.wandb = _wandb
        # else:
        #     self.wandb = None
        hidden_policy = '-no_hidden' if (not args.HIDDEN_POLICY and args.EXPLORATION) else ''
        start_time = time.strftime('%y-%m-%d/%X', time.localtime(time.time()))
        self.wandb = wandb.init(
            project=args.WANDB_PROJECT_NAME,
            config=vars(args),
            entity=args.WANDB_ENTITY,
            name="{}/{}/{}/{}".format(args.ENV, args.ALGORITHM+hidden_policy, args.MAP, start_time),
            reinit=True
        )

    def wandb_close(self):
        self.wandb.finish()

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        if key in self.logging_info.keys():
            self.logging_info["{}_T".format(key)].append(t)
            self.logging_info[key].append(value)
        else:
            self.logging_info["{}_T".format(key)] = [t]
            self.logging_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            items = []
            for x in self.stats[k][-window:]:
                item = x[1]
                if type(item) == th.Tensor:
                    item = item.cpu()
                items.append(item)
            item = "{:.4f}".format(np.mean(items))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"

            if k not in self.stats_for_saving.keys():
                self.stats_for_saving[k] = []
            self.stats_for_saving[k].append(item)

        self.log_info(log_str)

    def log_info(self, msg):
        self.console_logger.info(msg)

    def wandb_log_stats(self):
        if self.wandb is not None:
            _stats = {}
            i = 0
            for (k, v) in sorted(self.stats.items()):
                if k == "episode":
                    _stats[k] = int(self.stats["episode"][-1][1])
                else:
                    i += 1
                    window = 5 if k != "epsilon" else 1
                    items = []
                    for x in self.stats[k][-window:]:
                        item = x[1]
                        if type(item) == th.Tensor:
                            item = item.cpu()
                        items.append(item)
                    item = np.mean(items)
                    _stats[k] = item
            self.wandb.log(_stats)
        else:
            pass

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats_for_saving, f, pickle.HIGHEST_PROTOCOL)


# # set up a custom logger
# def get_logger():
#     logger = logging.getLogger()
#     logger.handlers = []
#     ch = logging.StreamHandler()
#     formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     logger.setLevel('DEBUG')
#
#     return logger
