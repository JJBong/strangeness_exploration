# Strangeness-driven Exploration Method

## Strangeness-driven Exploration in Multi-Agent Reinforcement Learning

This repository refers to open source [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac), and is created for the purpose of experimental research on exploration method in MARL.
Moreover, new exploration method is added to well-known MARL algorithms, including [VDN](https://arxiv.org/pdf/1706.05296.pdf), [QMIX](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf), [QTRAN](http://proceedings.mlr.press/v97/son19a/son19a.pdf), [Weighted-QMIX](https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf), [QPLEX](https://openreview.net/forum?id=Rcmk0xxIQV), [MATD3](https://www.sciencedirect.com/science/article/pii/S0925231220309796).

### Requirements

We use PyTorch for this code, and log results using wandb.
The main requirements can be found in requirements.txt.

### Configs

You need to create a config file `src/parameters/parameters.py` or `src/parameters/multi_parameters.py` 
    refer to `src/parameters/parameters_template.py" or "src/parameters/multi_parameters_template.py`. 

You should refer to the config files at `src/parameters/algs`, `src/parameters/envs` and `src/parameters/general` directories.

### Experiments

Just run the `python main.py` command.
