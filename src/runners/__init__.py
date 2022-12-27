REGISTRY = {}

from .episode_runner import EpisodeRunner
from .episode_exp_runner import ExpEpisodeRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["exp_episode"] = ExpEpisodeRunner

from .parallel_runner import ParallelRunner
from .parallel_exp_runner import ExpParallelRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["exp_parallel"] = ExpParallelRunner

from .noise_parallel_runner import ParallelRunner as NoiseParallelRunner
REGISTRY["noise_parallel"] = NoiseParallelRunner