REGISTRY = {}

from .rnn_agent import RNNAgent
from .auto_encoder import AutoEncoder
from .central_rnn_agent import CentralRNNAgent
from .random_network import RandomNetworkModel
from .icm import ICMModule
from .state_decoder import StateDecoder
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["auto_encoder"] = AutoEncoder
REGISTRY["random_network"] = RandomNetworkModel
REGISTRY["icm_module"] = ICMModule
REGISTRY["state_decoder"] = StateDecoder
REGISTRY["noise_rnn"] = NoiseRNNAgent
