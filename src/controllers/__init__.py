REGISTRY = {}

from .basic_controller import BasicMAC
from .exp_controller import ExpMAC
from .central_basic_controller import CentralBasicMAC
from .rnd_controller import RndMAC
from .icm_controller import IcmMAC
from .noise_controller import NoiseMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["exp_mac"] = ExpMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["rnd_mac"] = RndMAC
REGISTRY["icm_mac"] = IcmMAC
REGISTRY["noise_mac"] = NoiseMAC
