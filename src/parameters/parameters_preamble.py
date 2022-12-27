from parameters.general.general import General

from parameters.algs.qmix import QMIX
from parameters.algs.expqmix import EXPQMIX
from parameters.algs.qtran import QTRAN
from parameters.algs.expqtran import EXPQTRAN
from parameters.algs.qplex import QPLEX
from parameters.algs.expqplex import EXPQPLEX
from parameters.algs.matd3 import MATD3
from parameters.algs.expmatd3 import EXPMATD3
from parameters.algs.cw_qmix import CWQMIX
from parameters.algs.expcw_qmix import EXPCWQMIX
from parameters.algs.ow_qmix import OWQMIX
from parameters.algs.expow_qmix import EXPOWQMIX
from parameters.algs.semi_emc_qmix import SEMIEMCQMIX
from parameters.algs.rnd_qmix import RNDQMIX
from parameters.algs.icm_qmix import ICMQMIX
from parameters.algs.noise_qmix import NOISEQMIX as MAVEN

from parameters.envs.payoff_matrix import PayoffMatrix64Step, PayoffMatrix128Step, PayoffMatrix256Step
from parameters.envs.sc2 import SC2_3m, SC2_27m_vs_30m, SC2_3s_vs_5z, SC2_3s5z, SC2_2c_vs_64zg, SC2_MMM, SC2_Corridor, SC2_6h_vs_8z, SC2_5m_vs_6m, SC2_MMM2, SC2_3s5z_vs_3s6z, SC2_2s_vs_1sc
from parameters.envs.ma_mujoco import MaMujoco2AAnt, MaMujoco4AAnt, MaMujoco2AHalfcheetah, MaMujoco6AHalfcheetah, MaMujoco3AHopper, MaMujoco2AHumanoid, MaMujoco2AHumanoidStandup, MaMujocoManyAgentSwimmer, MaMujocoCoupledHalfCheetah, MaMujocoManyAgentAnt
from parameters.envs.pettingzoo import PettingZooPistonBall, PettingZooCooperativePong, PettingZooSimpleSpread, PettingZooPursuit
from parameters.envs.pressureplate import PressurePlateLinear4P, PressurePlateLinear5P, PressurePlateLinear6P
from parameters.envs.rwarehouse import RWarehouseTiny2Ag, RWarehouseSmall4Ag, RWarehouseHard6Ag

from utils.param_utils import check_parameters_overlapped


# Very Important !!!
# The below rule must be observed
# 1. class definition priority (env -> alg -> general)
# 2. checking parameters priority (general -> alg -> env)

######################################################################################################
# QMIX

# Payoff-matrix Game: K-step
class QmixPom64step(PayoffMatrix64Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix64Step)
class QmixPom128step(PayoffMatrix128Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix128Step)
class QmixPom256step(PayoffMatrix256Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix256Step)

class QmixPZPistonBall(PettingZooPistonBall, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooPistonBall)
class QmixPZCooperativePong(PettingZooCooperativePong, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooCooperativePong)
class QmixPZSimpleSpread(PettingZooSimpleSpread, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooSimpleSpread)
class QmixPZPursuit(PettingZooPursuit, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooPursuit)

class QmixPP4P(PressurePlateLinear4P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear4P)
class QmixPP5P(PressurePlateLinear5P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear5P)
class QmixPP6P(PressurePlateLinear6P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear6P)

class QmixRWTiny2Ag(RWarehouseTiny2Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseTiny2Ag)
class QmixRWSmall4Ag(RWarehouseSmall4Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseSmall4Ag)
class QmixRWHard6Ag(RWarehouseHard6Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class QmixSc3M(SC2_3m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class QmixSc27Mvs30M(SC2_27m_vs_30m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class QmixSc3Svs5Z(SC2_3s_vs_5z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class QmixSc3S5Z(SC2_3s5z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class QmixSc2Cvs64ZG(SC2_2c_vs_64zg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2c_vs_64zg)

# StarCraft2: MMM
class QmixScMMM(SC2_MMM, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_MMM)

# StarCraft2: corridor
class QmixScCorridor(SC2_Corridor, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class QmixSc5Mvs6M(SC2_5m_vs_6m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class QmixSc6Hvs8Z(SC2_6h_vs_8z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QmixScMMM2(SC2_MMM2, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QmixSc2Svs1SC(SC2_2s_vs_1sc, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2s_vs_1sc)

######################################################################################################


######################################################################################################
# EXPQMIX

class ExpQmixPom64step(PayoffMatrix64Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix64Step)
class ExpQmixPom128step(PayoffMatrix128Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix128Step)
class ExpQmixPom256step(PayoffMatrix256Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix256Step)

class ExpQmixPZPistonBall(PettingZooPistonBall, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooPistonBall)
class ExpQmixPZCooperativePong(PettingZooCooperativePong, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooCooperativePong)
class ExpQmixPZSimpleSpread(PettingZooSimpleSpread, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooSimpleSpread)
class ExpQmixPZPursuit(PettingZooPursuit, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooPursuit)

class ExpQmixPP4P(PressurePlateLinear4P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear4P)
class ExpQmixPP5P(PressurePlateLinear5P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear5P)
class ExpQmixPP6P(PressurePlateLinear6P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear6P)

class ExpQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseTiny2Ag)
class ExpQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseSmall4Ag)
class ExpQmixRWHard6Ag(RWarehouseHard6Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseHard6Ag)

class ExpQmixSc3M(SC2_3m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3m)

class ExpQmixSc27Mvs30M(SC2_27m_vs_30m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_27m_vs_30m)

class ExpQmixSc3Svs5Z(SC2_3s_vs_5z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s_vs_5z)

class ExpQmixSc3S5Z(SC2_3s5z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s5z)

class ExpQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2c_vs_64zg)

class ExpQmixScMMM(SC2_MMM, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_MMM)

class ExpQmixScCorridor(SC2_Corridor, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_Corridor)

class ExpQmixSc5Mvs6M(SC2_5m_vs_6m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_5m_vs_6m)

class ExpQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQmixScMMM2(SC2_MMM2, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# SEMIEMCQMIX

# Payoff-matrix Game: K-step
class SemiEmcQmixPom64step(PayoffMatrix64Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix64Step)
class SemiEmcQmixPom128step(PayoffMatrix128Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix128Step)
class SemiEmcQmixPom256step(PayoffMatrix256Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix256Step)

class SemiEmcQmixPZPistonBall(PettingZooPistonBall, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooPistonBall)
class SemiEmcQmixPZCooperativePong(PettingZooCooperativePong, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooCooperativePong)
class SemiEmcQmixPZSimpleSpread(PettingZooSimpleSpread, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooSimpleSpread)
class SemiEmcQmixPZPursuit(PettingZooPursuit, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooPursuit)

class SemiEmcQmixPP4P(PressurePlateLinear4P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear4P)
class SemiEmcQmixPP5P(PressurePlateLinear5P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear5P)
class SemiEmcQmixPP6P(PressurePlateLinear6P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear6P)

class SemiEmcQmixRWTiny2Ag(RWarehouseTiny2Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseTiny2Ag)
class SemiEmcQmixRWSmall4Ag(RWarehouseSmall4Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseSmall4Ag)
class SemiEmcQmixRWHard6Ag(RWarehouseHard6Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class SemiEmcQmixSc3M(SC2_3m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class SemiEmcQmixSc27Mvs30M(SC2_27m_vs_30m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class SemiEmcQmixSc3Svs5Z(SC2_3s_vs_5z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s_vs_5z)

class SemiEmcQmixSc3S5Z(SC2_3s5z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s5z)

class SemiEmcQmixSc2Cvs64ZG(SC2_2c_vs_64zg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_2c_vs_64zg)

class SemiEmcQmixScMMM(SC2_MMM, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_MMM)

class SemiEmcQmixScCorridor(SC2_Corridor, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_Corridor)

class SemiEmcQmixSc5Mvs6M(SC2_5m_vs_6m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_5m_vs_6m)

class SemiEmcQmixSc6Hvs8Z(SC2_6h_vs_8z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class SemiEmcQmixScMMM2(SC2_MMM2, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class SemiEmcQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class SemiEmcQmixSc2Svs1SC(SC2_2s_vs_1sc, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# RNDQMIX

# Payoff-matrix Game: K-step
class RndQmixPom64step(PayoffMatrix64Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix64Step)
class RndQmixPom128step(PayoffMatrix128Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix128Step)
class RndQmixPom256step(PayoffMatrix256Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix256Step)

class RndQmixPZPistonBall(PettingZooPistonBall, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooPistonBall)
class RndQmixPZCooperativePong(PettingZooCooperativePong, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooCooperativePong)
class RndQmixPZSimpleSpread(PettingZooSimpleSpread, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooSimpleSpread)
class RndQmixPZPursuit(PettingZooPursuit, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooPursuit)

class RndQmixPP4P(PressurePlateLinear4P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear4P)
class RndQmixPP5P(PressurePlateLinear5P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear5P)
class RndQmixPP6P(PressurePlateLinear6P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear6P)

class RndQmixRWTiny2Ag(RWarehouseTiny2Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseTiny2Ag)
class RndQmixRWSmall4Ag(RWarehouseSmall4Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseSmall4Ag)
class RndQmixRWHard6Ag(RWarehouseHard6Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class RndQmixSc3M(SC2_3m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class RndQmixSc27Mvs30M(SC2_27m_vs_30m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class RndQmixSc3Svs5Z(SC2_3s_vs_5z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s_vs_5z)

class RndQmixSc3S5Z(SC2_3s5z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s5z)

class RndQmixSc2Cvs64ZG(SC2_2c_vs_64zg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_2c_vs_64zg)

class RndQmixScMMM(SC2_MMM, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_MMM)

class RndQmixScCorridor(SC2_Corridor, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_Corridor)

class RndQmixSc5Mvs6M(SC2_5m_vs_6m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_5m_vs_6m)

class RndQmixSc6Hvs8Z(SC2_6h_vs_8z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class RndQmixScMMM2(SC2_MMM2, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class RndQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class RndQmixSc2Svs1SC(SC2_2s_vs_1sc, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# ICMQMIX

class IcmQmixPom64step(PayoffMatrix64Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix64Step)
class IcmQmixPom128step(PayoffMatrix128Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix128Step)
class IcmQmixPom256step(PayoffMatrix256Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix256Step)

class IcmQmixPZPistonBall(PettingZooPistonBall, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooPistonBall)
class IcmQmixPZCooperativePong(PettingZooCooperativePong, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooCooperativePong)
class IcmQmixPZSimpleSpread(PettingZooSimpleSpread, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooSimpleSpread)
class IcmQmixPZPursuit(PettingZooPursuit, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooPursuit)

class IcmQmixPP4P(PressurePlateLinear4P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear4P)
class IcmQmixPP5P(PressurePlateLinear5P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear5P)
class IcmQmixPP6P(PressurePlateLinear6P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear6P)

class IcmQmixRWTiny2Ag(RWarehouseTiny2Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseTiny2Ag)
class IcmQmixRWSmall4Ag(RWarehouseSmall4Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseSmall4Ag)
class IcmQmixRWHard6Ag(RWarehouseHard6Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseHard6Ag)

class IcmQmixSc3M(SC2_3m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3m)

class IcmQmixSc27Mvs30M(SC2_27m_vs_30m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_27m_vs_30m)

class IcmQmixSc3Svs5Z(SC2_3s_vs_5z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s_vs_5z)

class IcmQmixSc3S5Z(SC2_3s5z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s5z)

class IcmQmixSc2Cvs64ZG(SC2_2c_vs_64zg, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_2c_vs_64zg)

class IcmQmixScMMM(SC2_MMM, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_MMM)

class IcmQmixScCorridor(SC2_Corridor, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_Corridor)

class IcmQmixSc5Mvs6M(SC2_5m_vs_6m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_5m_vs_6m)

class IcmQmixSc6Hvs8Z(SC2_6h_vs_8z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class IcmQmixScMMM2(SC2_MMM2, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class IcmQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class IcmQmixSc2Svs1SC(SC2_2s_vs_1sc, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# QTRAN

class QtranPom64step(PayoffMatrix64Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix64Step)
class QtranPom128step(PayoffMatrix128Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix128Step)
class QtranPom256step(PayoffMatrix256Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix256Step)

class QtranPZPistonBall(PettingZooPistonBall, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooPistonBall)
class QtranPZCooperativePong(PettingZooCooperativePong, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooCooperativePong)
class QtranPZSimpleSpread(PettingZooSimpleSpread, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooSimpleSpread)
class QtranPZPursuit(PettingZooPursuit, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooPursuit)

class QtranPP4P(PressurePlateLinear4P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear4P)
class QtranPP5P(PressurePlateLinear5P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear5P)
class QtranPP6P(PressurePlateLinear6P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear6P)

class QtranRWTiny2Ag(RWarehouseTiny2Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseTiny2Ag)
class QtranRWSmall4Ag(RWarehouseSmall4Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseSmall4Ag)
class QtranRWHard6Ag(RWarehouseHard6Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseHard6Ag)

class QtranSc3M(SC2_3m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3m)

class QtranSc27Mvs30M(SC2_27m_vs_30m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_27m_vs_30m)

class QtranSc3Svs5Z(SC2_3s_vs_5z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s_vs_5z)

class QtranSc3S5Z(SC2_3s5z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s5z)

class QtranSc2Cvs64ZG(SC2_2c_vs_64zg, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_2c_vs_64zg)

class QtranScMMM(SC2_MMM, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_MMM)

class QtranScCorridor(SC2_Corridor, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_Corridor)

class QtranSc5Mvs6M(SC2_5m_vs_6m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_5m_vs_6m)

class QtranSc6Hvs8Z(SC2_6h_vs_8z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QtranScMMM2(SC2_MMM2, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QtranSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QtranSc2Svs1SC(SC2_2s_vs_1sc, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# EXPQTRAN

class ExpQtranPom64step(PayoffMatrix64Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix64Step)
class ExpQtranPom128step(PayoffMatrix128Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix128Step)
class ExpQtranPom256step(PayoffMatrix256Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix256Step)

class ExpQtranPZPistonBall(PettingZooPistonBall, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooPistonBall)
class ExpQtranPZCooperativePong(PettingZooCooperativePong, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooCooperativePong)
class ExpQtranPZSimpleSpread(PettingZooSimpleSpread, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooSimpleSpread)
class ExpQtranPZPursuit(PettingZooPursuit, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooPursuit)

class ExpQtranPP4P(PressurePlateLinear4P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear4P)
class ExpQtranPP5P(PressurePlateLinear5P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear5P)
class ExpQtranPP6P(PressurePlateLinear6P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear6P)

class ExpQtranRWTiny2Ag(RWarehouseTiny2Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseTiny2Ag)
class ExpQtranRWSmall4Ag(RWarehouseSmall4Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseSmall4Ag)
class ExpQtranRWHard6Ag(RWarehouseHard6Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseHard6Ag)

class ExpQtranSc3M(SC2_3m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3m)


class ExpQtranSc27Mvs30M(SC2_27m_vs_30m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_27m_vs_30m)


class ExpQtranSc3Svs5Z(SC2_3s_vs_5z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s_vs_5z)

class ExpQtranSc3S5Z(SC2_3s5z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s5z)

class ExpQtranSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_2c_vs_64zg)

class ExpQtranScMMM(SC2_MMM, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_MMM)

class ExpQtranScCorridor(SC2_Corridor, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_Corridor)

class ExpQtranSc5Mvs6M(SC2_5m_vs_6m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_5m_vs_6m)

class ExpQtranSc6Hvs8Z(SC2_6h_vs_8z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQtranScMMM2(SC2_MMM2, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQtranSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQtranSc2Svs1SC(SC2_2s_vs_1sc, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# QPLEX

# Payoff-matrix Game: K-step
class QplexPom64step(PayoffMatrix64Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix64Step)
class QplexPom128step(PayoffMatrix128Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix128Step)
class QplexPom256step(PayoffMatrix256Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix256Step)

class QplexPZPistonBall(PettingZooPistonBall, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooPistonBall)
class QplexPZCooperativePong(PettingZooCooperativePong, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooCooperativePong)
class QplexPZSimpleSpread(PettingZooSimpleSpread, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooSimpleSpread)
class QplexPZPursuit(PettingZooPursuit, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooPursuit)

class QplexPP4P(PressurePlateLinear4P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear4P)
class QplexPP5P(PressurePlateLinear5P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear5P)
class QplexPP6P(PressurePlateLinear6P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear6P)

class QplexRWTiny2Ag(RWarehouseTiny2Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseTiny2Ag)
class QplexRWSmall4Ag(RWarehouseSmall4Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseSmall4Ag)
class QplexRWHard6Ag(RWarehouseHard6Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseHard6Ag)

# StarCraft2: 3m
class QplexSc3M(SC2_3m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3m)

# StarCraft2: 27m_vs_30m
class QplexSc27Mvs30M(SC2_27m_vs_30m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class QplexSc3Svs5Z(SC2_3s_vs_5z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s_vs_5z)

class QplexSc3S5Z(SC2_3s5z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s5z)

class QplexSc2Cvs64ZG(SC2_2c_vs_64zg, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_2c_vs_64zg)

class QplexScMMM(SC2_MMM, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_MMM)

class QplexScCorridor(SC2_Corridor, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_Corridor)

class QplexSc5Mvs6M(SC2_5m_vs_6m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_5m_vs_6m)

class QplexSc6Hvs8Z(SC2_6h_vs_8z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QplexScMMM2(SC2_MMM2, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QplexSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QplexSc2Svs1SC(SC2_2s_vs_1sc, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# EXPQPLEX

# Payoff-matrix Game: K-step
class ExpQplexPom64step(PayoffMatrix64Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix64Step)
class ExpQplexPom128step(PayoffMatrix128Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix128Step)
class ExpQplexPom256step(PayoffMatrix256Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix256Step)

class ExpQplexPZPistonBall(PettingZooPistonBall, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooPistonBall)
class ExpQplexPZCooperativePong(PettingZooCooperativePong, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooCooperativePong)
class ExpQplexPZSimpleSpread(PettingZooSimpleSpread, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooSimpleSpread)
class ExpQplexPZPursuit(PettingZooPursuit, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooPursuit)

class ExpQplexPP4P(PressurePlateLinear4P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear4P)
class ExpQplexPP5P(PressurePlateLinear5P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear5P)
class ExpQplexPP6P(PressurePlateLinear6P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear6P)

class ExpQplexRWTiny2Ag(RWarehouseTiny2Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseTiny2Ag)
class ExpQplexRWSmall4Ag(RWarehouseSmall4Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseSmall4Ag)
class ExpQplexRWHard6Ag(RWarehouseHard6Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpQplexSc3M(SC2_3m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3m)


# StarCraft2: 27m_vs_30m
class ExpQplexSc27Mvs30M(SC2_27m_vs_30m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpQplexSc3Svs5Z(SC2_3s_vs_5z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s_vs_5z)

class ExpQplexSc3S5Z(SC2_3s5z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s5z)

class ExpQplexSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_2c_vs_64zg)

class ExpQplexScMMM(SC2_MMM, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_MMM)

class ExpQplexScCorridor(SC2_Corridor, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_Corridor)

class ExpQplexSc5Mvs6M(SC2_5m_vs_6m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_5m_vs_6m)

class ExpQplexSc6Hvs8Z(SC2_6h_vs_8z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQplexScMMM2(SC2_MMM2, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQplexSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQplexSc2Svs1SC(SC2_2s_vs_1sc, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# CWQMIX

# Payoff-matrix Game: K-step
class CwQmixPom64step(PayoffMatrix64Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix64Step)
class CwQmixPom128step(PayoffMatrix128Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix128Step)
class CwQmixPom256step(PayoffMatrix256Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix256Step)

class CwQmixPZPistonBall(PettingZooPistonBall, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooPistonBall)
class CwQmixPZCooperativePong(PettingZooCooperativePong, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooCooperativePong)
class CwQmixPZSimpleSpread(PettingZooSimpleSpread, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooSimpleSpread)
class CwQmixPZPursuit(PettingZooPursuit, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooPursuit)

class CwQmixPP4P(PressurePlateLinear4P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear4P)
class CwQmixPP5P(PressurePlateLinear5P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear5P)
class CwQmixPP6P(PressurePlateLinear6P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear6P)

class CwQmixRWTiny2Ag(RWarehouseTiny2Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseTiny2Ag)
class CwQmixRWSmall4Ag(RWarehouseSmall4Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseSmall4Ag)
class CwQmixRWHard6Ag(RWarehouseHard6Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class CwQmixSc3M(SC2_3m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class CwQmixSc27Mvs30M(SC2_27m_vs_30m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class CwQmixSc3Svs5Z(SC2_3s_vs_5z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s_vs_5z)

class CwQmixSc3S5Z(SC2_3s5z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s5z)

class CwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_2c_vs_64zg)

class CwQmixScMMM(SC2_MMM, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_MMM)

class CwQmixScCorridor(SC2_Corridor, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_Corridor)

class CwQmixSc5Mvs6M(SC2_5m_vs_6m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_5m_vs_6m)

class CwQmixSc6Hvs8Z(SC2_6h_vs_8z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class CwQmixScMMM2(SC2_MMM2, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class CwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class CwQmixSc2Svs1SC(SC2_2s_vs_1sc, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# EXPCWQMIX

# Payoff-matrix Game: K-step
class ExpCwQmixPom64step(PayoffMatrix64Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix64Step)
class ExpCwQmixPom128step(PayoffMatrix128Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix128Step)
class ExpCwQmixPom256step(PayoffMatrix256Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix256Step)

class ExpCwQmixPZPistonBall(PettingZooPistonBall, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooPistonBall)
class ExpCwQmixPZCooperativePong(PettingZooCooperativePong, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooCooperativePong)
class ExpCwQmixPZSimpleSpread(PettingZooSimpleSpread, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooSimpleSpread)
class ExpCwQmixPZPursuit(PettingZooPursuit, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooPursuit)

class ExpCwQmixPP4P(PressurePlateLinear4P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear4P)
class ExpCwQmixPP5P(PressurePlateLinear5P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear5P)
class ExpCwQmixPP6P(PressurePlateLinear6P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear6P)

class ExpCwQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseTiny2Ag)
class ExpCwQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseSmall4Ag)
class ExpCwQmixRWHard6Ag(RWarehouseHard6Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpCwQmixSc3M(SC2_3m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class ExpCwQmixSc27Mvs30M(SC2_27m_vs_30m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpCwQmixSc3Svs5Z(SC2_3s_vs_5z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s_vs_5z)

class ExpCwQmixSc3S5Z(SC2_3s5z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s5z)

class ExpCwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_2c_vs_64zg)

class ExpCwQmixScMMM(SC2_MMM, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_MMM)

class ExpCwQmixScCorridor(SC2_Corridor, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_Corridor)

class ExpCwQmixSc5Mvs6M(SC2_5m_vs_6m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_5m_vs_6m)

class ExpCwQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpCwQmixScMMM2(SC2_MMM2, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpCwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpCwQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# OWQMIX

# Payoff-matrix Game: K-step
class OwQmixPom64step(PayoffMatrix64Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix64Step)
class OwQmixPom128step(PayoffMatrix128Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix128Step)
class OwQmixPom256step(PayoffMatrix256Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix256Step)

class OwQmixPZPistonBall(PettingZooPistonBall, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooPistonBall)
class OwQmixPZCooperativePong(PettingZooCooperativePong, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooCooperativePong)
class OwQmixPZSimpleSpread(PettingZooSimpleSpread, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooSimpleSpread)
class OwQmixPZPursuit(PettingZooPursuit, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooPursuit)

class OwQmixPP4P(PressurePlateLinear4P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear4P)
class OwQmixPP5P(PressurePlateLinear5P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear5P)
class OwQmixPP6P(PressurePlateLinear6P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear6P)

class OwQmixRWTiny2Ag(RWarehouseTiny2Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseTiny2Ag)
class OwQmixRWSmall4Ag(RWarehouseSmall4Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseSmall4Ag)
class OwQmixRWHard6Ag(RWarehouseHard6Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class OwQmixSc3M(SC2_3m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class OwQmixSc27Mvs30M(SC2_27m_vs_30m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class OwQmixSc3Svs5Z(SC2_3s_vs_5z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s_vs_5z)

class OwQmixSc3S5Z(SC2_3s5z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s5z)

class OwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_2c_vs_64zg)

class OwQmixScMMM(SC2_MMM, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_MMM)

class OwQmixScCorridor(SC2_Corridor, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_Corridor)

class OwQmixSc5Mvs6M(SC2_5m_vs_6m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_5m_vs_6m)

class OwQmixSc6Hvs8Z(SC2_6h_vs_8z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class OwQmixScMMM2(SC2_MMM2, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class OwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class OwQmixSc2Svs1SC(SC2_2s_vs_1sc, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# EXPOWQMIX

# Payoff-matrix Game: K-step
class ExpOwQmixPom64step(PayoffMatrix64Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix64Step)
class ExpOwQmixPom128step(PayoffMatrix128Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix128Step)
class ExpOwQmixPom256step(PayoffMatrix256Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix256Step)

class ExpOwQmixPZPistonBall(PettingZooPistonBall, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooPistonBall)
class ExpOwQmixPZCooperativePong(PettingZooCooperativePong, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooCooperativePong)
class ExpOwQmixPZSimpleSpread(PettingZooSimpleSpread, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooSimpleSpread)
class ExpOwQmixPZPursuit(PettingZooPursuit, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooPursuit)

class ExpOwQmixPP4P(PressurePlateLinear4P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear4P)
class ExpOwQmixPP5P(PressurePlateLinear5P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear5P)
class ExpOwQmixPP6P(PressurePlateLinear6P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear6P)

class ExpOwQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseTiny2Ag)
class ExpOwQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseSmall4Ag)
class ExpOwQmixRWHard6Ag(RWarehouseHard6Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpOwQmixSc3M(SC2_3m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class ExpOwQmixSc27Mvs30M(SC2_27m_vs_30m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpOwQmixSc3Svs5Z(SC2_3s_vs_5z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s_vs_5z)

class ExpOwQmixSc3S5Z(SC2_3s5z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s5z)

class ExpOwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_2c_vs_64zg)

class ExpOwQmixScMMM(SC2_MMM, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_MMM)

class ExpOwQmixScCorridor(SC2_Corridor, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_Corridor)

class ExpOwQmixSc5Mvs6M(SC2_5m_vs_6m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_5m_vs_6m)

class ExpOwQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpOwQmixScMMM2(SC2_MMM2, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpOwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpOwQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_2s_vs_1sc)
######################################################################################################


######################################################################################################
# MAVEN

# Payoff-matrix Game: K-step
class MavenPom64step(PayoffMatrix64Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix64Step)
class MavenPom128step(PayoffMatrix128Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix128Step)
class MavenPom256step(PayoffMatrix256Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix256Step)

class MavenPZPistonBall(PettingZooPistonBall, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooPistonBall)
class MavenPZCooperativePong(PettingZooCooperativePong, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooCooperativePong)
class MavenPZSimpleSpread(PettingZooSimpleSpread, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooSimpleSpread)
class MavenPZPursuit(PettingZooPursuit, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooPursuit)

class MavenPP4P(PressurePlateLinear4P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear4P)
class MavenPP5P(PressurePlateLinear5P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear5P)
class MavenPP6P(PressurePlateLinear6P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear6P)

class MavenRWTiny2Ag(RWarehouseTiny2Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseTiny2Ag)
class MavenRWSmall4Ag(RWarehouseSmall4Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseSmall4Ag)
class MavenRWHard6Ag(RWarehouseHard6Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseHard6Ag)

# StarCraft2: 3m
class MavenSc3M(SC2_3m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3m)

# StarCraft2: 27m_vs_30m
class MavenSc27Mvs30M(SC2_27m_vs_30m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class MavenSc3Svs5Z(SC2_3s_vs_5z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class MavenSc3S5Z(SC2_3s5z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class MavenSc2Cvs64ZG(SC2_2c_vs_64zg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_2c_vs_64zg)

# StarCraft2: MMM
class MavenScMMM(SC2_MMM, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_MMM)

# StarCraft2: corridor
class MavenScCorridor(SC2_Corridor, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class MavenSc5Mvs6M(SC2_5m_vs_6m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class MavenSc6Hvs8Z(SC2_6h_vs_8z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class MavenScMMM2(SC2_MMM2, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class MavenSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class MavenSc2Svs1SC(SC2_2s_vs_1sc, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_2s_vs_1sc)

######################################################################################################


######################################################################################################
# MATD3

class MATd3MaMujoco2AAnt(MaMujoco2AAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AAnt)

class MATd3MaMujoco4AAnt(MaMujoco4AAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco4AAnt)

class MATd3MaMujoco6AHalfcheetah(MaMujoco6AHalfcheetah, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco6AHalfcheetah)

class MATd3MaMujoco3AHopper(MaMujoco3AHopper, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco3AHopper)

class MATd3MaMujoco2AHumanoid(MaMujoco2AHumanoid, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AHumanoid)

class MATd3MaMujoco2AHumanoidStandup(MaMujoco2AHumanoidStandup, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AHumanoidStandup)

class MATd3MaMujocoManyAgentSwimmer(MaMujocoManyAgentSwimmer, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoManyAgentSwimmer)

class MATd3MaMujocoCoupledHalfCheetah(MaMujocoCoupledHalfCheetah, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoCoupledHalfCheetah)

class MATd3MaMujocoManyAgentAnt(MaMujocoManyAgentAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoManyAgentAnt)
######################################################################################################


######################################################################################################
# EXPMATD3

class EXPMATD3MaMujoco2AAnt(MaMujoco2AAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AAnt)

class EXPMATD3MaMujoco4AAnt(MaMujoco4AAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco4AAnt)

class EXPMATD3MaMujoco6AHalfcheetah(MaMujoco6AHalfcheetah, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco6AHalfcheetah)

class EXPMATD3MaMujoco3AHopper(MaMujoco3AHopper, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco3AHopper)

class EXPMATD3MaMujoco2AHumanoid(MaMujoco2AHumanoid, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AHumanoid)

class EXPMATD3MaMujoco2AHumanoidStandup(MaMujoco2AHumanoidStandup, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AHumanoidStandup)

class EXPMATD3MaMujocoManyAgentSwimmer(MaMujocoManyAgentSwimmer, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoManyAgentSwimmer)

class EXPMATD3MaMujocoCoupledHalfCheetah(MaMujocoCoupledHalfCheetah, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoCoupledHalfCheetah)

class EXPMATD3MaMujocoManyAgentAnt(MaMujocoManyAgentAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoManyAgentAnt)
######################################################################################################
