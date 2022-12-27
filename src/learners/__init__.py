from .q_learner import QLearner
from .exp_q_learner import ExpQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .exp_qtran_learner import ExpQLearner as ExpQTranLearner
from .exp_matd3_learner import ExpMATd3Learner
from .matd3_learner import MATd3Learner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .exp_dmaq_qatten_learner import ExpDMAQ_qattenLearner
from .max_q_learner import MAXQLearner
from .exp_max_q_learner import ExpMAXQLearner
from .semi_emc_q_learner import SemiEMCQLearner
from .rnd_q_learner import RNDQLearner
from .icm_q_learner import IcmQLearner
from .noise_q_learner import QLearner as NoiseQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["exp_q_learner"] = ExpQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["exp_qtran_learner"] = ExpQTranLearner
REGISTRY["matd3_learner"] = MATd3Learner
REGISTRY["exp_matd3_learner"] = ExpMATd3Learner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["exp_dmaq_qatten_learner"] = ExpDMAQ_qattenLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["exp_max_q_learner"] = ExpMAXQLearner
REGISTRY["semi_emc_q_learner"] = SemiEMCQLearner
REGISTRY["rnd_q_learner"] = RNDQLearner
REGISTRY["icm_q_learner"] = IcmQLearner
REGISTRY["noise_q_learner"] = NoiseQLearner
