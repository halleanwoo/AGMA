from .q_learner import QLearner
from .coma_learner import COMALearner
from .s_learner import SLearner
from .q_learner_convention import QLearnerConvention
from .q_learner_perception import QLearnerPerception

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner

REGISTRY["s_learner"] = SLearner
REGISTRY["q_learner_convention"] = QLearnerConvention
REGISTRY["q_learner_perception"] = QLearnerPerception
