REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent_convention import RNNAgentWithConvention
from .rnn_agent_perception import RNNAgentWithPerception

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_convention"] = RNNAgentWithConvention
REGISTRY["rnn_perception"] = RNNAgentWithPerception
