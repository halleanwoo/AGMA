REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_convention import BasicMACConvention
from .basic_controller_perception import BasicMACPerception

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_convention"] = BasicMACConvention
REGISTRY["basic_mac_perception"] = BasicMACPerception

