from typing import TypedDict, NotRequired

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionConfig, RegionBatchConfig
from pyangstrom.signal import SignalProcessor
from pyangstrom.fit import Solver, Fitter


class Config(TypedDict):
    experimental_setup: ExperimentalSetup
    region_information: RegionConfig | RegionBatchConfig | list[RegionConfig] | list[RegionBatchConfig]
    signal_processor: NotRequired[SignalProcessor]
    solver: NotRequired[Solver]
    fitter: NotRequired[Fitter]
