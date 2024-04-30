from typing import TypedDict, NotRequired

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionInformation
from pyangstrom.signal import SignalProcessor
from pyangstrom.fit import Solver, Fitter


class Config(TypedDict):
    experimental_setup: ExperimentalSetup
    region_information: RegionInformation
    signal_processor: NotRequired[SignalProcessor]
    solver: NotRequired[Solver]
    fitter: NotRequired[Fitter]
