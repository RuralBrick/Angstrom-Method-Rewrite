from typing import TypedDict

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionInformation
from pyangstrom.signal import SignalProcessor
from pyangstrom.fit import Solver, Fitter


class Config(TypedDict, total=False):
    experimental_setup: ExperimentalSetup
    region_information: RegionInformation
    signal_processor: SignalProcessor
    solver: Solver
    fitter: Fitter
