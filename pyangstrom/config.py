from typing import TypedDict

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionInformation
from pyangstrom.signal import SignalProcessorInformation
from pyangstrom.fit import SolverInformation, FitterInformation


class Config(TypedDict, total=False):
    """The full specification of how a recording should be processed."""
    experimental_setup: ExperimentalSetup
    region_information: RegionInformation
    signal_processor: SignalProcessorInformation
    solver: SolverInformation
    fitter: FitterInformation
