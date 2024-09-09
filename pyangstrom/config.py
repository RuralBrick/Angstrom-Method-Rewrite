from typing import TypedDict

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionInformation
from pyangstrom.signal import SignalProcessorInformation
from pyangstrom.fit import SolverInformation, FitterInformation


class Config(TypedDict, total=False):
    # TODO: Docstring
    experimental_setup: ExperimentalSetup
    region_information: RegionInformation
    signal_processor: SignalProcessorInformation
    solver: SolverInformation
    fitter: FitterInformation
