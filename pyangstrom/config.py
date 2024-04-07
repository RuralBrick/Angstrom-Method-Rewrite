from typing import TypedDict, NotRequired

from pyangstrom.transform import Geometry
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.fit import Solver, Fitter


class Config(TypedDict):
    region_geometry: Geometry
    experimental_setup: ExperimentalSetup
    solver: NotRequired[Solver]
    fitter: NotRequired[Fitter]
