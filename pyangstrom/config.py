from typing import TypedDict, NotRequired

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import RegionConfig, RegionBatchConfig
from pyangstrom.fit import Solver, Fitter


class Config(TypedDict):
    experimental_setup: ExperimentalSetup
    region_information: RegionConfig | list[RegionConfig]
    region_batch_transform: NotRequired[RegionBatchConfig]
    solver: NotRequired[Solver]
    fitter: NotRequired[Fitter]
