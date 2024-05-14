from typing import NamedTuple

import numpy as np

from pyangstrom.fit import ExperimentalSetup, SignalProperties


class KilCircularRoomTempUnknowns(NamedTuple):
    thermal_diffusivity_m2_s: float
    thermal_transfer_coefficient_kg_s2_K_m2: float

def solve(
        unknowns: KilCircularRoomTempUnknowns,
        displacements_meters: np.ndarray,
        setup: ExperimentalSetup,
) -> SignalProperties:
    raise NotImplementedError()
