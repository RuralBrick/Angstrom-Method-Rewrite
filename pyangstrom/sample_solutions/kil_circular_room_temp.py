from typing import NamedTuple

import numpy as np
from scipy.special import jv, yv

from pyangstrom.fit import ExperimentalSetup, SignalProperties


class KilCircularRoomTempUnknowns(NamedTuple):
    thermal_diffusivity_m2_s: float
    convective_heat_transfer_term: float

def calc_convective_heat_transfer_coefficient():
    pass

def solve(
        unknowns: KilCircularRoomTempUnknowns,
        displacements_meters: np.ndarray,
        setup: ExperimentalSetup,
        min_radius_meters: float,
        max_radius_meters: float,
) -> SignalProperties:
    thermal_diffusivity_m2_s, convective_heat_transfer_term = unknowns
    angular_frequency = 2.0 * np.pi * setup['heating_frequency_hertz']

    temp_var1 = convective_heat_transfer_term * convective_heat_transfer_term
    temp_var2 = 1j * angular_frequency / thermal_diffusivity_m2_s

    temp_var3 = np.sqrt(temp_var1 + temp_var2)
    temp_var4 = None

    temp_var5 = np.sqrt(temp_var1 - temp_var2)


    raise NotImplementedError()
