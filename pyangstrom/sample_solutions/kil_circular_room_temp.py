from typing import TypedDict

import numpy as np
from scipy.special import jv, yv

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import RegionProperties, SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations
from pyangstrom.fitting_methods.metropolis_hastings import MetropolisHastingsEquations


class KilCircularRoomTempUnknowns(TypedDict):
    thermal_diffusivity_m2_s: float
    convective_heat_transfer_term: float

def J0(x):
    return jv(0, x)

def Y0(x):
    return yv(0, x)

class Solution(NelderMeadEquations, LsrEquations, MetropolisHastingsEquations):
    def __init__(
            self,
            region_properties: RegionProperties,
            setup: ExperimentalSetup,
            min_radius_meters: float,
            max_radius_meters: float,
    ) -> None:
        # HACK: Actually need to use 2D array
        self.time_seconds = region_properties.seconds_elapsed
        self.radii_meters = (region_properties.displacements_meters
                             + min_radius_meters)
        # end HACK
        self.angular_frequency_hertz = 2*np.pi*setup['heating_frequency_hertz']
        self.min_radius_meters = min_radius_meters
        self.max_radius_meters = max_radius_meters

    def calc_convective_heat_transfer_coefficient(self):
        raise NotImplementedError()

    def calc_normalized_T1(
            self,
            thermal_diffusivity_m2_s,
            convective_heat_transfer_term,
    ):
        i = 1j
        r = self.radii_meters
        a = self.min_radius_meters
        b = self.max_radius_meters
        w = self.angular_frequency_hertz
        D = thermal_diffusivity_m2_s
        m = convective_heat_transfer_term

        X = i * np.sqrt(m*m + i*w/D)
        numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
        denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
        T1 = 0.5 * numerator / denominator

        return T1

    def calc_normalized_T2(
            self,
            thermal_diffusivity_m2_s,
            convective_heat_transfer_term,
    ):
        i = 1j
        r = self.radii_meters
        a = self.min_radius_meters
        b = self.max_radius_meters
        w = self.angular_frequency_hertz
        A = thermal_diffusivity_m2_s
        m = convective_heat_transfer_term

        X = i * np.sqrt(m*m - i*w/A)
        numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
        denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
        T2 = 0.5 * numerator / denominator

        return T2

    def calc_normalized_temps(
            self,
            thermal_diffusivity_m2_s,
            convective_heat_transfer_term,
    ):
        i = 1j
        t = self.time_seconds
        w = self.angular_frequency_hertz
        T1 = self.calc_normalized_T1(
            thermal_diffusivity_m2_s,
            convective_heat_transfer_term,
        )
        T2 = self.calc_normalized_T2(
            thermal_diffusivity_m2_s,
            convective_heat_transfer_term,
        )

        T = T1 * np.exp(i*w*t) + T2 * np.exp(-i*w*t)

        return T

    def solve(self, unknowns: KilCircularRoomTempUnknowns) -> SignalProperties:
        raise NotImplementedError()
