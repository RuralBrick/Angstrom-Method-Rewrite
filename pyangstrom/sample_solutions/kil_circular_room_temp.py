from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.special import jv, yv

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins, Region
from pyangstrom.signal import fft_signal_processing, SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations


class KilCircularRoomTempUnknowns(TypedDict):
    thermal_diffusivity_m2__s: float
    convective_heat_transfer_term: float

def J0(x):
    return jv(0, x)

def Y0(x):
    return yv(0, x)

class Solution(NelderMeadEquations, LsrEquations):
    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            heating_source_radius_meters: float,
            outer_boundary_radius_meters: float,
    ) -> None:
        self.margins = margins
        disp_axes = len(margins.displacements_meters.shape) * [np.newaxis]
        self.time_seconds = margins.seconds_elapsed[:, *disp_axes]
        self.radii_meters = margins.displacements_meters
        self.heating_source_radius_meters = heating_source_radius_meters
        self.outer_boundary_radius_meters = outer_boundary_radius_meters
        self.setup = setup
        self.angular_frequency_hertz = 2*np.pi*setup['heating_frequency_hertz']

    def calc_convective_heat_transfer_coefficient(self):
        raise NotImplementedError()

    def calc_normalized_T1(
            self,
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
    ):
        i = 1j
        r = self.radii_meters
        a = self.heating_source_radius_meters
        b = self.outer_boundary_radius_meters
        w = self.angular_frequency_hertz
        D = thermal_diffusivity_m2__s
        m = convective_heat_transfer_term

        X = i * np.sqrt(m*m + i*w/D)
        numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
        denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
        T1 = 0.5 * numerator / denominator

        return T1

    def calc_normalized_T2(
            self,
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
    ):
        i = 1j
        r = self.radii_meters
        a = self.heating_source_radius_meters
        b = self.outer_boundary_radius_meters
        w = self.angular_frequency_hertz
        D = thermal_diffusivity_m2__s
        m = convective_heat_transfer_term

        X = i * np.sqrt(m*m - i*w/D)
        numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
        denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
        T2 = 0.5 * numerator / denominator

        return T2

    def calc_normalized_temps(
            self,
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
    ):
        i = 1j
        t = self.time_seconds
        w = self.angular_frequency_hertz
        T1 = self.calc_normalized_T1(
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
        )
        T2 = self.calc_normalized_T2(
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
        )

        T = T1 * np.exp(i*w*t) + T2 * np.exp(-i*w*t)

        return T

    def unknowns_to_vector(
            self,
            unknowns: KilCircularRoomTempUnknowns,
    ) -> np.ndarray:
        vector = np.array([
            unknowns['thermal_diffusivity_m2__s'],
            unknowns['convective_heat_transfer_term'],
        ])
        return vector

    def vector_to_unknowns(
            self,
            vector: np.ndarray
    ) -> KilCircularRoomTempUnknowns:
        unknowns: KilCircularRoomTempUnknowns = {
            'thermal_diffusivity_m2__s': vector[0],
            'convective_heat_transfer_term': vector[1],
        }
        return unknowns

    def solve(self, unknowns: KilCircularRoomTempUnknowns) -> SignalProperties:
        timestamps = pd.DatetimeIndex(
            1e9 * self.margins.seconds_elapsed,
            dtype='datetime64[ns]',
        )
        temps = self.calc_normalized_temps(
            unknowns['thermal_diffusivity_m2__s'],
            unknowns['convective_heat_transfer_term'],
        )
        region = Region(timestamps, temps, self.margins)
        return fft_signal_processing(region, self.setup)

    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties:
        thermal_diffusivity_m2__s, convective_heat_transfer_term = unknowns_vector
        timestamps = pd.DatetimeIndex(
            1e9 * self.margins.seconds_elapsed,
            dtype='datetime64[ns]',
        )
        temps = self.calc_normalized_temps(
            thermal_diffusivity_m2__s,
            convective_heat_transfer_term,
        )
        region = Region(timestamps, temps, self.margins)
        return fft_signal_processing(region, self.setup)
