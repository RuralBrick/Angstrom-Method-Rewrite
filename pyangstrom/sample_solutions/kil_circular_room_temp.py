from typing import NamedTuple

import numpy as np
from scipy.special import jv, yv

from pyangstrom.fit import ExperimentalSetup, SignalProperties


class KilCircularRoomTempUnknowns(NamedTuple):
    thermal_diffusivity_m2_s: float
    convective_heat_transfer_term: float

def calc_convective_heat_transfer_coefficient():
    pass

def J0(x):
    return jv(0, x)

def Y0(x):
    return yv(0, x)

def calc_normalized_T1(
        radii_meters,
        min_radius_meters,
        max_radius_meters,
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        convective_heat_transfer_term,
):
    i = 1j
    r = radii_meters
    a = min_radius_meters
    b = max_radius_meters
    w = angular_frequency_hertz
    A = thermal_diffusivity_m2_s
    m = convective_heat_transfer_term

    X = i * np.sqrt(m*m + i*w/A)
    numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
    denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
    T1 = 0.5 * numerator / denominator

    return T1

def calc_normalized_T2(
        radii_meters,
        min_radius_meters,
        max_radius_meters,
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        convective_heat_transfer_term,
):
    i = 1j
    r = radii_meters
    a = min_radius_meters
    b = max_radius_meters
    w = angular_frequency_hertz
    A = thermal_diffusivity_m2_s
    m = convective_heat_transfer_term

    X = i * np.sqrt(m*m - i*w/A)
    numerator = -Y0(-b*X) * J0(r*X) + J0(b*X) * Y0(-r*X)
    denominator = J0(b*X) * Y0(-a*X) - J0(a*X) * Y0(-b*X)
    T2 = 0.5 * numerator / denominator

    return T2

def calc_normalized_temps(
        radii_meters,
        time_seconds,
        min_radius_meters,
        max_radius_meters,
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        convective_heat_transfer_term,
):
    i = 1j
    t = time_seconds
    w = angular_frequency_hertz
    T1 = calc_normalized_T1(
        radii_meters,
        min_radius_meters,
        max_radius_meters,
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        convective_heat_transfer_term,
    )
    T2 = calc_normalized_T2(
        radii_meters,
        min_radius_meters,
        max_radius_meters,
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        convective_heat_transfer_term,
    )

    T = T1 * np.exp(i*w*t) + T2 * np.exp(-i*w*t)

    return T

def solve(
        unknowns: KilCircularRoomTempUnknowns,
        displacements_meters: np.ndarray,
        setup: ExperimentalSetup,
        min_radius_meters: float,
        max_radius_meters: float,
) -> SignalProperties:
    thermal_diffusivity_m2_s, convective_heat_transfer_term = unknowns
    angular_frequency = 2.0 * np.pi * setup['heating_frequency_hertz']


    raise NotImplementedError()
