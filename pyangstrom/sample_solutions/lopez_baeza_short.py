from typing import NamedTuple, TypedDict

import numpy as np

from pyangstrom.fit import Displacement, ExperimentalSetup, SignalProperties
from pyangstrom.helpers import calc_thermal_conductivity


class LopezBaezaShortUnknowns(NamedTuple):
    thermal_diffusivity_m2_s: float
    thermal_transfer_coefficient_kg_s2_K_m2: float

class LogLopezBaezaShortUnknowns(NamedTuple):
    thermal_diffusivity_log10_m2_s: float
    thermal_transfer_coefficient_log10_kg_s2_K_m2: float

class LopezBaezaShortParameters(TypedDict):
    r_meters: float
    length_meters: float

def calc_wavenumber(
        angular_frequency_hertz,
        thermal_diffusivity_m2_s,
        thermal_transfer_coefficient_kg_s2_K_m2,
        r_meters,
        thermal_conductivity_W_m_K,
):
    heat_conduction = angular_frequency_hertz / (2.0*thermal_diffusivity_m2_s)
    thermal_losses = thermal_transfer_coefficient_kg_s2_K_m2 / (r_meters * thermal_conductivity_W_m_K)
    temp_var1 = np.sqrt(thermal_losses**2.0 + heat_conduction**2.0)
    temp_var2 = np.sqrt(-thermal_losses + temp_var1)
    temp_var3 = 1.0j*np.sqrt(thermal_losses + temp_var1)
    wavenumber = temp_var2 + temp_var3
    return wavenumber

def calc_xi(wavenumber, length, displacement):
    temp_var1 = np.cos(wavenumber * (length - displacement))
    temp_var2 = np.cos(wavenumber * length)
    xi = temp_var1 / temp_var2
    return xi

def calc_props(
        unknowns: LopezBaezaShortUnknowns,
        displacement: Displacement,
        setup: ExperimentalSetup,
        params: LopezBaezaShortParameters,
) -> SignalProperties:
    thermal_diffusivity_m2_s, thermal_transfer_coefficient_kg_s2_K_m2 = unknowns
    wavenumber = calc_wavenumber(
        2 * np.pi * setup['heating_frequency_hertz'],
        thermal_diffusivity_m2_s,
        thermal_transfer_coefficient_kg_s2_K_m2,
        params['r_meters'],
        calc_thermal_conductivity(
            thermal_diffusivity_m2_s,
            setup['material_properties']['specific_heat_capacity_J_kg_K'],
            setup['material_properties']['density_kg_m3'],
        ),
    )
    xi = calc_xi(
        wavenumber,
        params['length_meters'],
        displacement,
    )

    amps = np.abs(xi)
    amp_ratio = amps / amps[0]

    phases = np.angle(xi)
    phase_diff = phases - phases[0]

    return SignalProperties(amp_ratio, phase_diff)

def log_calc_props(
        unknowns: LogLopezBaezaShortUnknowns,
        displacement: Displacement,
        setup: ExperimentalSetup,
        params: LopezBaezaShortParameters,
):
    return calc_props(np.power(10.0, unknowns), displacement, setup, params)
