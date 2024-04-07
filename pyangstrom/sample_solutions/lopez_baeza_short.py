from typing import NamedTuple, TypedDict

import numpy as np

from pyangstrom.fit import Region, ExperimentalSetup, SignalProperties
from pyangstrom.helpers import calc_thermal_conductivity


class LopezBaezaShortUnknowns(NamedTuple):
    thermal_diffusivity: float
    thermal_transfer_coefficient: float

class LopezBaezaShortParameters(TypedDict):
    r: float
    length: float

def calc_wavenumber(
        angular_frequency,
        thermal_diffusivity,
        thermal_transfer_coefficient,
        r,
        thermal_conductivity,
):
    heat_conduction = angular_frequency / (2.0*thermal_diffusivity)
    thermal_losses = thermal_transfer_coefficient / (r * thermal_conductivity)
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
        region: Region,
        setup: ExperimentalSetup,
        params: LopezBaezaShortParameters,
) -> SignalProperties:
    thermal_diffusivity, thermal_transfer_coefficient = unknowns
    wavenumber = calc_wavenumber(
        2 * np.pi * setup['heating_frequency'],
        thermal_diffusivity,
        thermal_transfer_coefficient,
        params['r'],
        calc_thermal_conductivity(
            thermal_diffusivity,
            setup['material_properties']['specific_heat_capacity'],
            setup['material_properties']['density'],
        ),
    )
    disp = np.linspace(0, region.margins[1], region.temps.shape[1])
    xi = calc_xi(
        wavenumber,
        params['length'],
        disp,
    )

    amps = np.abs(xi)
    amp_ratio = amps / amps[0]

    phases = np.angle(xi)
    phase_diff = phases - phases[0]

    return SignalProperties(amp_ratio, phase_diff)
