import numpy as np


def wavenumber(
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

def xi(wavenumber, length, displacement):
    temp_var1 = np.cos(wavenumber * (length - displacement))
    temp_var2 = np.cos(wavenumber * length)
    xi = temp_var1 / temp_var2
    return xi
