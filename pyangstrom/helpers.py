def thermal_conductivity(thermal_diffusivity, specific_heat_capacity, density):
    thermal_conductivity = (thermal_diffusivity
                            * specific_heat_capacity
                            * density)
    return thermal_conductivity

def thermal_diffusivity(thermal_conductivity, specific_heat_capacity, density):
    temp_var = specific_heat_capacity * density
    thermal_diffusivity = thermal_conductivity / temp_var
    return thermal_diffusivity
