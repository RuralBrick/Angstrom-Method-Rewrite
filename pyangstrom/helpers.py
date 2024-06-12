def calc_thermal_conductivity(
        thermal_diffusivity_m2__s,
        specific_heat_capacity_J__kg_K,
        density_kg__m3,
):
    thermal_conductivity_W_m_K = (thermal_diffusivity_m2__s
                                  * specific_heat_capacity_J__kg_K
                                  * density_kg__m3)
    return thermal_conductivity_W_m_K

def calc_thermal_diffusivity(
        thermal_conductivity_W_m_K,
        specific_heat_capacity_J__kg_K,
        density_kg__m3,
):
    temp_var = specific_heat_capacity_J__kg_K * density_kg__m3
    thermal_diffusivity_m2__s = thermal_conductivity_W_m_K / temp_var
    return thermal_diffusivity_m2__s
