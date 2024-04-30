def calc_thermal_conductivity(
        thermal_diffusivity_m2_s,
        specific_heat_capacity_J_kg_K,
        density_kg_m3,
):
    thermal_conductivity_W_m_K = (thermal_diffusivity_m2_s
                                  * specific_heat_capacity_J_kg_K
                                  * density_kg_m3)
    return thermal_conductivity_W_m_K

def calc_thermal_diffusivity(
        thermal_conductivity_W_m_K,
        specific_heat_capacity_J_kg_K,
        density_kg_m3,
):
    temp_var = specific_heat_capacity_J_kg_K * density_kg_m3
    thermal_diffusivity_m2_s = thermal_conductivity_W_m_K / temp_var
    return thermal_diffusivity_m2_s
