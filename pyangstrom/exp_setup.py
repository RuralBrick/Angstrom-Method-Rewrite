from typing import TypedDict


class MaterialProperties(TypedDict):
    specific_heat_capacity_J_kg_K: float
    density_kg_m3: float

class ExperimentalSetup(TypedDict):
    heating_frequency_hertz: float
    meters_per_pixel: float
    material_properties: MaterialProperties
