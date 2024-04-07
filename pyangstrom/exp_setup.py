from typing import TypedDict


class MaterialProperties(TypedDict):
    specific_heat_capacity: float
    density: float

class ExperimentalSetup(TypedDict):
    heating_frequency: float
    meters_per_pixel: float
    material_properties: MaterialProperties
