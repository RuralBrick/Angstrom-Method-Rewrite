from typing import TypedDict

from transform import Geometry


class ExperimentalSetup(TypedDict):
    heating_frequency: float

class MaterialProperties(TypedDict):
    specific_heat_capacity: float
    density: float

class Config(TypedDict):
    region_geometry: Geometry
    experimental_setup: ExperimentalSetup
