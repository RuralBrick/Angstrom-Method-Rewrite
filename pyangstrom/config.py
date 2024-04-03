from typing import TypedDict


class ExperimentalSetup(TypedDict):
    heating_frequency: float

class MaterialProperties(TypedDict):
    specific_heat_capacity: float
    density: float
