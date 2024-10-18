from typing import TypedDict


class MaterialProperties(TypedDict):
    """The material properties of the recorded sample."""
    specific_heat_capacity_J__kg_K: float
    density_kg__m3: float

class ExperimentalSetup(TypedDict):
    """Constant experimental conditions of the recording.

    Attributes
    ----------
    heating_frequency_hertz
        The frequency at which the heat source is turned on and off.
    meters_per_pixel
        The width one pixel covers on the recorded sample.
    material_properties
        Material properties of the recorded sample.
    """
    heating_frequency_hertz: float
    meters_per_pixel: float
    material_properties: MaterialProperties
