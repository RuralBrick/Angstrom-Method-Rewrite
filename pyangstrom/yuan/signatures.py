from typing import TypedDict


class RtFileConfig(TypedDict):
    directory_path: str
    file_name: str
    file_path: str # = directory_path / "temperature data" / file_name /
    file_skip: str # = 0
    px: float # = 25/10**6

class RtAmpPhaseConfig(RtFileConfig):
    """For temperature_preprocessing_extract_phase_amplitude"""
    x_heater: int
    y_heater: int
    x_region_line_center: int
    y_region_line_center: int
    dx: int
    dy: int
    gap: int
    direction: str
    f_heating: float
    analysis_method: str

class HtAmpPhaseConfig(TypedDict):
    """For parallel_temperature_average_batch_experimental_results"""
    rec_name: str
    x0_pixels: int
    y0_pixels: int
    R0_pixels: int
    R_analysis_pixels: int
    Nr_pixels: int
    gap_pixels: int
    anguler_range: str
    f_heating: float
    focal_shift: float
    V_DC: float
    exp_amp_phase_extraction_method: str

class LsrConfig(TypedDict):
    """For least_square_regression_Angstrom"""
    rec_name: str
    f_heating: float
    L: float
    r: float
    cp: float
    rho: float

class TheorConfig(TypedDict):
    """For calculate_theoretical_results"""
    L: float
    r: float
    cp: float
    rho: float
    f_heating: float
