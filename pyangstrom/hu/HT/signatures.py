from typing import TypedDict


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
