from typing import overload

import pandas as pd
import numpy as np
import pyangstromRT.blmcmc
from pyangstromRT.blmcmc import calculate_theoretical_results

from pyangstrom.hu.RT.signatures import (
    RtFileConfig,
    RtAmpPhaseConfig,
    LsrConfig,
    TheorConfig,
)


class temperature_preprocessing_extract_phase_amplitude(
    pyangstromRT.blmcmc.temperature_preprocessing_extract_phase_amplitude
):
    def __init__(self, analysis_region: RtAmpPhaseConfig) -> None:
        super().__init__(analysis_region)

class least_square_regression_Angstrom(
    pyangstromRT.blmcmc.least_square_regression_Angstrom
):
    def __init__(
            self,
            params_init: tuple[int, int],
            analysis_region: LsrConfig,
            df_phase_diff_amp_ratio: pd.DataFrame,
            material_properties: LsrConfig,
    ) -> None:
        super().__init__(
            params_init,
            analysis_region,
            df_phase_diff_amp_ratio,
            material_properties,
        )

@overload
def calculate_theoretical_results(
        material_properties: TheorConfig,
        analysis_region: TheorConfig,
        df_phase_diff_amp_ratio: pd.DataFrame,
        params: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ...
