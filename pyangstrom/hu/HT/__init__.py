from typing import overload

import pandas as pd
from pyangstromHT.high_T_angstrom_method import (
    parallel_temperature_average_batch_experimental_results,
)

from pyangstrom.hu.HT.signatures import HtAmpPhaseConfig


@overload
def parallel_temperature_average_batch_experimental_results(
        df_exp_condition_spreadsheet_filename: str,
        data_directory: str,
        num_cores: int,
        code_directory: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    ...
