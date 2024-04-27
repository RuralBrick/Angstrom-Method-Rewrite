from typing import Optional
from pathlib import Path

from pyangstrom.caching import Cache
from pyangstrom.io import load_exp_condition
from pyangstrom.wrappers.config import exp_condition_to_config
from pyangstrom.pipeline import analyze_recording


def hu_batch_process(
        df_exp_condition_filename: str,
        code_directory: str | Path,
        data_directory: Optional[str | Path] = None,
        *,
        memory_cache: Optional[Cache] = None,
        recording_cache_path: Optional[str | Path] = None,
):
    code_directory = Path(code_directory)
    if data_directory:
        data_directory = Path(data_directory)
    else:
        data_directory = code_directory / 'temperature data'
    exp_cond = load_exp_condition(
        code_directory / 'batch process information' / df_exp_condition_filename
    )
    exp_condition_to_config(exp_cond)
    for config in []:
        recording_name = ...
        analyze_recording(
            data_directory / recording_name,
            config,
            memory_cache=memory_cache,
            recording_cache_path=recording_cache_path
        )
