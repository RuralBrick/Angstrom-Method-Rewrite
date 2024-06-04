import logging
from typing import Optional
from pathlib import Path

from pyangstrom.wrappers.config import exp_condition_to_config
from pyangstrom.wrappers.io import load_exp_condition
from pyangstrom.config import Config
from pyangstrom.pipeline import analyze_recording
from pyangstrom.io import save_config
from pyangstrom.caching import Cache


logger = logging.getLogger('pipeline')

def convert_old_csv_to_new_config(
        exp_condition_path: str | Path,
        config_directory_path: Optional[str | Path] = None,
) -> dict[str, Config]:
    p_exp_cond = Path(exp_condition_path)
    exp_cond = load_exp_condition(p_exp_cond)
    logger.debug(f'{exp_cond=}')
    dict_config = exp_condition_to_config(exp_cond)
    if config_directory_path:
        config_directory_path = Path(config_directory_path)
        for recording_name, config in dict_config.items():
            save_config(config, config_directory_path, recording_name)
    return dict_config

def hu_batch_process(
        df_exp_condition_filename: str,
        code_directory: str | Path,
        data_directory: Optional[str | Path] = None,
        *,
        debug: bool = False,
        memory_cache: Optional[Cache] = None,
        config_directory_path: Optional[str | Path] = None,
        recording_cache_path: Optional[str | Path] = None,
):
    """Compute thermal conductivity using old configuration.

    Arguments
    ---------
    df_exp_condition_filename
        Name of the CSV file containing batch processing information; equivalent
        to HT's df_exp_condition_spreadsheet_filename and RT's
        df_exp_condition_spreadsheet_filename, df_exp_condition_file_name, and
        df_exp_conditions_file_name.
    code_directory
        Path to the directory holding information, dump, and optionally data
        files; equivalent to HT's code_directory and RT's directory_path, but
        also supports Path objects.
    data_directory
        Path to the directory holding data files; equivalent to HT's
        data_directory, but also supports Path objects.

    Keyword Arguments
    -----------------
    verbose
        If True, print status messages while processing data.
    memory_cache
        Holds intermediate results in memory, allowing for faster reruns.
    config_directory_path
        Path to a directory in which new config files equivalent to
        df_exp_condition_filename will be saved.
    recording_cache_path
        Path to a directory in which cached IR camera data will be saved.
    """
    code_directory = Path(code_directory)
    if data_directory:
        data_directory = Path(data_directory)
    else:
        data_directory = code_directory / 'temperature data'
    logger.info("Loading exp_condition")
    p_exp_cond = code_directory / 'batch process information' / df_exp_condition_filename
    dict_config = convert_old_csv_to_new_config(
        p_exp_cond,
        config_directory_path,
    )
    logger.debug(f'{dict_config=}')
    results = []
    for recording_name, config in dict_config.items():
        logger.info(f"Processing {recording_name}")
        result = analyze_recording(
            data_directory / recording_name,
            config,
            debug=debug,
            memory_cache=memory_cache,
            recording_cache_path=recording_cache_path,
        )
        results.append(result)
    return results
