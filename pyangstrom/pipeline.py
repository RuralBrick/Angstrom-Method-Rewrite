import warnings
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from pyangstrom.config import Config
from pyangstrom.caching import Cache
from pyangstrom.io import (
    recording_cache_exists,
    load_recording_csv,
    save_recording_cache,
    load_recording_cache,
)
from pyangstrom.transform import fully_extract_region
from pyangstrom.signal import fft_signal_processing
from pyangstrom.fit import PropertiesCalculator, UnknownsFormatter, fit
import pyangstrom.sample_solutions.lopez_baeza_short as lopez_baeza_short
import pyangstrom.sample_solutions.hu_high_temp as hu_high_temp
import pyangstrom.fitting_methods.lsr as lsr
import pyangstrom.fitting_methods.mcmc as mcmc


@dataclass
class Solution:
    solver: PropertiesCalculator
    unknowns_formatter: UnknownsFormatter

SIGNAL_PROCESSORS = {
    'fft': fft_signal_processing,
}

SOLUTIONS = {
    'lopez-baeza': Solution(
        lopez_baeza_short.calc_props,
        lopez_baeza_short.LopezBaezaShortUnknowns,
    ),
}

FITTERS = {
    'lsr': lsr.fitter,
}

def analyze_recording(
        recording_path: str | Path,
        config: str | Path | Config,
        *,
        memory_cache: Optional[Cache] = None,
        recording_cache_path: Optional[str | Path] = None,
):
    """Compute the thermal conductivity of the recorded sample.

    Arguments
    ---------
    recording_path
        Path to the directory holding CSV IR camera data
    config
        Path to a YAML config file or the deserialized config object itself

    Keyword Arguments
    -----------------
    memory_cache
        Holds intermediate results in memory, allowing for faster reruns
    recording_cache_path
        Path to a directory in which cached IR camera data will be saved
    """
    recording_path = Path(recording_path)
    if recording_cache_path:
        recording_cache_path = Path(recording_cache_path)
        if recording_cache_exists(recording_cache_path, recording_path.stem):
            df_recording = load_recording_cache(
                recording_cache_path,
                recording_cache_path.stem,
            )
        else:
            df_recording = load_recording_csv(recording_path)
            save_recording_cache(
                df_recording,
                recording_cache_path,
                recording_path.stem,
            )
    else:
        df_recording = load_recording_csv(recording_path)
    region_result = fully_extract_region(
        df_recording,
        config['region_information'],
        config['experimental_setup'],
    )
    if 'signal_processor' not in config:
        return region_result
    try:
        # TODO: apply_filter
        signal_processor = SIGNAL_PROCESSORS[config['signal_processor']['name']]
        if isinstance(region_result, list):
            props = [
                signal_processor(r, config['experimental_setup'])
                for r in region_result
            ]
        else:
            props = signal_processor(
                region_result,
                config['experimental_setup'],
            )
    except KeyError as e:
        warnings.warn(f"Signal processor {e} not found.")
        return region_result
    # TODO: Custom solver
    if 'solver' not in config or 'fitter' not in config:
        return props
    try:
        if isinstance(props, list):
            fitting_result = [
                fit(
                    p,
                    SOLUTIONS[config['solver']['name']].solver,
                    config['solver']['parameters'],
                    SOLUTIONS[config['solver']['name']].unknowns_formatter,
                    FITTERS[config['fitter']['name']],
                    config['fitter']['parameters'],
                    config['fitter']['guesses'],
                    r,
                    config['experimental_setup'],
                )
                for p, r in zip(props, region_result)
            ]
        else:
            fitting_result = fit(
                props,
                SOLUTIONS[config['solver']['name']].solver,
                config['solver']['parameters'],
                SOLUTIONS[config['solver']['name']].unknowns_formatter,
                FITTERS[config['fitter']['name']],
                config['fitter']['parameters'],
                config['fitter']['guesses'],
                region_result,
                config['experimental_setup'],
            )
    except KeyError as e:
        warnings.warn(f"Solver or fitter {e} not found.")
        return props
    return fitting_result
    # TODO: Automatic visualizations/better formatted return
