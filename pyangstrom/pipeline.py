import logging
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
    load_config,
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

logger = logging.getLogger('pipeline')

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
    logger.info("Loading recording")
    if recording_cache_path:
        recording_cache_path = Path(recording_cache_path)
        if recording_cache_exists(recording_cache_path, recording_path.stem):
            logger.debug("Recording cache found")
            df_recording = load_recording_cache(
                recording_cache_path,
                recording_path.stem,
            )
            logger.debug(f"{df_recording[:1]=}")
        else:
            logger.debug("Recording cache not found")
            df_recording = load_recording_csv(recording_path)
            save_recording_cache(
                df_recording,
                recording_cache_path,
                recording_path.stem,
            )
            logger.debug(f"Saved to cache: {df_recording[:1]=}")
    else:
        df_recording = load_recording_csv(recording_path)
        logger.debug(f"Loaded csv: {df_recording[:1]=}")
    if not isinstance(config, dict):
        logger.info("Loading config")
        config = load_config(Path(config))
        logger.debug(f"{config=}")
    logger.info("Extracting region(s)")
    region_result = fully_extract_region(
        df_recording,
        config['region_information'],
        config['experimental_setup'],
    )
    if 'signal_processor' not in config:
        return region_result
    if isinstance(region_result, list):
        debug_region = region_result[0]
    else:
        debug_region = region_result
    logger.debug(
        f"{debug_region.time[:1]=}, {debug_region.temps_kelvin[:1, :1]=}, "
        f"{debug_region.margins=}"
    )
    logger.info("Signal processing")
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
    if isinstance(props, list):
        debug_props = props[0]
    else:
        debug_props = props
    for name, prop in debug_props._asdict().items():
        logger.debug(f"{name}={prop[:1]}")
    logger.info("Fitting")
    try:
        fit_kwargs = dict(
            calc_props=SOLUTIONS[config['solver']['name']].solver,
            guess_converter=SOLUTIONS[config['solver']['name']].unknowns_formatter,
            fitter=FITTERS[config['fitter']['name']],
            guesses=config['fitter']['guesses'],
            setup=config['experimental_setup'],
        )
        if 'parameters' in config['solver']:
            fit_kwargs['solver_parameters'] = config['solver']['parameters']
        if 'parameters' in config['fitter']:
            fit_kwargs['fitter_parameters'] = config['fitter']['parameters']
        if isinstance(props, list):
            fitting_result = [
                fit(r, p, **fit_kwargs) for r, p in zip(region_result, props)
            ]
        else:
            fitting_result = fit(region_result, props, **fit_kwargs,)
    except KeyError as e:
        warnings.warn(f"Solver or fitter {e} not found.")
        return props
    return fitting_result
    # TODO: Automatic visualizations/better formatted return
