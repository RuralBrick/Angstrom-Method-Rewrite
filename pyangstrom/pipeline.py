import logging
import warnings
from typing import Optional
from pathlib import PurePath, Path

import matplotlib.pyplot as plt

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
from pyangstrom.signal import signal_process_region
from pyangstrom.fit import autofit
from pyangstrom.visualization.recording import animate_recording
from pyangstrom.visualization.region import (
    plot_geometry,
    plot_spatiotemporal_heat_map,
    plot_isotherms,
    plot_groups,
)
from pyangstrom.visualization.signal import (
    plot_amplitude_ratios,
    plot_phase_differences,
)


logger = logging.getLogger('pipeline')

def fully_extract_result(func, data, *args):
    if isinstance(data, list):
        return [fully_extract_result(func, d, *args) for d in data]
    return func(data, *args)

def visualize_recording(df_recording):
    return df_recording, animate_recording(df_recording)

def visualize_region(df_recording, region_result, region_information):
    if isinstance(region_result, list):
        _, figs = zip(*(
            visualize_region(df_recording, r, i) for r, i
            in zip(
                region_result,
                region_information
                if isinstance(region_information, list)
                else [{'geometry': g} for g in region_information['geometries']]
            )
        ))
        return region_result, list(figs)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    if 'geometry' in region_information:
        plot_geometry(ax1, df_recording, region_information['geometry'])
    else:
        plot_geometry(ax1, df_recording, region_information['geometries'])
    plot_spatiotemporal_heat_map(ax2, region_result)
    plot_isotherms(ax3, region_result)
    plot_groups(ax4, region_result)
    return region_result, fig

def visualize_signal(df_recording, signal_result, region_information):
    if isinstance(signal_result, list):
        _, figs = zip(*(
            visualize_signal(df_recording, s, i) for s, i
            in zip(
                signal_result,
                region_information
                if isinstance(region_information, list)
                else [{'geometry': g} for g in region_information['geometries']]
            )
        ))
        return signal_result, list(figs)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    if 'geometry' in region_information:
        plot_geometry(ax1, df_recording, region_information['geometry'])
    else:
        plot_geometry(ax1, df_recording, region_information['geometries'])
    plot_isotherms(ax2, signal_result.processed_region)
    plot_amplitude_ratios(ax3, signal_result)
    plot_phase_differences(ax4, signal_result)
    return signal_result, fig

def visualize_fit(
        df_recording,
        signal_result,
        fitting_result,
        region_information,
):
    if isinstance(fitting_result, list):
        _, figs = zip(*(
            visualize_fit(df_recording, s, f, i) for s, f, i
            in zip(
                signal_result,
                fitting_result,
                region_information
                if isinstance(region_information, list)
                else [{'geometry': g} for g in region_information['geometries']]
            )
        ))
        return fitting_result, list(figs)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    if 'geometry' in region_information:
        plot_geometry(ax1, df_recording, region_information['geometry'])
    else:
        plot_geometry(ax1, df_recording, region_information['geometries'])
    plot_isotherms(ax2, signal_result.processed_region)
    plot_amplitude_ratios(ax3, signal_result, fitting_result)
    plot_phase_differences(ax4, signal_result, fitting_result)
    # ax1: geometry
    # ax2: isotherms (maybe observed + theoretical)
    # ax3: amp ratio / displacement (observed + theoretical)
    # ax4: phase diff / displacement (observed + theoretical)
    # axN: fitting method-specific plots
    return fitting_result, fig

def analyze_recording(
        recording_path: str | Path,
        config: str | Path | Config,
        *,
        debug: bool = False,
        return_visualization: bool = False,
        memory_cache: Optional[Cache] = None,
        recording_cache_path: Optional[str | Path] = None,
):
    """Compute the thermal conductivity of the recorded sample.

    Parameters
    ----------
    recording_path
        Path to the directory holding CSV IR camera data. Does not have to be a
        real path if there is a cache file with a matching name in
        recording_cache_path.
    config
        Path to a JSON config file or the deserialized config object itself.

    Other Parameters
    ----------------
    memory_cache
        Holds intermediate results in memory, allowing for faster reruns.
    recording_cache_path
        Path to a directory in which cached IR camera data will be saved.
    """
    # TODO: debug, return_visualization
    logger.info("Loading recording")
    if recording_cache_path:
        recording_path = PurePath(recording_path)
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
            recording_path = Path(recording_path)
            df_recording = load_recording_csv(recording_path)
            save_recording_cache(
                df_recording,
                recording_cache_path,
                recording_path.stem,
            )
            logger.debug(f"Saved to cache: {df_recording[:1]=}")
    else:
        recording_path = Path(recording_path)
        df_recording = load_recording_csv(recording_path)
        logger.debug(f"Loaded csv: {df_recording[:1]=}")
    if not isinstance(config, dict):
        logger.info("Loading config")
        config = load_config(Path(config))
        logger.debug(f"{config=}")
    if 'region_information' not in config:
        if return_visualization:
            return visualize_recording(df_recording)
        else:
            return df_recording
    if 'experimental_setup' not in config:
        warnings.warn("Field experimental_setup required to extract regions")
        if return_visualization:
            return visualize_recording(df_recording)
        else:
            return df_recording
    logger.info("Extracting region(s)")
    try:
        region_result = fully_extract_region(
            df_recording,
            config['region_information'],
            config['experimental_setup'],
        )
    except Exception as e:
        if debug:
            raise
        warnings.warn(repr(e))
        if return_visualization:
            return visualize_recording(df_recording)
        else:
            return df_recording
    if 'signal_processor' not in config:
        if return_visualization:
            return visualize_region(
                df_recording,
                region_result,
                config['region_information'],
            )
        else:
            return region_result
    if isinstance(region_result, list):
        if isinstance(region_result[0], list):
            debug_region = region_result[0][0]
        else:
            debug_region = region_result[0]
    else:
        debug_region = region_result
    logger.debug(f"{debug_region.timestamps[:1]=}")
    logger.debug(f"{debug_region.temperatures_kelvin.flatten()[:1]=}")
    logger.debug(f"{debug_region.margins.seconds_elapsed[:1]=}")
    logger.debug(f"{debug_region.margins.displacements_meters.flatten()[:1]=}")
    logger.info("Signal processing")
    try:
        signal_result = fully_extract_result(
            signal_process_region,
            region_result,
            config['signal_processor'],
            config['experimental_setup']
        )
    except Exception as e:
        if debug:
            raise
        warnings.warn(repr(e))
        if return_visualization:
            return visualize_region(
                df_recording,
                region_result,
                config['region_information'],
            )
        else:
            return region_result
    if 'solver' not in config or 'fitter' not in config:
        if return_visualization:
            return visualize_signal(
                df_recording,
                signal_result,
                config['region_information'],
            )
        else:
            return signal_result
    if isinstance(signal_result, list):
        if isinstance(signal_result[0], list):
            debug_signal = signal_result[0][0]
        else:
            debug_signal = signal_result[0]
    else:
        debug_signal = signal_result
    for name, prop in debug_signal.signal_properties._asdict().items():
        logger.debug(f"{name}={prop[:3]}")
    logger.info("Fitting")
    try:
        fitting_result = fully_extract_result(
            autofit,
            signal_result,
            config['solver'],
            config['fitter'],
            config['experimental_setup'],
        )
    except Exception as e:
        if debug:
            raise
        warnings.warn(repr(e))
        if return_visualization:
            return visualize_signal(
                df_recording,
                signal_result,
                config['region_information'],
            )
        else:
            return signal_result
    if return_visualization:
        return visualize_fit(
            df_recording,
            signal_result,
            fitting_result,
            config['region_information'],
        )
    else:
        return fitting_result
