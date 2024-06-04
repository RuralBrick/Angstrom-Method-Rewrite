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
    plot_isotherms,
    plot_groups,
)


logger = logging.getLogger('pipeline')

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

    Arguments
    ---------
    recording_path
        Path to the directory holding CSV IR camera data. Does not have to be a
        real path if there is a cache file with a matching name in
        recording_cache_path.
    config
        Path to a YAML config file or the deserialized config object itself.

    Keyword Arguments
    -----------------
    verbose
        If True, print status messages while processing data.
    memory_cache
        Holds intermediate results in memory, allowing for faster reruns.
    recording_cache_path
        Path to a directory in which cached IR camera data will be saved.
    """
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
            return df_recording, animate_recording(df_recording)
        else:
            return df_recording
    if 'experimental_setup' not in config:
        warnings.warn("Field experimental_setup required to extract regions")
        if return_visualization:
            return df_recording, animate_recording(df_recording)
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
            return df_recording, animate_recording(df_recording)
        else:
            return df_recording
    if 'signal_processor' not in config:
        if return_visualization:
            if isinstance(region_result, list):
                figs = []
                for region, info in zip(region_result, config['region_information']):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                    if 'geometry' in info:
                        plot_geometry(ax1, df_recording, info['geometry'])
                    else:
                        plot_geometry(ax1, df_recording, info['geometries'])
                    plot_isotherms(ax2, region)
                    plot_groups(ax3, region)
                    figs.append(fig)
                return region_result, figs
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                if 'geometry' in config['region_information']:
                    plot_geometry(ax1, df_recording, info['geometry'])
                else:
                    plot_geometry(ax1, df_recording, info['geometries'])
                plot_isotherms(ax2, region)
                plot_groups(ax3, region)
                return region_result, fig
        else:
            return region_result
    if isinstance(region_result, list):
        debug_region = region_result[0]
    else:
        debug_region = region_result
    logger.debug(f"{debug_region.timestamps[:1]=}")
    logger.debug(f"{debug_region.temperatures_kelvin[:1, :1]=}")
    logger.debug(f"{debug_region.margins.seconds_elapsed[:1]=}")
    logger.debug(f"{debug_region.margins.displacements_meters[:1, :1]=}")
    logger.info("Signal processing")
    try:
        if isinstance(region_result, list):
            signal_result = [
                signal_process_region(
                    r,
                    config['signal_processor'],
                    config['experimental_setup'],
                ) for r in region_result
            ]
        else:
            signal_result = signal_process_region(
                region_result,
                config['signal_processor'],
                config['experimental_setup'],
            )
    except Exception as e:
        if debug:
            raise
        warnings.warn(repr(e))
        if return_visualization:
            if isinstance(region_result, list):
                figs = []
                for region, info in zip(region_result, config['region_information']):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                    if 'geometry' in info:
                        plot_geometry(ax1, df_recording, info['geometry'])
                    else:
                        plot_geometry(ax1, df_recording, info['geometries'])
                    plot_isotherms(ax2, region)
                    plot_groups(ax3, region)
                    figs.append(fig)
                return region_result, figs
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                if 'geometry' in config['region_information']:
                    plot_geometry(ax1, df_recording, info['geometry'])
                else:
                    plot_geometry(ax1, df_recording, info['geometries'])
                plot_isotherms(ax2, region)
                plot_groups(ax3, region)
                return region_result, fig
        else:
            return region_result
    if 'solver' not in config or 'fitter' not in config:
        return signal_result
    if isinstance(signal_result, list):
        debug_signal = signal_result[0]
    else:
        debug_signal = signal_result
    for name, prop in debug_signal.signal_properties._asdict().items():
        logger.debug(f"{name}={prop[:3]}")
    logger.info("Fitting")
    try:
        if isinstance(signal_result, list):
            fitting_result = [
                autofit(
                    r,
                    config['solver'],
                    config['fitter'],
                    config['experimental_setup'],
                ) for r in signal_result
            ]
        else:
            fitting_result = autofit(
                signal_result,
                config['solver'],
                config['fitter'],
                config['experimental_setup'],
            )
    except Exception as e:
        if debug:
            raise
        warnings.warn(repr(e))
        return signal_result
    return fitting_result
