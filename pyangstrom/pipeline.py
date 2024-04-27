from typing import Optional
from pathlib import Path

from pyangstrom.config import Config
from pyangstrom.caching import Cache
from pyangstrom.io import (
    recording_cache_exists,
    load_recording_csv,
    save_recording_cache,
    load_recording_cache,
)
from pyangstrom.transform import extract_cartesian_region, extract_polar_region
from pyangstrom.signal import fft_signal_processing
from pyangstrom.fit import fit
import pyangstrom.sample_solutions.lopez_baeza_short as lopez_baeza_short
import pyangstrom.sample_solutions.hu_high_temp as hu_high_temp
import pyangstrom.fitting_methods.lsr as lsr
import pyangstrom.fitting_methods.mcmc as mcmc


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
    recording_path: Path to the directory holding CSV IR camera data
    config: All metadata needed to compute thermal conductivity

    Keyword Arguments
    -----------------
    memory_cache: Holds intermediate results in memory, allowing for faster
        reruns
    recording_cache_path: Path to a directory in which cached IR camera data
        will be saved
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
    ...
