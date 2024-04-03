from typing import NamedTuple

import numpy as np

from pyangstrom.transform import Region
from pyangstrom.config import ExperimentalSetup


class SignalProperties(NamedTuple):
    amp_ratio: np.ndarray
    phase_diff: np.ndarray

def fft_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        tol=2,
) -> SignalProperties:

    fundamental_freq = 1.0 / region.margins[0].total_seconds()
    target_harmonic = int(setup['heating_frequency'] / fundamental_freq)
    window_start = max(target_harmonic - tol, 0)
    window_end = min(target_harmonic + tol, region.time.size)

    freq = np.fft.fft(region.temps, axis=0)[window_start:window_end]

    amps = np.abs(freq)
    target_idx = (amps.reshape(freq.shape[0], -1)
                      .sum(axis=1)
                      .argmax())
    amps = amps[target_idx]
    amp_ratio = amps / amps[0]

    phases = np.angle(freq[target_idx])
    phase_diff = phases[0] - phases

    return SignalProperties(amp_ratio, phase_diff)
