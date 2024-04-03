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
    target_idx = amps.argmax(axis=0, keepdims=True)
    amps = np.take_along_axis(amps, target_idx, axis=0).squeeze(axis=0)
    amp_ratio = amps / amps[0]

    freq = np.take_along_axis(freq, target_idx, axis=0).squeeze(axis=0)
    phases = np.angle(freq)
    phase_diff = np.mod(np.abs(phases - phases[0]), np.pi)
    phase_diff = np.where(
        phase_diff > np.pi / 2.0,
        np.pi - phase_diff,
        phase_diff,
    )

    return SignalProperties(amp_ratio, phase_diff)
