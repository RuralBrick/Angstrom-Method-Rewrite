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

    fundamental_freq = 1.0 / region.margins[-1]
    target_harmonic = int(setup['heating_frequency'] / fundamental_freq)
    window_start = max(target_harmonic - tol, 0)
    window_end = min(target_harmonic + tol, region.time.size)

    freq = np.fft.fft(region.temps, axis=-1)

    amps = np.abs(freq)
    amp_ratio = amps / amps[0]

    phases = np.angle(freq)
    phase_diff = np.mod(np.abs(phases - phases[0]), np.pi)
    phase_diff = np.where(
        phase_diff > np.pi / 2.0,
        np.pi - phase_diff,
        phase_diff,
    )

    return SignalProperties(amp_ratio, phase_diff)
