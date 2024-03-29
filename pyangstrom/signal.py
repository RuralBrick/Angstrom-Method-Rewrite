from typing import NamedTuple

import numpy as np

from pyangstrom.transform import Region


class SignalProperties(NamedTuple):
    amp_ratio: np.ndarray
    phase_diff: np.ndarray

def fft_signal_processing(region: Region) -> SignalProperties:
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
