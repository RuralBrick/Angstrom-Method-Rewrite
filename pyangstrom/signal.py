from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass

import numpy as np

from pyangstrom.transform import (
    Region,
    region_normalized_timestamps,
    region_to_displacement,
)
from pyangstrom.exp_setup import ExperimentalSetup


class SignalProcessorInformation(TypedDict, total=False):
    name: str
    apply_filter: bool
    parameters: dict

class SignalProperties(NamedTuple):
    amplitude_ratios: np.ndarray
    phase_differences: np.ndarray

class SignalProcessor(Protocol):
    def __call__(
            self,
            region: Region,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> SignalProperties: ...

@dataclass
class SignalResult:
    signal_properties: SignalProperties
    timestamps: np.ndarray
    displacements_meters: np.ndarray

def fft_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        tol=2,
) -> SignalProperties:
    fundamental_freq = 1.0 / region.margins[0].total_seconds()
    target_harmonic = int(setup['heating_frequency_hertz'] / fundamental_freq)
    window_start = max(target_harmonic - tol, 0)
    window_end = min(target_harmonic + tol, region.timestamps.size)

    freq = np.fft.fft(
        region.temperatures_kelvin,
        axis=0,
    )[window_start:window_end]

    amps = np.abs(freq)
    target_idx = (amps.reshape(freq.shape[0], -1)
                      .sum(axis=1)
                      .argmax())
    amps = amps[target_idx]
    amp_ratio = amps / amps[0]

    phases = np.angle(freq[target_idx])
    phase_diff = phases[0] - phases

    return SignalProperties(amp_ratio, phase_diff)

SIGNAL_PROCESSORS: dict[str, SignalProcessor] = {
    'fft': fft_signal_processing,
}

def signal_process_region(
        region: Region,
        information: SignalProcessorInformation,
        setup: ExperimentalSetup,
) -> SignalResult:
    """
    Exceptions
    ----------
    KeyError
        Field not found in information.
    ValueError
        Named signal processor not found.
    """
    # TODO: apply_filter
    match information['name']:
        case 'fft':
            processor = fft_signal_processing
        case _:
            raise ValueError(
                f"Signal processor {information['name']} not found."
            )
    params = information['parameters'] if 'parameters' in information else {}
    props = processor(region, setup, **params)
    result = SignalResult(
        props,
        region_normalized_timestamps(region),
        region_to_displacement(region),
    )
    return result
