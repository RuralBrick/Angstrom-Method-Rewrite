from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass

import numpy as np
from scipy import signal

from pyangstrom.transform import Region, Margins
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
    margins: Margins

def filter_signal(
        region: Region,
        setup: ExperimentalSetup,
        cutoff: float = 0.5,
        order: int = 5,
) -> Region:
    cutoff_frequency = cutoff * setup['heating_frequency_hertz']
    sampling_frequency = region.temperatures_kelvin.shape[0] / region.margins[0]
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(
        order,
        normal_cutoff,
        btype='high',
        analog=False,
    )
    new_temps = signal.filtfilt(b, a, region.temperatures_kelvin, axis=0)
    new_region = Region(
        region.timestamps,
        new_temps,
        region.margins,
    )
    return new_region

def filter_signal(
        region: Region,
        setup: ExperimentalSetup,
        cutoff: float = 0.5,
        order: int = 5,
) -> Region:
    cutoff_frequency = cutoff * setup['heating_frequency_hertz']
    sampling_frequency = (region.temperatures_kelvin.shape[0]
                          / region.margins.seconds_elapsed.max())
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(
        order,
        normal_cutoff,
        btype='high',
        analog=False,
    )
    new_temps = signal.filtfilt(b, a, region.temperatures_kelvin, axis=0)
    new_region = Region(
        region.timestamps,
        new_temps,
        region.margins,
    )
    return new_region

def fft_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        tol=2,
) -> SignalProperties:
    fundamental_freq = 1.0 / region.margins.seconds_elapsed.max()
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
    if 'apply_filter' in information and information['apply_filter']:
        region = filter_signal(region, setup)
    match information['name']:
        case 'fft':
            processor = fft_signal_processing
        case _:
            raise ValueError(
                f"Signal processor {information['name']} not found."
            )
    params = information['parameters'] if 'parameters' in information else {}
    props = processor(region, setup, **params)
    result = SignalResult(props, region.margins)
    return result
