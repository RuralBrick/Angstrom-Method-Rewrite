import logging
from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass

import numpy as np
from scipy import signal
from lmfit import Parameters, minimize

from pyangstrom.transform import Region, Margins
from pyangstrom.exp_setup import ExperimentalSetup


logger = logging.getLogger('signal')

class SignalProperties(NamedTuple):
    amplitude_ratios: np.ndarray
    phase_differences: np.ndarray

@dataclass
class SignalResult:
    signal_properties: SignalProperties
    processed_region: Region
    margins: Margins

class SignalProcessor(Protocol):
    def __call__(
            self,
            region: Region,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> SignalProperties: ...

class SignalProcessorInformation(TypedDict, total=False):
    name: str
    processor: SignalProcessor
    parameters: dict
    apply_filter: bool

class SineParameters(TypedDict):
    amplitude: float
    phiase: float
    bias: float

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

def sine_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        initial_parameters: SineParameters=dict(
            amplitude=1.0,
            phase=0.1,
            bias=298.0,
        ),
) -> SignalProperties:

    params = Parameters()
    params.add_many(
        ('amplitude', initial_parameters['amplitude'], True, None, None, None, None),
        ('phase', initial_parameters['phiase'], True, None, None, None, None),
        ('bias', initial_parameters['bias'], True, None, None, None, None),
        ('frequency', setup['heating_frequency_hertz'], False, None, None, None, None),
    )

    raise NotImplementedError()

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

def max_min_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
) -> SignalProperties:

    idx_maxes = signal.argrelextrema(None, np.greater, axis=None)
    idx_mins = signal.argrelextrema(None, np.less, axis=None)

    raise NotImplementedError()

def extract_processor(
        information: SignalProcessorInformation
) -> SignalProcessor:
    """
    Exceptions
    ----------
    KeyError
        Field not found in information.
    ValueError
        Named signal processor not found.
    """
    if 'processor' in information:
        return information['processor']
    elif 'name' in information:
        match information['name']:
            case 'sin' | 'sine':
                return sine_signal_processing
            case 'fft':
                return fft_signal_processing
            case 'max_min':
                return max_min_signal_processing
            case _:
                raise ValueError(
                    f"Signal processor {information['name']} not found."
                )
    else:
        raise KeyError("Must have either name or processor in information.")

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
    processor = extract_processor(information)
    params = information['parameters'] if 'parameters' in information else {}
    props = processor(region, setup, **params)
    result = SignalResult(props, region, region.margins)
    return result
