import logging
from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import signal
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from pyangstrom.transform import Region, Margins
from pyangstrom.exp_setup import ExperimentalSetup


logger = logging.getLogger('signal')

class SignalProperties(NamedTuple):
    # TODO: Docstring
    amplitude_ratios: np.ndarray
    phase_differences: np.ndarray

@dataclass
class SignalResult:
    # TODO: Docstring
    signal_properties: SignalProperties
    processed_region: Region
    margins: Margins

class SignalProcessor(Protocol):
    # TODO: Docstring

    def __call__(
            self,
            region: Region,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> SignalProperties: ...

class SignalProcessorInformation(TypedDict, total=False):
    # TODO: Docstring
    #       * explain apply_filter
    #       * explain what's optional/mutually exclusive
    #       * list valid names
    #       * link to wiki page
    name: str
    processor: SignalProcessor
    parameters: dict
    apply_filter: bool

class SineParameters(TypedDict):
    # TODO: Docstring (internal)
    amplitude: float
    phase: float
    bias: float
    frequency: float

def filter_signal(
        region: Region,
        setup: ExperimentalSetup,
        cutoff: float = 0.5,
        order: int = 5,
) -> Region:
    # TODO: Docstring

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

def calc_sine_temps(
        params: SineParameters,
        seconds_elapsed: np.ndarray,
) -> np.ndarray:
    A = params['amplitude']
    p = params['phase']
    b = params['bias']
    f = params['frequency']
    t = seconds_elapsed

    return A * np.sin(2.0*np.pi*f*t + p) + b

def minimize_sine_residuals(
        node_temps: np.ndarray,
        params: Parameters,
        margins: Margins,
) -> MinimizerResult:

    def calc_sine_residuals(params: SineParameters):
        theoretical_temps = calc_sine_temps(params, margins.seconds_elapsed)
        residuals = node_temps - theoretical_temps
        return residuals

    result = minimize(calc_sine_residuals, params)
    return result

@np.vectorize
def extract_result_amplitudes(result: MinimizerResult) -> float:
    return result.params['amplitude']

@np.vectorize
def extract_result_phases(result: MinimizerResult) -> float:
    return result.params['phase']

def sine_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        initial_amplitude=1.0,
        initial_phase=0.1,
        initial_bias=298.0,
) -> SignalProperties:
    # TODO: Docstring (maybe)

    params = Parameters()
    params.add_many(
        ('amplitude', initial_amplitude, True, 0.0, None, None, None),
        ('phase', initial_phase, True, 0.0, 2.0*np.pi, None, None),
        ('bias', initial_bias, True, None, None, None, None),
        ('frequency', setup['heating_frequency_hertz'], False, None, None, None, None),
    )

    results = np.apply_along_axis(
        partial(minimize_sine_residuals, params=params, margins=region.margins),
        0,
        region.temperatures_kelvin,
    )

    amps = extract_result_amplitudes(results)
    amp_ratio = amps / amps[0]

    phases = extract_result_phases(results)
    phase_diff = phases - phases[0]

    return SignalProperties(amp_ratio, phase_diff)

def fft_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
        tol=2,
) -> SignalProperties:
    # TODO: Docstring (maybe)

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

def max_min_amp(node_temps: np.ndarray) -> float:
    maxes = node_temps[signal.argrelmax(node_temps)]
    mins = node_temps[signal.argrelmin(node_temps)]
    amp = maxes.mean() - mins.mean()
    return amp

def max_min_phase(
        node_temps: np.ndarray,
        region: Region,
        setup: ExperimentalSetup,
) -> float:
    idx_first_min = signal.argrelmin(node_temps)[0][0]
    phase = (2.0
             * np.pi
             * idx_first_min
             * setup['heating_frequency_hertz']
             * region.margins.seconds_elapsed.max()
             / len(node_temps))
    return phase

def max_min_signal_processing(
        region: Region,
        setup: ExperimentalSetup,
) -> SignalProperties:
    # TODO: Docstring (maybe)

    amps = np.apply_along_axis(max_min_amp, 0, region.temperatures_kelvin)
    amp_ratio = amps / amps[0]

    phases = np.apply_along_axis(
        partial(max_min_phase, region=region, setup=setup),
        0,
        region.temperatures_kelvin,
    )
    phase_diff = phases - phases[0]

    return SignalProperties(amp_ratio, phase_diff)

def extract_processor(
        information: SignalProcessorInformation
) -> SignalProcessor:
    """
    Raises
    ------
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
    Raises
    ------
    KeyError
        Field not found in information.
    ValueError
        Named signal processor not found.
    """
    # TODO: Improve docstring
    if 'apply_filter' in information and information['apply_filter']:
        region = filter_signal(region, setup)
    processor = extract_processor(information)
    params = information['parameters'] if 'parameters' in information else {}
    props = processor(region, setup, **params)
    result = SignalResult(props, region, region.margins)
    return result
