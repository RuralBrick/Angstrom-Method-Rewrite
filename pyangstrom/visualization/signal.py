from typing import Optional

from matplotlib.axes import Axes

from pyangstrom.signal import SignalResult
from pyangstrom.fit import FittingResult


def plot_amplitude_ratios(
        ax: Axes,
        signal_result: SignalResult,
        fitting_result: Optional[FittingResult] = None,
) -> Axes:
    ax.plot(
        signal_result.margins.displacements_meters,
        signal_result.signal_properties.amplitude_ratios,
        'o',
        label="Observed",
    )
    if fitting_result:
        ax.plot(
            signal_result.margins.displacements_meters,
            fitting_result.theoretical_properties.amplitude_ratios,
            label="Theoretical",
        )
        ax.legend()
    ax.set_title("Amplitude Ratios")
    ax.set_xlabel("Displacement from heat source (meters)")
    ax.set_ylabel("Ratio")
    return ax

def plot_phase_differences(
        ax: Axes,
        signal_result: SignalResult,
        fitting_result: Optional[FittingResult] = None,
) -> Axes:
    ax.plot(
        signal_result.margins.displacements_meters,
        signal_result.signal_properties.phase_differences,
        'o',
        label="Observed",
    )
    if fitting_result:
        ax.plot(
            signal_result.margins.displacements_meters,
            fitting_result.theoretical_properties.phase_differences,
            label="Theoretical",
        )
        ax.legend()
    ax.set_title("Phase Differences")
    ax.set_xlabel("Displacement from heat source (meters)")
    ax.set_ylabel("Difference")
    return ax
