from typing import Optional

from matplotlib.axes import Axes

from pyangstrom.signal import SignalResult
from pyangstrom.fit import FittingResult


MARGINS_MULTIPLIERS = {
    'm': 1.,
    'meters': 1.,
    'mm': 1e3,
    'millimeters': 1e3,
}

def plot_amplitude_ratios(
        ax: Axes,
        signal_result: SignalResult,
        fitting_result: Optional[FittingResult] = None,
        displacements_unit: str = 'mm',
) -> Axes:
    x = (MARGINS_MULTIPLIERS[displacements_unit]
         * signal_result.margins.displacements_meters)
    ax.plot(
        x,
        signal_result.signal_properties.amplitude_ratios,
        'o',
        label="Observed",
    )
    if fitting_result:
        ax.plot(
            x,
            fitting_result.theoretical_properties.amplitude_ratios,
            label="Theoretical",
        )
        ax.legend()
    ax.set_title("Amplitude Ratios")
    ax.set_xlabel(f"Displacement from heat source ({displacements_unit})")
    ax.set_ylabel("Ratio")
    return ax

def plot_phase_differences(
        ax: Axes,
        signal_result: SignalResult,
        fitting_result: Optional[FittingResult] = None,
        displacements_unit: str = 'mm',
) -> Axes:
    x = (MARGINS_MULTIPLIERS[displacements_unit]
         * signal_result.margins.displacements_meters)
    ax.plot(
        x,
        signal_result.signal_properties.phase_differences,
        'o',
        label="Observed",
    )
    if fitting_result:
        ax.plot(
            x,
            fitting_result.theoretical_properties.phase_differences,
            label="Theoretical",
        )
        ax.legend()
    ax.set_title("Phase Differences")
    ax.set_xlabel(f"Displacement from heat source ({displacements_unit})")
    ax.set_ylabel("Difference")
    return ax
