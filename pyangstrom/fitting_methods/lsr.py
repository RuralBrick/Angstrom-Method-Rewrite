import logging

import numpy as np
from scipy.optimize import least_squares

from pyangstrom.fit import (
    Unknowns,
    TheoreticalCalculator,
    SignalProperties,
    FittingResult
)


logger = logging.getLogger('fit')

def fitting_function(signal_properties: SignalProperties) -> np.ndarray:
    linear_data = (np.log(1/signal_properties.amplitude_ratios)
                   * signal_properties.phase_differences)
    return linear_data

def fit(
        unknowns_guesses: Unknowns,
        unknowns_to_props: TheoreticalCalculator,
        signal_properties: SignalProperties,
        **least_squares_kwargs,
) -> FittingResult:
    level = logger.getEffectiveLevel()
    if level <= logging.DEBUG:
        verbosity = 2
    elif level <= logging.INFO:
        verbosity = 1
    else:
        verbosity = 0

    observed = fitting_function(signal_properties)
    def calc_residuals(unknowns):
        answers = unknowns_to_props(unknowns)
        theoretical = fitting_function(answers)
        theoretical = np.expand_dims(
            theoretical,
            tuple(range(1, len(observed.shape))),
        )
        all_residuals = observed - theoretical
        residuals = all_residuals.flatten()
        return residuals

    result = least_squares(
        calc_residuals,
        unknowns_guesses,
        verbose=verbosity,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
