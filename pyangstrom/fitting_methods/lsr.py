import logging
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FittingResult,
)


logger = logging.getLogger('fit')

UnknownsVectorizer = Callable[[Unknowns], np.ndarray]
VectorTheoreticalCalculator = Callable[[Unknowns], SignalProperties]

def fitting_function(signal_properties: SignalProperties) -> np.ndarray:
    linear_data = (np.log(1/signal_properties.amplitude_ratios)
                   * signal_properties.phase_differences)
    return linear_data

def fit(
        unknowns_guesses: Unknowns,
        solution: EquationPackage,
        observed_properties: SignalProperties,
        **least_squares_kwargs,
) -> FittingResult:
    level = logger.getEffectiveLevel()
    if level <= logging.DEBUG:
        verbosity = 2
    elif level <= logging.INFO:
        verbosity = 1
    else:
        verbosity = 0

    vectorize_unknowns: UnknownsVectorizer = solution.unknowns_to_vector
    vector_solve: VectorTheoreticalCalculator = solution.vector_solve

    observed = fitting_function(observed_properties)
    def calc_residuals(unknowns_vector):
        theoretical_properties = vector_solve(unknowns_vector)
        theoretical = fitting_function(theoretical_properties)
        theoretical = np.expand_dims(
            theoretical,
            tuple(range(1, len(observed.shape))),
        )
        all_residuals = observed - theoretical
        residuals = all_residuals.flatten()
        return residuals

    result = least_squares(
        calc_residuals,
        vectorize_unknowns(unknowns_guesses),
        verbose=verbosity,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
