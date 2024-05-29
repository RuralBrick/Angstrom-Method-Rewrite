import logging
import abc

import numpy as np
from scipy.optimize import least_squares

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FittingResult,
)


logger = logging.getLogger('fit')

class LsrEquations(EquationPackage):
    @abc.abstractmethod
    def unknowns_to_vector(self, unknowns: Unknowns) -> np.ndarray: ...

    @abc.abstractmethod
    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties: ...

def fitting_function(signal_properties: SignalProperties) -> np.ndarray:
    linear_data = (np.log(1/signal_properties.amplitude_ratios)
                   * signal_properties.phase_differences)
    return linear_data

def fit(
        unknowns_guesses: Unknowns,
        solution: LsrEquations,
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

    observed = fitting_function(observed_properties)
    def calc_residuals(unknowns_vector):
        theoretical_properties = solution.vector_solve(unknowns_vector)
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
        solution.unknowns_to_vector(unknowns_guesses),
        verbose=verbosity,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
