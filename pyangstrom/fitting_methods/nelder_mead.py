import logging
import abc

import numpy as np
from scipy.optimize import minimize

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FittingResult,
)
from pyangstrom.signal import SignalProperties


logger = logging.getLogger('fit')

class NelderMeadEquations(EquationPackage):
    @abc.abstractmethod
    def unknowns_to_vector(self, unknowns: Unknowns) -> np.ndarray: ...

    @abc.abstractmethod
    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties: ...

def fit(
        unknowns_guesses: Unknowns,
        solution: NelderMeadEquations,
        observed_properties: SignalProperties,
        **minimize_kwargs,
) -> FittingResult:

    def calc_error(unknowns_vector):
        theoretical_properties = solution.vector_solve(unknowns_vector)
        all_residuals = [
            p - np.expand_dims(a, tuple(range(1, len(p.shape))))
            for p, a in zip(observed_properties, theoretical_properties)
        ]
        residuals = np.stack(all_residuals).flatten()
        error = np.sum(np.square(residuals))
        return error

    result = minimize(
        calc_error,
        solution.unknowns_to_vector(unknowns_guesses),
        method='Nelder-Mead',
        options=dict(disp=(logger.getEffectiveLevel() <= logging.DEBUG)),
        **minimize_kwargs,
    )
    return FittingResult(result.x)
