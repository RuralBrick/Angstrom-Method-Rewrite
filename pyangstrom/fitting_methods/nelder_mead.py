import logging
from typing import Callable

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

UnknownsVectorizer = Callable[[Unknowns], np.ndarray]
VectorTheoreticalCalculator = Callable[[Unknowns], SignalProperties]

def fit(
        unknowns_guesses: Unknowns,
        solution: EquationPackage,
        observed_properties: SignalProperties,
        **minimize_kwargs,
) -> FittingResult:
    vectorize_unknowns: UnknownsVectorizer = solution.unknowns_to_vector
    vector_solve: VectorTheoreticalCalculator = solution.vector_solve

    def calc_error(unknowns_vector):
        theoretical_properties = vector_solve(unknowns_vector)
        all_residuals = [
            p - np.expand_dims(a, tuple(range(1, len(p.shape))))
            for p, a in zip(observed_properties, theoretical_properties)
        ]
        residuals = np.stack(all_residuals).flatten()
        error = np.sum(np.square(residuals))
        return error

    result = minimize(
        calc_error,
        vectorize_unknowns(unknowns_guesses),
        method='Nelder-Mead',
        options=dict(disp=(logger.getEffectiveLevel() <= logging.DEBUG)),
        **minimize_kwargs,
    )
    return FittingResult(result.x)
