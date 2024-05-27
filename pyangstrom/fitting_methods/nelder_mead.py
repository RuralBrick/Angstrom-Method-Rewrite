import logging

import numpy as np
from scipy.optimize import minimize

from pyangstrom.fit import (
    Unknowns,
    TheoreticalCalculator,
    SignalProperties,
    FittingResult
)


logger = logging.getLogger('fit')

def fit(
        unknowns_guesses: Unknowns,
        unknowns_to_props: TheoreticalCalculator,
        signal_properties: SignalProperties,
        **minimize_kwargs,
) -> FittingResult:

    def calc_error(unknowns):
        answers = unknowns_to_props(unknowns)
        all_residuals = [
            p - np.expand_dims(a, tuple(range(1, len(p.shape))))
            for p, a in zip(signal_properties, answers)
        ]
        residuals = np.stack(all_residuals).flatten()
        error = np.sum(np.square(residuals))
        return error

    result = minimize(
        calc_error,
        unknowns_guesses,
        method='Nelder-Mead',
        options=dict(disp=(logger.getEffectiveLevel() <= logging.DEBUG)),
        **minimize_kwargs,
    )
    return FittingResult(result.x)
