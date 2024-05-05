import logging

import numpy as np
from scipy.optimize import minimize

from pyangstrom.fit import ResidualsCallable, Unknowns, FittingResult


logger = logging.getLogger('fit')

def fitter(
        residuals: ResidualsCallable,
        unknowns_guesses: Unknowns,
        minimize_kwargs,
) -> FittingResult:
    result = minimize(
        lambda unknowns: np.sum(np.square(residuals(unknowns))),
        unknowns_guesses,
        method='Nelder-Mead',
        options=dict(disp=(logger.getEffectiveLevel() <= logging.DEBUG)),
        **minimize_kwargs,
    )
    return FittingResult(result.x)
