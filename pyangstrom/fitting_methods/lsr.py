import logging

from scipy.optimize import least_squares

from pyangstrom.fit import ResidualsCallable, Unknowns, FittingResult


logger = logging.getLogger('fit')

def fitter(
        residuals: ResidualsCallable,
        unknowns_guesses: Unknowns,
        least_squares_kwargs,
) -> FittingResult:
    level = logger.getEffectiveLevel()
    if level <= logging.DEBUG:
        verbosity = 2
    elif level <= logging.INFO:
        verbosity = 1
    else:
        verbosity = 0
    result = least_squares(
        residuals,
        unknowns_guesses,
        verbose=verbosity,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
