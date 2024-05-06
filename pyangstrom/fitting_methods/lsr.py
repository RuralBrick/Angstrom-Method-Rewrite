import logging

from scipy.optimize import least_squares

from pyangstrom.fit import ResidualsCalculator, Unknowns, FittingResult


logger = logging.getLogger('fit')

def fit(
        calc_residuals: ResidualsCalculator,
        unknowns_guesses: Unknowns,
        **least_squares_kwargs,
) -> FittingResult:
    level = logger.getEffectiveLevel()
    if level <= logging.DEBUG:
        verbosity = 2
    elif level <= logging.INFO:
        verbosity = 1
    else:
        verbosity = 0
    result = least_squares(
        calc_residuals,
        unknowns_guesses,
        verbose=verbosity,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
