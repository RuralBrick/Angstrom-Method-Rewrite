from scipy.optimize import least_squares

from pyangstrom.fit import ResidualsCallable, Unknowns, FittingResult


def fitter(
        residuals: ResidualsCallable,
        unknowns_guesses: Unknowns,
        least_squares_kwargs,
) -> FittingResult:
    result = least_squares(
        residuals,
        unknowns_guesses,
        **least_squares_kwargs,
    )
    return FittingResult(result.x)
