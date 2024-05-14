from pyangstrom.fit import ResidualsCalculator, Unknowns, FittingResult


def fit(
        calc_residuals: ResidualsCalculator, # FIXME: Can't use
        unknowns_guesses: Unknowns,
        # TODO: Need injected distribution sampler
) -> FittingResult:
    raise NotImplementedError()
