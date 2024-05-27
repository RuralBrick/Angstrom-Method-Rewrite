from pyangstrom.fit import (
    Unknowns,
    TheoreticalCalculator,
    SignalProperties,
    FittingResult
)


def fit(
        unknowns_guesses: Unknowns,
        unknowns_to_props: TheoreticalCalculator,
        signal_properties: SignalProperties,
        # TODO: Need injected distribution sampler
) -> FittingResult:
    raise NotImplementedError()
