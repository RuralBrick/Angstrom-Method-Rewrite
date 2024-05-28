from typing import Callable
import random

import numpy as np

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FittingResult,
)


UnknownsProposer = Callable[[Unknowns], Unknowns]
LogPosteriorCalculator = Callable[[Unknowns, SignalProperties], float]

def fit(
        unknowns_guesses: Unknowns,
        solution: EquationPackage,
        observed_properties: SignalProperties,
        target_num_accepted_samples,
) -> FittingResult:
    propose: UnknownsProposer = solution.propose
    calc_log_posterior: LogPosteriorCalculator = solution.log_posterior

    accepted_samples = []

    current_unknowns = unknowns_guesses
    current_log_posterior = calc_log_posterior(
        current_unknowns,
        observed_properties,
    )
    while len(accepted_samples) >= target_num_accepted_samples:
        proposed_unknowns = propose(current_unknowns)
        new_log_posterior = calc_log_posterior(
            proposed_unknowns,
            observed_properties,
        )
        acceptance_ratio = np.exp(new_log_posterior - current_log_posterior)
        if random.uniform(0, 1) <= acceptance_ratio:
            accepted_samples.append(proposed_unknowns)
            current_unknowns = proposed_unknowns
            current_log_posterior = new_log_posterior

    return FittingResult(accepted_samples[-1])
