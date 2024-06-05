import logging
import abc
import random

import numpy as np

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FitterOutput,
)


logger = logging.getLogger('fit')

class MetropolisHastingsEquations(EquationPackage):
    @abc.abstractmethod
    def propose(self, unknowns: Unknowns) -> Unknowns:...

    @abc.abstractmethod
    def log_posterior(
            self,
            unknowns: Unknowns,
            observed_properties: SignalProperties,
    ) -> float: ...

def fit(
        unknowns_guesses: Unknowns,
        solution: MetropolisHastingsEquations,
        observed_properties: SignalProperties,
        target_num_accepted_samples,
) -> FitterOutput:
    accepted_samples = []

    current_unknowns = unknowns_guesses
    current_log_posterior = solution.log_posterior(
        current_unknowns,
        observed_properties,
    )
    while len(accepted_samples) >= target_num_accepted_samples:
        proposed_unknowns = solution.propose(current_unknowns)
        new_log_posterior = solution.log_posterior(
            proposed_unknowns,
            observed_properties,
        )
        acceptance_ratio = np.exp(new_log_posterior - current_log_posterior)
        if random.uniform(0, 1) <= acceptance_ratio:
            accepted_samples.append(proposed_unknowns)
            current_unknowns = proposed_unknowns
            current_log_posterior = new_log_posterior

    return FitterOutput(accepted_samples[-1])
