import logging
import abc
import random
from math import floor
from pprint import pformat

import numpy as np
import matplotlib.pyplot as plt

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FitterOutput,
)


logger = logging.getLogger('fit')

class MetropolisHastingsEquations(EquationPackage):
    @abc.abstractmethod
    def propose(self, unknowns: Unknowns) -> Unknowns:
        """Proposes a new set of values for the unknowns based on the previous
        set.
        """
        ...

    @abc.abstractmethod
    def log_posterior(
            self,
            unknowns: Unknowns,
            observed_properties: SignalProperties,
    ) -> float:
        """Calculates the probability of the values for the unknowns, given the
        experimental signal properties.
        """
        ...

def fit(
        unknowns_guesses: Unknowns,
        solution: MetropolisHastingsEquations,
        observed_properties: SignalProperties,
        target_num_accepted_samples: int,
        percent_burn_in: float = 0.2,
) -> FitterOutput:
    accepted_samples = []

    current_unknowns = unknowns_guesses
    current_log_posterior = solution.log_posterior(
        current_unknowns,
        observed_properties,
    )
    i = 0
    j = 0
    while len(accepted_samples) < target_num_accepted_samples:
        proposed_unknowns = solution.propose(current_unknowns)
        new_log_posterior = solution.log_posterior(
            proposed_unknowns,
            observed_properties,
        )
        acceptance_ratio = new_log_posterior - current_log_posterior
        if np.log(random.uniform(0, 1)) <= acceptance_ratio:
            if j >= 500:
                msg = f"Iteration {i}, {len(accepted_samples)} accepted\n"
                msg += f"{current_log_posterior=}, {new_log_posterior=}, {acceptance_ratio=}\n"
                msg += pformat(current_unknowns)
                logger.info(msg)
                j %= 500
            accepted_samples.append(proposed_unknowns)
            current_unknowns = proposed_unknowns
            current_log_posterior = new_log_posterior
        i += 1
        j += 1

    fig, axes = plt.subplots(
        1,
        len(
            [
                10**v for v in unknowns_guesses.values()
                if isinstance(v, float) or isinstance(v, int)
            ]
        )
    )

    for ax, k in zip(axes, unknowns_guesses.keys()):
        ax.plot([s[k] for s in accepted_samples])
        ax.set_title(k)

    num_burn = floor(len(accepted_samples) * percent_burn_in)
    metadata = {
        'accepted_samples': accepted_samples[num_burn:],
        'trace_plots': fig,
    }

    return FitterOutput(accepted_samples[-1], metadata)
