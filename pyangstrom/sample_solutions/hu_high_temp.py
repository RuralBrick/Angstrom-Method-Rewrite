from typing import TypedDict

import numpy as np

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins
from pyangstrom.signal import SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations
from pyangstrom.fitting_methods.metropolis_hastings import MetropolisHastingsEquations

class HuHighTempUnknowns(TypedDict):
    pass


class Solution(NelderMeadEquations, LsrEquations, MetropolisHastingsEquations):
    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            **kwargs
    ) -> None:
        self.margins = margins
        self.setup = setup

    def unknowns_to_vector(self, unknowns: HuHighTempUnknowns) -> np.ndarray:
        return super().unknowns_to_vector(unknowns)

    def vector_to_unknowns(self, vector: np.ndarray) -> HuHighTempUnknowns:
        return super().vector_to_unknowns(vector)

    def solve(self, unknowns: HuHighTempUnknowns) -> SignalProperties:
        return super().solve(unknowns)

    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties:
        return super().vector_solve(unknowns_vector)

    def propose(self, unknowns: dict) -> dict:
        return super().propose(unknowns)

    def log_posterior(self, unknowns: HuHighTempUnknowns, observed_properties: SignalProperties) -> float:
        return super().log_posterior(unknowns, observed_properties)
