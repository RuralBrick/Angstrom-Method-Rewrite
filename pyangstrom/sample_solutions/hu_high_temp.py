from typing import TypedDict

import numpy as np
from sklearn.linear_model import LinearRegression
import chaospy as cp

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins
from pyangstrom.signal import SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations
from pyangstrom.fitting_methods.metropolis_hastings import MetropolisHastingsEquations


class ViewFactor(TypedDict):
    T_LBW_C: float

class SampleInformation(TypedDict):
    R # sample radius
    t_z # sample thickness
    rho
    cp_const # Cp = cp_const + cp_c1*T + cp_c2/T + cp_c3/T**2, T unit in K
    cp_c1
    cp_c2
    cp_c3
    alpha_r_A: float
    alpha_r_B: float
    T_initial # unit in K
    emissivity_front # assumed to be constant
    emissivity_back # assumed to be constant
    absorptivity_solar
    absorptivity_front # assumed to be constant
    absorptivity_back # assumed to be constant
    rec_name

class NumericalSimulationSettings(TypedDict):
    Nr_node # number of discretization along radial direction
    N_cycle
    N_stable_cycle_output
    simulated_amp_phase_extraction_method
    simulated_num_data_per_cycle

class VacuumChamberSettings(TypedDict):
    R0_node
    focal_shift

class SolarSimulatorSettings(TypedDict):
    f_heating # periodic heating frequency

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
