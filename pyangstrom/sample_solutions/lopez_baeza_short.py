from typing import TypedDict

import numpy as np

from pyangstrom.helpers import calc_thermal_conductivity
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins
from pyangstrom.signal import SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations
from pyangstrom.fitting_methods.metropolis_hastings import MetropolisHastingsEquations


class LopezBaezaShortUnknowns(TypedDict):
    # TODO: Docstring
    thermal_diffusivity_m2__s: float
    convective_heat_transfer_coefficient_W__m2_K: float

class LogLopezBaezaShortUnknowns(TypedDict):
    # TODO: Docstring
    thermal_diffusivity_log10_m2__s: float
    convective_heat_transfer_coefficient_log10_W__m2_K: float

class Solution(
    NelderMeadEquations,
    LsrEquations,
    MetropolisHastingsEquations,
):
    # TODO: Docstring

    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            r_meters: float,
            length_meters: float,
    ) -> None:
        mp = setup['material_properties']
        self.specific_heat_capacity_J__kg_K = mp['specific_heat_capacity_J__kg_K']
        self.density_kg__m3 = mp['density_kg__m3']
        self.displacements_meters = margins.displacements_meters
        self.angular_frequency_hertz = 2*np.pi*setup['heating_frequency_hertz']
        self.r_meters = r_meters
        self.length_meters = length_meters

    def unknowns_to_vector(
            self,
            unknowns: LopezBaezaShortUnknowns,
    ) -> np.ndarray:
        vector = np.array([
            unknowns['thermal_diffusivity_m2__s'],
            unknowns['convective_heat_transfer_coefficient_W__m2_K'],
        ])
        return vector

    def vector_to_unknowns(self, vector: np.ndarray) -> LopezBaezaShortUnknowns:
        unknowns: LopezBaezaShortUnknowns = {
            'thermal_diffusivity_m2__s': vector[0],
            'convective_heat_transfer_coefficient_W__m2_K': vector[1],
        }
        return unknowns

    def calc_wavenumber(
            self,
            thermal_diffusivity_m2__s,
            convective_heat_transfer_coefficient_W__m2_K,
            thermal_conductivity_W__m_K,
    ):
        w = self.angular_frequency_hertz
        D = thermal_diffusivity_m2__s
        h = convective_heat_transfer_coefficient_W__m2_K
        r = self.r_meters
        K = thermal_conductivity_W__m_K

        heat_conduction = w / (2.0*D)
        thermal_losses = h / (r*K)
        temp_var1 = np.sqrt(thermal_losses**2.0 + heat_conduction**2.0)
        temp_var2 = np.sqrt(-thermal_losses + temp_var1)
        temp_var3 = 1.0j*np.sqrt(thermal_losses + temp_var1)
        wavenumber = temp_var2 + temp_var3
        return wavenumber

    def calc_xi(self, wavenumber):
        k = wavenumber
        L = self.length_meters
        x = self.displacements_meters

        xi = np.cos(k*(L - x)) / np.cos(k*L)
        return xi

    def solve(self, unknowns: LopezBaezaShortUnknowns) -> SignalProperties:
        wavenumber = self.calc_wavenumber(
            unknowns['thermal_diffusivity_m2__s'],
            unknowns['convective_heat_transfer_coefficient_W__m2_K'],
            calc_thermal_conductivity(
                unknowns['thermal_diffusivity_m2__s'],
                self.specific_heat_capacity_J__kg_K,
                self.density_kg__m3,
            ),
        )
        xi = self.calc_xi(wavenumber)

        amps = np.abs(xi)
        amp_ratio = amps / amps[0]

        phases = np.angle(xi)
        phase_diff = phases - phases[0]

        return SignalProperties(amp_ratio, phase_diff)

    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties:
        thermal_diffusivity_m2__s, convective_heat_transfer_coefficient_W__m2_K = unknowns_vector
        wavenumber = self.calc_wavenumber(
            thermal_diffusivity_m2__s,
            convective_heat_transfer_coefficient_W__m2_K,
            calc_thermal_conductivity(
                thermal_diffusivity_m2__s,
                self.specific_heat_capacity_J__kg_K,
                self.density_kg__m3,
            ),
        )
        xi = self.calc_xi(wavenumber)

        amps = np.abs(xi)
        amp_ratio = amps / amps[0]

        phases = np.angle(xi)
        phase_diff = phases - phases[0]

        return SignalProperties(amp_ratio, phase_diff)

    def propose(
            self,
            unknowns: LopezBaezaShortUnknowns
    ) -> LopezBaezaShortUnknowns:
        raise NotImplementedError()

    def log_posterior(
            self,
            unknowns: LopezBaezaShortUnknowns,
            observed_properties: SignalProperties,
    ) -> float:
        raise NotImplementedError()

class LogSolution(Solution):
    # TODO: Docstring

    def unknowns_to_vector(
            self,
            unknowns: LogLopezBaezaShortUnknowns,
    ) -> np.ndarray:
        vector = np.array([
            unknowns['thermal_diffusivity_log10_m2__s'],
            unknowns['convective_heat_transfer_coefficient_log10_W__m2_K']
        ])
        return vector

    def vector_to_unknowns(
            self,
            vector: np.ndarray,
    ) -> LogLopezBaezaShortUnknowns:
        unknowns: LogLopezBaezaShortUnknowns = {
            'thermal_diffusivity_log10_m2__s': vector[0],
            'convective_heat_transfer_coefficient_log10_W__m2_K': vector[1],
        }
        return unknowns

    def solve(self, unknowns: LogLopezBaezaShortUnknowns) -> SignalProperties:
        unknowns_vector = np.power(10.0, self.unknowns_to_vector(unknowns))
        return super().vector_solve(unknowns_vector)

    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties:
        return super().vector_solve(np.power(10.0, unknowns_vector))
