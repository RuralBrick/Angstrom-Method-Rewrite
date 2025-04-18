import logging
from random import randint
from typing import TypedDict

import numpy as np
from scipy.stats import norm, multivariate_normal

from pyangstrom.helpers import calc_thermal_conductivity
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins
from pyangstrom.signal import SignalProperties
from pyangstrom.fitting_methods.nelder_mead import NelderMeadEquations
from pyangstrom.fitting_methods.lsr import LsrEquations
from pyangstrom.fitting_methods.metropolis_hastings import MetropolisHastingsEquations


logger = logging.getLogger('fit')

class LopezBaezaShortUnknowns(TypedDict):
    """The unknowns in Lopez-Baeza's Solution for Short Samples."""
    thermal_diffusivity_m2__s: float
    convective_heat_transfer_coefficient_W__m2_K: float

class LogLopezBaezaShortUnknowns(TypedDict):
    """The log variants of the unknowns in Lopez-Baeza's Solution for Short
    Samples.
    """
    thermal_diffusivity_log10_m2__s: float
    convective_heat_transfer_coefficient_log10_W__m2_K: float

class LopezBaezaShortMcmcValues(LogLopezBaezaShortUnknowns):
    # TODO
    """"""
    log_stdev_amplitude_ratio: float
    log_stdev_phase_difference: float
    fisher_signal_properties_correlation_coefficient: float

class LopezBaezaShortMcmcUnknowns(LopezBaezaShortMcmcValues):
    # TODO
    """"""
    prior_means: LopezBaezaShortMcmcValues
    prior_stdev: LopezBaezaShortMcmcValues
    proposal_stdev: LopezBaezaShortMcmcValues
    signal_properties_proposal_correlation_coefficient: float
    thermal_properties_proposal_correlation_coefficient: float

class Solution(
    NelderMeadEquations,
    LsrEquations,
):
    """Implements equations for Lopez-Baeza's Solution for Short Samples."""
    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            r_meters: float,
            length_meters: float,
    ) -> None:
        """For more details, see
        https://github.com/RuralBrick/Angstrom-Method-Rewrite/wiki/Sample-Solutions#lopez-baezas-solution-for-short-samples
        """
        self.rng = np.random.default_rng()
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

class LogSolution(
    Solution,
    MetropolisHastingsEquations,
):
    """Implements equations for the log variant of Lopez-Baeza's Solution for
    Short Samples.
    """
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

    def propose(
            self,
            unknowns: LopezBaezaShortMcmcUnknowns,
    ) -> LopezBaezaShortMcmcUnknowns:
        sigma_alpha = unknowns['proposal_stdev']['thermal_diffusivity_log10_m2__s']
        sigma_h = unknowns['proposal_stdev']['convective_heat_transfer_coefficient_log10_W__m2_K']
        rho_alpha_h = unknowns['thermal_properties_proposal_correlation_coefficient']
        cov_thermal_properties = [
            [sigma_alpha * sigma_alpha, rho_alpha_h * sigma_alpha * sigma_h],
            [rho_alpha_h * sigma_alpha * sigma_h, sigma_h * sigma_h],
        ]
        new_thermal_properties = self.rng.multivariate_normal(
            [
                unknowns['thermal_diffusivity_log10_m2__s'],
                unknowns['convective_heat_transfer_coefficient_log10_W__m2_K'],
            ],
            cov_thermal_properties,
        )

        # HACK
        new_thermal_properties[1] = unknowns['convective_heat_transfer_coefficient_log10_W__m2_K']
        # end HACK

        sigma_dA = unknowns['proposal_stdev']['log_stdev_amplitude_ratio']
        sigma_dP = unknowns['proposal_stdev']['log_stdev_phase_difference']
        rho_sigma_dA_dP = unknowns['signal_properties_proposal_correlation_coefficient']
        cov_stdev_signal_properties = [
            [sigma_dA * sigma_dA, rho_sigma_dA_dP * sigma_dA * sigma_dP],
            [rho_sigma_dA_dP * sigma_dA * sigma_dP, sigma_dP * sigma_dP],
        ]
        new_stdev_signal_properties = self.rng.multivariate_normal(
            [
                unknowns['log_stdev_amplitude_ratio'],
                unknowns['log_stdev_phase_difference'],
            ],
            cov_stdev_signal_properties,
        )

        new_fisher_signal_properties_correlation_coefficient = self.rng.normal(
            unknowns['fisher_signal_properties_correlation_coefficient'],
            unknowns['proposal_stdev']['fisher_signal_properties_correlation_coefficient'],
        )

        proposal: LopezBaezaShortMcmcUnknowns = {
            'thermal_diffusivity_log10_m2__s': new_thermal_properties[0],
            'convective_heat_transfer_coefficient_log10_W__m2_K': new_thermal_properties[1],
            'log_stdev_amplitude_ratio': new_stdev_signal_properties[0],
            'log_stdev_phase_difference': new_stdev_signal_properties[1],
            'fisher_signal_properties_correlation_coefficient': new_fisher_signal_properties_correlation_coefficient,
            'prior_means': unknowns['prior_means'],
            'prior_stdev': unknowns['prior_stdev'],
            'proposal_stdev': unknowns['proposal_stdev'],
            'thermal_properties_proposal_correlation_coefficient': unknowns['thermal_properties_proposal_correlation_coefficient'],
            'signal_properties_proposal_correlation_coefficient': unknowns['signal_properties_proposal_correlation_coefficient'],
        }

        return proposal

    def manual_priors(self, unknowns: LopezBaezaShortMcmcUnknowns):
        log_p_log_alpha = norm.logpdf(
            unknowns['thermal_diffusivity_log10_m2__s'],
            loc=unknowns['prior_means']['thermal_diffusivity_log10_m2__s'],
            scale=unknowns['prior_stdev']['thermal_diffusivity_log10_m2__s'],
        )
        log_p_log_h = norm.logpdf(
            unknowns['convective_heat_transfer_coefficient_log10_W__m2_K'],
            loc=unknowns['prior_means']['convective_heat_transfer_coefficient_log10_W__m2_K'],
            scale=unknowns['prior_stdev']['convective_heat_transfer_coefficient_log10_W__m2_K'],
        )

        log_p_log_sigma_dA = norm.logpdf(
            unknowns['log_stdev_amplitude_ratio'],
            loc=unknowns['prior_means']['log_stdev_amplitude_ratio'],
            scale=unknowns['prior_stdev']['log_stdev_amplitude_ratio'],
        )
        log_p_log_sigma_dP = norm.logpdf(
            unknowns['log_stdev_phase_difference'],
            loc=unknowns['prior_means']['log_stdev_phase_difference'],
            scale=unknowns['prior_stdev']['log_stdev_phase_difference'],
        )

        priors_total = (
            log_p_log_alpha
            + log_p_log_h
            + log_p_log_sigma_dA
            + log_p_log_sigma_dP
        )
        return priors_total

    def log_likelihood(
            self,
            unknowns: LopezBaezaShortMcmcUnknowns,
            observed_properties: SignalProperties,
    ):
        theoretical_properties = self.solve(unknowns)

        observed_pairs = np.stack(
            (
                observed_properties.amplitude_ratios.flatten(),
                observed_properties.phase_differences.flatten(),
            ),
            axis=-1,
        )
        theoretical_pairs = np.stack(
            (
                theoretical_properties.amplitude_ratios.flatten(),
                theoretical_properties.phase_differences.flatten(),
            ),
            axis=-1,
        )

        sigma_dA = max(unknowns['log_stdev_amplitude_ratio'], 1e-3)
        sigma_dP = max(unknowns['log_stdev_phase_difference'], 1e-3)
        rho_dA_dP = np.clip(
            np.tanh(
                unknowns['fisher_signal_properties_correlation_coefficient']
            ),
            -0.99,
            0.99,
        )
        cov_errs = [
            [sigma_dA ** 2, sigma_dA * sigma_dP * rho_dA_dP],
            [sigma_dA * sigma_dP * rho_dA_dP, sigma_dP ** 2],
        ] + np.eye(2) * 1e-6

        likelihood_total = sum(
            multivariate_normal.logpdf(o, t, cov_errs)
            for o, t in zip(observed_pairs, theoretical_pairs)
        )

        # if randint(0, 100) == 100:
        #     logger.info(list(multivariate_normal.pdf(o, t, cov_errs)
        #     for o, t in zip(observed_pairs, theoretical_pairs))[-1])
        #     logger.info(list(zip(observed_pairs, theoretical_pairs))[-1])

        return likelihood_total

    def log_posterior(
            self,
            unknowns: LopezBaezaShortMcmcUnknowns,
            observed_properties: SignalProperties,
    ) -> float:
        likelihood = self.log_likelihood(unknowns, observed_properties)

        priors = self.manual_priors(unknowns)

        log_alpha = unknowns['thermal_diffusivity_log10_m2__s']
        log_h = unknowns['convective_heat_transfer_coefficient_log10_W__m2_K']
        log_sigma_dA = unknowns['log_stdev_amplitude_ratio']
        log_sigma_dP = unknowns['log_stdev_phase_difference']
        z = unknowns['fisher_signal_properties_correlation_coefficient']
        jac = (
            np.log(
                10 ** -(log_alpha+log_h+log_sigma_dA+log_sigma_dP)
            )
            + np.log(
                (1 + np.exp(2*z))
                / (4 * np.exp(2*z))
            )
        )

        # jac = (
        #     np.log(10 ** (log_alpha+log_h+log_sigma_dA+log_sigma_dP))
        #     + 2 * np.log(1 + np.exp(z))
        #     + np.log(1 + np.exp(4 * z))
        # )

        # top = np.log(
        #         10 ** -(log_alpha+log_h+log_sigma_dA+log_sigma_dP)
        #     )
        # bottom = np.log(
        #         (1 + np.exp(2*z))
        #         / (4 * np.exp(2*z))
        #     )

        # if randint(0, 100) == 100:
        #     logger.info(f"{likelihood=}, {priors=}")

        posterior_total = likelihood + priors + jac

        return posterior_total
