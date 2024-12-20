import logging
from enum import Enum
import abc

import numpy as np
from scipy.optimize import minimize

from pyangstrom.fit import (
    Unknowns,
    EquationPackage,
    SignalProperties,
    FitterOutput,
)
from pyangstrom.signal import SignalProperties


logger = logging.getLogger('fit')

class UsedProperties(Enum):
    """The signal properties with which to calculate error.

    Click ">EXPAND SOURCE CODE" to see string equivalents.
    """
    AMPLITUDE_RATIOS_AND_PHASE_DIFFERENCES = 'phase-amplitude'
    AMPLITUDE_RATIOS_ONLY = 'amplitude'
    PHASE_DIFFERENCES_ONLY = 'phase'

class NelderMeadEquations(EquationPackage):
    @abc.abstractmethod
    def unknowns_to_vector(self, unknowns: Unknowns) -> np.ndarray:
        """Arranges unknowns into 1D array."""
        ...

    @abc.abstractmethod
    def vector_to_unknowns(self, vector: np.ndarray) -> Unknowns:
        """Parses unknowns from 1D array."""
        ...

    @abc.abstractmethod
    def vector_solve(self, unknowns_vector: np.ndarray) -> SignalProperties:
        """Calculates signal properties and metadata based on unknowns given as
        a 1D array.
        """
        ...

def extract_used_properties(
        properties: SignalProperties,
        properties_to_use: UsedProperties | str,
) -> np.ndarray:
    match UsedProperties(properties_to_use):
        case UsedProperties.AMPLITUDE_RATIOS_AND_PHASE_DIFFERENCES:
            return np.stack(properties)
        case UsedProperties.AMPLITUDE_RATIOS_ONLY:
            return properties.amplitude_ratios
        case UsedProperties.PHASE_DIFFERENCES_ONLY:
            return properties.phase_differences

def fit(
        unknowns_guesses: Unknowns,
        solution: NelderMeadEquations,
        observed_properties: SignalProperties,
        properties_to_use: UsedProperties | str = UsedProperties.AMPLITUDE_RATIOS_AND_PHASE_DIFFERENCES,
        **minimize_kwargs,
) -> FitterOutput:
    used_observed_properties = extract_used_properties(
        observed_properties,
        properties_to_use,
    )

    def calc_error(unknowns_vector):
        theoretical_properties = solution.vector_solve(unknowns_vector)
        used_theoretical_properties = extract_used_properties(
            theoretical_properties,
            properties_to_use,
        )
        residuals = used_observed_properties - used_theoretical_properties
        error = np.sum(np.square(residuals))
        return error

    nelder_mead_result = minimize(
        calc_error,
        solution.unknowns_to_vector(unknowns_guesses),
        method='Nelder-Mead',
        options=dict(disp=(logger.getEffectiveLevel() <= logging.DEBUG)),
        **minimize_kwargs,
    )
    output = FitterOutput(solution.vector_to_unknowns(nelder_mead_result.x))
    return output
