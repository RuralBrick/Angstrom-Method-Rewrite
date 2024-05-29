from typing import TypedDict, Protocol
import abc
from dataclasses import dataclass

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import RegionProperties, SignalProperties, SignalResult


Unknowns = dict

class SolverInformation(TypedDict, total=False):
    name: str
    guesses: Unknowns
    parameters: dict

class FitterInformation(TypedDict, total=False):
    name: str
    parameters: dict

@dataclass
class FittingResult:
    unknowns_solutions: Unknowns

class EquationPackage(abc.ABC):
    @abc.abstractmethod
    def __init__(
            self,
            region_properties: RegionProperties,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> None: ...

class Fitter(Protocol):
    def __call__(
            self,
            unknowns_guesses: Unknowns,
            solution: EquationPackage,
            observed_properties: SignalProperties,
            **kwargs,
    ) -> FittingResult: ...

def autofit(
        signal_result: SignalResult,
        solver_information: SolverInformation,
        fitter_information: FitterInformation,
        setup: ExperimentalSetup,
) -> FittingResult:
    """
    Exceptions
    ----------
    KeyError
        Field not found in information.
    ValueError
        Named solver or fitter not found.
    """
    Solution = EquationPackage
    match solver_information['name']:
        case 'lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import Solution
        case 'log_lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import LogSolution as Solution
        case _:
            raise ValueError(f"Solver {solver_information['name']} not found.")

    solver_params = (solver_information['parameters']
                     if 'parameters' in solver_information
                     else {})

    fit: Fitter = None
    match fitter_information['name']:
        case 'least_squares_regression' | 'lsr':
            from pyangstrom.fitting_methods.lsr import fit
        case 'nelder-mead':
            from pyangstrom.fitting_methods.nelder_mead import fit
        case _:
            raise ValueError(f"Fitter {fitter_information['name']} not found.")

    fitter_params = (fitter_information['parameters']
                     if 'parameters' in fitter_information
                     else {})

    result = fit(
        solver_information['guesses'],
        Solution(signal_result.region_properties, setup, **solver_params),
        signal_result.signal_properties,
        **fitter_params,
    )
    return result
