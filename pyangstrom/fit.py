import logging
from typing import TypedDict, Protocol, Optional, Type
import abc
from dataclasses import dataclass

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import Margins
from pyangstrom.signal import SignalProperties, SignalResult


logger = logging.getLogger('fit')

Unknowns = dict

@dataclass
class FitterOutput:
    unknowns_solutions: Unknowns

@dataclass
class FittingResult:
    unknowns_solutions: Unknowns
    theoretical_properties: SignalProperties

class EquationPackage(abc.ABC):
    @abc.abstractmethod
    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> None: ...

    @abc.abstractmethod
    def solve(self, unknowns: Unknowns) -> SignalResult: ...

class Fitter(Protocol):
    def __call__(
            self,
            unknowns_guesses: Unknowns,
            solution: EquationPackage,
            observed_properties: SignalProperties,
            **kwargs,
    ) -> FitterOutput: ...

class SolverInformation(TypedDict, total=False):
    name: str
    solution: EquationPackage
    guesses: Unknowns
    parameters: dict

class FitterInformation(TypedDict, total=False):
    name: str
    fitter: Fitter
    parameters: dict

def extract_solution_class(
        solver_information: SolverInformation
) -> Optional[Type[EquationPackage]]:
    """Returns None if 'name' not in solver_information.

    Exceptions
    ----------
    ValueError
        Named solver not found.
    """
    if 'name' not in solver_information:
        return None
    match solver_information['name']:
        case 'lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import Solution
        case 'log_lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import LogSolution as Solution
        case 'kil':
            from pyangstrom.sample_solutions.kil_circular_room_temp import Solution
        case _:
            raise ValueError(f"Solver {solver_information['name']} not found.")
    return Solution

def extract_solution(
        solver_information: SolverInformation,
        signal_result: SignalResult,
        setup: ExperimentalSetup,
) -> EquationPackage:
    """
    Exceptions
    ----------
    KeyError
        Field not found in information.
    ValueError
        Named solver not found.
    """
    if 'solution' in solver_information:
        return solver_information['solution']
    elif Solution := extract_solution_class(solver_information):
        solver_params = (solver_information['parameters']
                         if 'parameters' in solver_information
                         else {})
        return Solution(signal_result.margins, setup, **solver_params)
    else:
        raise KeyError(
            "Must have either name or solution in solver_information."
        )

def extract_fit(fitter_information: FitterInformation) -> Fitter:
    """
    Exceptions
    ----------
    KeyError
        Field not found in information.
    ValueError
        Named fitter not found.
    """
    if 'fitter' in fitter_information:
        return fitter_information['fitter']
    elif 'name' in fitter_information:
        match fitter_information['name']:
            case 'least_squares_regression' | 'lsr':
                from pyangstrom.fitting_methods.lsr import fit
            case 'nelder-mead':
                from pyangstrom.fitting_methods.nelder_mead import fit
            case _:
                raise ValueError(
                    f"Fitter {fitter_information['name']} not found."
                )
        return fit
    else:
        raise KeyError("Must have either name or fitter in fitter_information.")

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
    solution = extract_solution(solver_information, signal_result, setup)
    fit = extract_fit(fitter_information)
    fitter_params = (fitter_information['parameters']
                     if 'parameters' in fitter_information
                     else {})
    output = fit(
        solver_information['guesses'],
        solution,
        signal_result.signal_properties,
        **fitter_params,
    )
    result = FittingResult(
        output.unknowns_solutions,
        solution.solve(output.unknowns_solutions),
    )
    return result
