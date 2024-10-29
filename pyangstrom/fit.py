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
    """Contains solutions to unknowns and method-specific metadata."""
    unknowns_solutions: Unknowns

@dataclass
class FittingResult:
    """Contains thermal properties and metadata for validating the results."""
    unknowns_solutions: Unknowns
    theoretical_properties: SignalProperties

class EquationPackage(abc.ABC):
    """Declares methods required by the fitting method to calculate heat model
    properties.
    """
    @abc.abstractmethod
    def __init__(
            self,
            margins: Margins,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> None:
        """EquationPackages are expected to be initialized with only margins,
        setup, and any constants relevant to its heat model.
        """
        ...

    @abc.abstractmethod
    def solve(self, unknowns: Unknowns) -> SignalResult:
        """Calculates signal properties and metadata based on the given values
        for the unknowns.
        """
        ...

class Fitter(Protocol):
    """Function signature for a valid fitting method."""
    def __call__(
            self,
            unknowns_guesses: Unknowns,
            solution: EquationPackage,
            observed_properties: SignalProperties,
            **kwargs,
    ) -> FitterOutput: ...

class SolverInformation(TypedDict, total=False):
    """Specifies all heat model details.

    Attributes
    ----------
    name
        The name of a sample solution. Names can be found on the Sample
        Solutions wiki page. Ignored if 'solution' is present.
    solution
        A reference to an EquationPackage class. Takes precedence over 'name'.
    guesses
        Initial guess values for the the heat model's unknowns.
    parameters
        Arguments passed onto the chosen sample solution. Check the Sample
        Solutions wiki page for exact details.

    References
    ----------
    Sample Solutions:
    https://github.com/RuralBrick/Angstrom-Method-Rewrite/wiki/Sample-Solutions
    """
    name: str
    solution: EquationPackage
    guesses: Unknowns
    parameters: dict

class FitterInformation(TypedDict, total=False):
    """Specifies how to fit to the signal properties.

    Attributes
    ----------
    name
        The name of a built-in fitting method. Names can be found on the
        Built-in Fitting Methods wiki page. Ignored if 'fitter' is present.
    fitter
        A reference to a fitter function. Takes precedence over 'name'.
    parameters
        Arguments passed onto the chosen built-in fitting method. Check the
        Build-in Fitting Methods wiki page for exact details.

    References
    ----------
    Built-in Fitting Methods:
    https://github.com/RuralBrick/Angstrom-Method-Rewrite/wiki/Built%E2%80%90in-Fitting-Methods
    """
    name: str
    fitter: Fitter
    parameters: dict

def extract_solution_class(
        solver_information: SolverInformation
) -> Optional[Type[EquationPackage]]:
    """Returns None if 'name' not in solver_information.

    Raises
    ------
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
        case 'log_kil':
            from pyangstrom.sample_solutions.kil_circular_room_temp import LogSolution as Solution
        case _:
            raise ValueError(f"Solver {solver_information['name']} not found.")
    return Solution

def extract_solution(
        solver_information: SolverInformation,
        signal_result: SignalResult,
        setup: ExperimentalSetup,
) -> EquationPackage:
    """
    Raises
    ------
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
    Raises
    ------
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
    """Fits the specified heat model to the experimental signal properties as
    specified by fitter information configuration.

    Raises
    ------
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
