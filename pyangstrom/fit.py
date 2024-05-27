from typing import TypedDict, Iterable, Callable, Any, Protocol
from dataclasses import dataclass

import numpy as np

from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import RegionProperties, SignalProperties, SignalResult


class SolverInformation(TypedDict, total=False):
    name: str
    parameters: dict

class FitterInformation(TypedDict, total=False):
    name: str
    guesses: dict
    parameters: dict

Unknowns = Iterable
UnknownsFormatter = Callable[..., Unknowns]
TheoreticalCalculator = Callable[[Unknowns], SignalProperties]

@dataclass
class FittingResult:
    unknowns_solutions: Unknowns

class Solver(Protocol):
    def __call__(
            self,
            unknowns: Unknowns,
            region_properties: RegionProperties,
            setup: ExperimentalSetup,
            **kwargs,
    ) -> SignalProperties: ...

class Fitter(Protocol):
    def __call__(
            self,
            unknowns_guesses: Unknowns,
            unknowns_to_props: TheoreticalCalculator,
            signal_properties: SignalProperties,
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
    solve: Solver = None
    guess_converter: UnknownsFormatter = None
    match solver_information['name']:
        case 'lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import solve
            from pyangstrom.sample_solutions.lopez_baeza_short import LopezBaezaShortUnknowns as guess_converter
        case 'log_lopez-baeza':
            from pyangstrom.sample_solutions.lopez_baeza_short import log_solve as solve
            from pyangstrom.sample_solutions.lopez_baeza_short import LogLopezBaezaShortUnknowns as guess_converter
        case _:
            raise ValueError(f"Solver {solver_information['name']} not found.")

    solver_params = (solver_information['parameters']
                     if 'parameters' in solver_information
                     else {})

    def unknowns_to_props(unknowns):
        props = solve(
            unknowns,
            signal_result.region_properties,
            setup,
            **solver_params,
        )
        return props

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
        guess_converter(**fitter_information['guesses']),
        unknowns_to_props,
        signal_result.signal_properties,
        **fitter_params,
    )
    return result
