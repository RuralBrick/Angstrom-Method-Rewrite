from typing import NamedTuple, Dict, Callable, Any, TypedDict
from dataclasses import dataclass

from pyangstrom.transform import Region
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import SignalProperties


Unknowns = NamedTuple
Guesses = Dict
Parameters = Dict
PropertiesCalculator = Callable[
    [Unknowns, Region, ExperimentalSetup, Parameters],
    SignalProperties,
]
ResidualsCallable = Callable[[Unknowns], Any]

@dataclass
class FittingResult:
    unknowns_solutions: Unknowns

FitterCallable = Callable[[ResidualsCallable, Unknowns], FittingResult]

class Solver(TypedDict):
    name: str
    parameters: Parameters

class Fitter(TypedDict):
    name: str
    guesses: Guesses
    parameters: Parameters

def fit(
        props: SignalProperties,
        calc_props: PropertiesCalculator,
        solver_parameters: Parameters,
        fitter: FitterCallable,
        fitter_parameters: Parameters,
        guesses: Guesses,
        region: Region,
        setup: ExperimentalSetup,
) -> FittingResult:
    def residuals(unknowns):
        residuals = sum(
            prop - calc_prop for prop, calc_prop
            in zip(
                props,
                calc_props(unknowns, region, setup, solver_parameters),
            )
        )
        return residuals
    result = fitter(residuals, guesses, fitter_parameters)
    return result
