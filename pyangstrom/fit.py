from typing import Dict, Callable, Any, TypedDict
from dataclasses import dataclass

from pyangstrom.transform import Region
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import SignalProperties


Unknowns = Dict
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
    unknowns_guesses: Unknowns
    parameters: Parameters

def fit(
        props: SignalProperties,
        calc_props: PropertiesCalculator,
        fitter: FitterCallable,
        unknowns_guesses: Unknowns,
        region: Region,
        setup: ExperimentalSetup,
) -> FittingResult:
    def residuals(unknowns):
        residuals = sum(
            prop - calc_prop for prop, calc_prop
            in zip(props, calc_props(unknowns, region, setup))
        )
        return residuals
    result = fitter(residuals, unknowns_guesses)
    return result
