from typing import NamedTuple, Dict, Callable, Any, TypedDict, NotRequired
from dataclasses import dataclass

import numpy as np

from pyangstrom.transform import Region
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.signal import SignalProperties


Unknowns = NamedTuple
Guesses = Dict
Displacement = np.ndarray
Parameters = Dict
PropertiesCalculator = Callable[
    [Unknowns, Displacement, ExperimentalSetup, Parameters],
    SignalProperties,
]
UnknownsFormatter = Callable[..., Unknowns]
ResidualsCallable = Callable[[Unknowns], Any]

@dataclass
class FittingResult:
    unknowns_solutions: Unknowns

FitterCallable = Callable[
    [ResidualsCallable, Unknowns, Parameters],
    FittingResult,
]

class Solver(TypedDict):
    name: str
    parameters: NotRequired[Parameters]

class Fitter(TypedDict):
    name: str
    guesses: Guesses
    parameters: NotRequired[Parameters]

def region_to_displacement(region: Region) -> np.ndarray:
    disp = np.linspace(0, region.margins[1], region.temps_kelvin.shape[1])
    return disp

def fit(
        region: Region,
        props: SignalProperties,
        calc_props: PropertiesCalculator,
        guess_converter: UnknownsFormatter,
        fitter: FitterCallable,
        guesses: Guesses,
        setup: ExperimentalSetup,
        solver_parameters: Parameters = {},
        fitter_parameters: Parameters = {},
) -> FittingResult:
    disp = region_to_displacement(region)
    def residuals(unknowns):
        residuals = np.stack([
            prop - np.expand_dims(calc_prop, tuple(range(1, len(prop.shape))))
            for prop, calc_prop in zip(
                props,
                calc_props(unknowns, disp, setup, solver_parameters),
            )
        ]).flatten()
        return residuals
    result = fitter(residuals, guess_converter(**guesses), fitter_parameters)
    return result
