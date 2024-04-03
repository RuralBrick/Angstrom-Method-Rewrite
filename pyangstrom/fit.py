from typing import TypeVarTuple, Callable, Any
from dataclasses import dataclass

from pyangstrom.signal import SignalProperties


Ts = TypeVarTuple('Ts')
Unknowns = tuple[*Ts]
ResidualsCallable = Callable[[Unknowns], Any]

@dataclass
class FittingResult:
    solution_params: Unknowns

def fit(
        props: SignalProperties,
        calc_props: Callable[[Unknowns], SignalProperties],
        fitter: Callable[[ResidualsCallable, Unknowns], FittingResult],
        guess_params: Unknowns,
) -> FittingResult:
    def residuals(params):
        residuals = sum(
            prop - calc_prop for prop, calc_prop
            in zip(props, calc_props(params))
        )
        return residuals
    result = fitter(residuals, guess_params)
    return result
