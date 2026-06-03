"""State-space (Markovian) Gaussian processes for GPJax.

See plans/2026-04-21-state-space-gps-design.md for the full design.
"""

from gpjax.state_space.fit import fit, fit_lbfgs, fit_scipy
from gpjax.state_space.gps import StateSpaceConjugatePosterior, StateSpacePrior
from gpjax.state_space.kernels import TruncatedPeriodic, to_sde
from gpjax.state_space.objectives import state_space_mll

__all__ = [
    "StateSpaceConjugatePosterior",
    "StateSpacePrior",
    "TruncatedPeriodic",
    "fit",
    "fit_lbfgs",
    "fit_scipy",
    "state_space_mll",
    "to_sde",
]
