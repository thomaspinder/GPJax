"""State-space (Markovian) GP benchmarks.

The headline win for state-space GPs is O(N · d^3) inference where the
equivalent dense GP is O(N^3). The MLL suite below is the one that
exposes that scaling on the dashboard.
"""

from __future__ import annotations

import gpjax as gpx
from gpjax.state_space import StateSpacePrior, state_space_mll
from gpjax.state_space.gps import StateSpaceConjugateModel
import jax.numpy as jnp
import jax.random as jr

from benchmarks._setup import KEY, realise


def _temporal_dataset(n: int, key=KEY) -> gpx.Dataset:
    times = jnp.sort(jr.uniform(key, (n,), dtype=jnp.float64) * 10.0)
    targets = jnp.sin(times) + 0.05 * jr.normal(
        jr.fold_in(key, 1), (n,), dtype=jnp.float64
    )
    return gpx.Dataset(X=times.reshape(-1, 1), y=targets.reshape(-1, 1))


class StateSpaceMllSuite:
    """Square-root Kalman filter MLL — the inner objective for fitting."""

    params = ([1000, 10000],)
    param_names = ("n",)

    def setup(self, n):
        prior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        )
        likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        self.posterior = StateSpaceConjugateModel(prior=prior, likelihood=likelihood)
        self.data = _temporal_dataset(n)
        realise(state_space_mll(self.posterior, self.data))

    def time_state_space_mll(self, n):
        realise(state_space_mll(self.posterior, self.data))
