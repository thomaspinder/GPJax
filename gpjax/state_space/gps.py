"""StateSpacePrior and StateSpaceConjugatePosterior classes.

See plans/2026-04-21-state-space-gps-design.md.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import paramax

from gpjax.distributions import GaussianDistribution
from gpjax.gps import ConjugatePosterior, Prior
from gpjax.likelihoods import Gaussian, MultiOutputGaussian


class StateSpacePrior(Prior):
    """Prior for a state-space (Markovian) GP.

    Identical to ``gpjax.gps.Prior`` except predictions are diagonal-only
    (the prior is stationary in time, so off-diagonal covariance carries no
    extra information for v1's diagonal-only predictive contract).
    """

    def __call__(self, test_inputs, *, return_covariance_type="diagonal"):
        return self.predict(test_inputs, return_covariance_type=return_covariance_type)

    def predict(self, test_inputs, *, return_covariance_type="diagonal"):
        if return_covariance_type != "diagonal":
            raise NotImplementedError(
                "State-space prior predict only supports return_covariance_type='diagonal' in v1."
            )
        from gpjax.state_space.kernels import to_sde

        sde = to_sde(self.kernel)
        H = sde.observation_matrix
        L_inf = sde.stationary_state_cov_sqrt
        P_inf = L_inf @ L_inf.T
        marginal_variance = (H @ P_inf @ H.T).squeeze() + self.jitter

        n_test = test_inputs.shape[0]
        mean_at_test = self.mean_function(test_inputs)
        loc = jnp.atleast_1d(mean_at_test.squeeze())
        scale = lx.DiagonalLinearOperator(jnp.full(n_test, marginal_variance))
        return GaussianDistribution(loc=loc, scale=scale)

    def __mul__(self, other):
        _require_scalar_gaussian_likelihood(other)
        return StateSpaceConjugatePosterior(prior=self, likelihood=other)


class StateSpaceConjugatePosterior(ConjugatePosterior):
    """Conjugate posterior for a state-space (Markovian) GP.

    v1 prediction surface:
      - ``predict``        : smoothed-latent prediction (Phase 10)
      - ``predict_filter`` : causal filtered prediction (Phase 10)
      - ``__call__``       : delegates to ``predict``
    Both ``predict`` and ``predict_filter`` reject ``return_covariance_type="dense"``
    in favour of v1's diagonal-only contract before any further dispatch.
    """

    def __call__(
        self,
        test_inputs,
        train_data,
        *,
        return_covariance_type="diagonal",
        observation_mask=None,
    ):
        return self.predict(
            test_inputs,
            train_data,
            return_covariance_type=return_covariance_type,
            observation_mask=observation_mask,
        )

    def predict(
        self,
        test_inputs,
        train_data,
        *,
        return_covariance_type="diagonal",
        observation_mask=None,
    ):
        if return_covariance_type != "diagonal":
            raise NotImplementedError(
                "State-space posterior predict only supports return_covariance_type='diagonal' in v1."
            )
        raise NotImplementedError("Smoothed state-space prediction lands in Phase 10.")

    def predict_filter(
        self,
        test_inputs,
        train_data,
        *,
        return_covariance_type="diagonal",
        observation_mask=None,
    ):
        if return_covariance_type != "diagonal":
            raise NotImplementedError(
                "State-space posterior predict_filter only supports return_covariance_type='diagonal' in v1."
            )
        raise NotImplementedError("Filtered state-space prediction lands in Phase 10.")


def _require_scalar_gaussian_likelihood(likelihood) -> None:
    """Raise if ``likelihood`` is not a single-output, scalar-stddev Gaussian.

    State-space v1 supports only ``gpjax.likelihoods.Gaussian`` with a scalar
    ``obs_stddev`` and ``num_outputs == 1``.
    """
    if isinstance(likelihood, MultiOutputGaussian):
        raise TypeError(
            "State-space inference requires a single-output Gaussian likelihood; "
            "MultiOutputGaussian is not supported in v1."
        )
    if not isinstance(likelihood, Gaussian):
        raise TypeError(
            f"State-space inference requires a Gaussian (conjugate) likelihood; "
            f"got {type(likelihood).__name__}."
        )
    obs_stddev_value = paramax.unwrap(likelihood.obs_stddev)
    if jnp.asarray(obs_stddev_value).ndim != 0:
        raise ValueError(
            f"State-space Gaussian likelihood requires a scalar obs_stddev; "
            f"got shape {jnp.asarray(obs_stddev_value).shape}."
        )
