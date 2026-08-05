"""StateSpacePrior and StateSpaceConjugatePosterior classes.

See plans/2026-04-21-state-space-gps-design.md.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import paramax

from gpjax.distributions import GaussianDistribution
from gpjax.gps import ConjugateModel, Prior
from gpjax.likelihoods import Gaussian, MultiOutputGaussian


class StateSpacePrior(Prior):
    """Prior for a state-space (Markovian) GP.

    Identical to ``gpjax.gps.Prior`` except predictions are diagonal-only
    (the prior is stationary in time, so off-diagonal covariance carries no
    extra information for v1's diagonal-only predictive contract).

    **Predictive contract (v1):** prediction returns diagonal (marginal)
    covariance only; the marginals are exact. A dense joint predictive is not
    implemented in v1 and is tracked as a follow-up. This predictive is
    therefore not Liskov-substitutable for a dense
    dense ``gpjax.gps.ConjugateModel`` predictive.

    Example:
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> isinstance(prior.kernel, gpx.kernels.Matern32)
        True
    """

    def __call__(self, test_inputs, *, covariance="diagonal"):
        return self.predict(test_inputs, covariance=covariance)

    def predict(self, test_inputs, *, covariance="diagonal"):
        if covariance != "diagonal":
            raise NotImplementedError(
                "State-space prior prediction returns diagonal (marginal) covariance "
                "only; a dense joint predictive is not implemented in v1 and is "
                "tracked as a follow-up. The marginal variances returned are exact, "
                "so for diagonal-only use pass covariance='diagonal'."
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


class StateSpaceConjugatePosterior(ConjugateModel):
    """Conjugate posterior for a state-space (Markovian) GP.

    v1 prediction surface:
      - ``predict``        : smoothed-latent prediction (Phase 10)
      - ``predict_filter`` : causal filtered prediction (Phase 10)
      - ``__call__``       : delegates to ``predict``

    Both ``predict`` and ``predict_filter`` reject ``covariance="dense"``
    in favour of v1's diagonal-only contract before any further dispatch.

    **Predictive contract (v1):** prediction returns diagonal (marginal)
    covariance only; the marginals are exact. A dense joint predictive is not
    implemented in v1 and is tracked as a follow-up. This predictive is
    therefore not Liskov-substitutable for a dense
    dense ``gpjax.gps.ConjugateModel`` predictive.

    Example:
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> posterior.__class__.__name__
        'StateSpaceConjugatePosterior'
    """

    def __call__(
        self,
        test_inputs,
        train_data,
        *,
        covariance="diagonal",
        observation_mask=None,
    ):
        return self.predict(
            test_inputs,
            train_data,
            covariance=covariance,
            observation_mask=observation_mask,
        )

    def predict(
        self,
        test_inputs,
        train_data,
        *,
        covariance="diagonal",
        observation_mask=None,
    ):
        if covariance != "diagonal":
            raise NotImplementedError(
                "State-space posterior predict returns diagonal (marginal) "
                "covariance only; a dense joint predictive is not implemented in v1 "
                "and is tracked as a follow-up. The marginal variances returned are "
                "exact, so for diagonal-only use pass "
                "covariance='diagonal'."
            )
        from gpjax.state_space.prediction import predict_smoothed

        return predict_smoothed(
            self, train_data, test_inputs, observation_mask=observation_mask
        )

    def predict_filter(
        self,
        test_inputs,
        train_data,
        *,
        covariance="diagonal",
        observation_mask=None,
    ):
        if covariance != "diagonal":
            raise NotImplementedError(
                "State-space posterior predict_filter returns diagonal (marginal) "
                "covariance only; a dense joint predictive is not implemented in v1 "
                "and is tracked as a follow-up. The marginal variances returned are "
                "exact, so for diagonal-only use pass "
                "covariance='diagonal'."
            )
        from gpjax.state_space.prediction import predict_filtered

        return predict_filtered(
            self, train_data, test_inputs, observation_mask=observation_mask
        )


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
