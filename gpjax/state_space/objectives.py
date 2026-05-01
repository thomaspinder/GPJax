"""Marginal log-likelihood objective for state-space GPs."""

from __future__ import annotations

import jax.numpy as jnp
import paramax

from gpjax.state_space.inference import kalman_filter
from gpjax.state_space.kernels import to_sde


def state_space_mll(
    posterior,
    train_data,
    *,
    observation_mask=None,
):
    """Marginal log-likelihood via the square-root Kalman filter.

    Internally:
      1. Unwraps the posterior (resolving paramax-wrapped parameters).
      2. Builds the SDE via ``to_sde(kernel)``.
      3. Centres targets with ``y - mean_function(X)``.
      4. Computes ``sigma_eff = sqrt(obs_stddev² + prior.jitter)``.
      5. Delegates to ``kalman_filter``.

    Pure-JAX; assumes prevalidated, sorted data. Validation belongs upstream
    in the ``state_space.fit*`` wrappers.

    See plans/2026-04-21-state-space-gps-design.md §Stage 1.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior, state_space_mll
        >>> X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        >>> y = jnp.sin(X)
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=20, obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> train_data = gpx.Dataset(X=X, y=y)
        >>> mll = state_space_mll(posterior, train_data)
        >>> bool(jnp.isfinite(mll))
        True
    ```
    """
    posterior = paramax.unwrap(posterior)
    prior = posterior.prior
    likelihood = posterior.likelihood

    X = train_data.X
    y = train_data.y
    times = X.squeeze(-1)
    targets = y.squeeze(-1)

    mean_at_train = prior.mean_function(X).squeeze(-1)
    centred_targets = targets - mean_at_train

    sde = to_sde(prior.kernel)
    obs_variance = likelihood.obs_stddev**2
    sigma_eff = jnp.sqrt(obs_variance + prior.jitter)

    time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(times)])
    if observation_mask is None:
        is_observed = jnp.ones(times.shape[0], dtype=bool)
    else:
        is_observed = observation_mask

    return kalman_filter(sde, centred_targets, time_steps, is_observed, sigma_eff)
