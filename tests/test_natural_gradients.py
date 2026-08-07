# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the Salimbeni et al. (2018) natural-gradient machinery."""

from pathlib import Path
import re

import equinox as eqx
import gpjax
from gpjax.dataset import Dataset
from gpjax.linalg import add_jitter
from gpjax.natural_gradients import (
    _first_valid_trial,
    _symmetrise,
    expectation_from_moments,
    moments_from_expectation,
    moments_from_natural,
    natural_from_moments,
    natural_gradient_step,
    partition_variational,
)
from gpjax.objectives import (
    elbo,
    variational_expectation,
)
from gpjax.parameters import (
    LowerTriangular,
    Real,
    _val,
)
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    GraphVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
from hypothesis import (
    given,
    strategies as st,
)
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import networkx as nx
import numpy as np
import paramax
import pytest

from tests._reference.conjugate_svgp import (
    conjugate_optimum as _conjugate_optimum,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)

_NUM_INDUCING = 5
_NUM_DATA = 30
_FAMILY_JITTER = 1e-8

_SALIMBENI_FAMILIES = [VariationalGaussian, WhitenedVariationalGaussian]


# ---------------------------------------------------------------------------
# Shared setups
# ---------------------------------------------------------------------------
def _conjugate_setup(
    num_inducing: int = _NUM_INDUCING,
    num_data: int = _NUM_DATA,
    jitter: float = _FAMILY_JITTER,
):
    """Build a Gaussian-likelihood SVGP setup with a non-zero mean function.

    Returns the same 4-tuple shape as :func:`_bernoulli_setup`; the observation noise
    is recoverable from ``posterior.likelihood.obs_stddev`` where a test needs it.
    """
    key_inputs, key_noise = jr.split(jr.key(42))
    inputs = jr.uniform(key_inputs, (num_data, 1), minval=-2.0, maxval=2.0)
    outputs = jnp.sin(3.0 * inputs) + 0.1 * jr.normal(key_noise, (num_data, 1))
    dataset = Dataset(X=inputs, y=outputs)

    kernel = gpjax.kernels.RBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.3))
    mean_function = gpjax.mean_functions.Constant(jnp.array(0.4))
    noise_stddev = 0.37
    likelihood = gpjax.likelihoods.Gaussian(obs_stddev=jnp.array(noise_stddev))
    posterior = gpjax.gps.Prior(mean_function=mean_function, kernel=kernel) * likelihood
    inducing_inputs = jnp.linspace(-2.0, 2.0, num_inducing).reshape(-1, 1)
    return posterior, dataset, inducing_inputs, jitter


def _bernoulli_setup(
    num_inducing: int = _NUM_INDUCING,
    num_data: int = _NUM_DATA,
    jitter: float = _FAMILY_JITTER,
):
    """Build a Bernoulli-likelihood SVGP setup -- a genuinely non-conjugate model."""
    key_inputs, key_labels = jr.split(jr.key(7))
    inputs = jr.uniform(key_inputs, (num_data, 1), minval=-2.0, maxval=2.0)
    outputs = (jr.bernoulli(key_labels, 0.5, (num_data, 1))).astype(jnp.float64)
    dataset = Dataset(X=inputs, y=outputs)

    kernel = gpjax.kernels.RBF(lengthscale=jnp.array(0.9), variance=jnp.array(1.1))
    mean_function = gpjax.mean_functions.Constant(jnp.array(0.2))
    likelihood = gpjax.likelihoods.Bernoulli()
    posterior = gpjax.gps.Prior(mean_function=mean_function, kernel=kernel) * likelihood
    inducing_inputs = jnp.linspace(-2.0, 2.0, num_inducing).reshape(-1, 1)
    return posterior, dataset, inducing_inputs, jitter


def _random_moments(seed: int, num_inducing: int):
    """Draw a random mean and a random well-conditioned Cholesky factor."""
    key_mean, key_root = jr.split(jr.key(seed))
    raw = jr.normal(key_root, (num_inducing, num_inducing))
    covariance = raw @ raw.T + jnp.eye(num_inducing)
    return jr.normal(key_mean, (num_inducing, 1)), jnp.linalg.cholesky(covariance)


def _negative_elbo(family, data):
    return -elbo(family, data)


def _moments_of(family):
    unwrapped = paramax.unwrap(family)
    return (
        _val(unwrapped.variational_mean),
        _val(unwrapped.variational_root_covariance),
    )


def _take_step(family, data, natgrad_lr, objective=_negative_elbo, **kwargs):
    """Run one natural-gradient step and recombine the partitions."""
    variational, hyper = partition_variational(family)
    updated, loss_value = natural_gradient_step(
        variational, hyper, data, objective, jnp.asarray(natgrad_lr), **kwargs
    )
    return eqx.combine(updated, hyper), loss_value


def _kernel_matrices(family, inputs):
    """Densify the kernel quantities entering the conjugate closed form."""
    unwrapped = paramax.unwrap(family)
    inducing_inputs = _val(unwrapped.inducing_inputs)
    kernel = unwrapped.posterior.prior.kernel
    mean_function = unwrapped.posterior.prior.mean_function
    gram = add_jitter(kernel.gram(inducing_inputs).as_matrix(), family.jitter)
    cross = kernel.cross_covariance(inputs, inducing_inputs)
    return (
        gram,
        cross,
        jnp.linalg.cholesky(gram),
        mean_function(inducing_inputs),
        mean_function(inputs),
    )


# ---------------------------------------------------------------------------
# A standalone non-conjugate loss, used for the exponential-family identities.
# Mirrors spec-salimbeni section 9.1: a Bernoulli/Gauss-Hermite SVGP bound with
# M = 3, written from scratch so it is independent of GPJax.
# ---------------------------------------------------------------------------
_IDENTITY_M = 3
_IDENTITY_N = 11

_gh_nodes, _gh_weights = np.polynomial.hermite_e.hermegauss(21)
_GH_NODES = jnp.asarray(_gh_nodes)
_GH_WEIGHTS = jnp.asarray(_gh_weights) / jnp.sqrt(2.0 * jnp.pi)

_key_design, _key_residual, _key_labels, _key_prior = jr.split(jr.key(0), 4)
_DESIGN = jr.normal(_key_design, (_IDENTITY_N, _IDENTITY_M)) / jnp.sqrt(_IDENTITY_M)
_RESIDUAL_VARIANCE = 0.3 + 0.2 * jr.uniform(_key_residual, (_IDENTITY_N,))
_LABELS = jnp.sign(jr.normal(_key_labels, (_IDENTITY_N,)))
_raw_prior = jr.normal(_key_prior, (_IDENTITY_M, _IDENTITY_M))
_PRIOR_PRECISION = jnp.linalg.inv(_raw_prior @ _raw_prior.T + jnp.eye(_IDENTITY_M))

_ROW, _COL = jnp.tril_indices(_IDENTITY_M)
_SCALE = jnp.where(_ROW == _COL, 1.0, jnp.sqrt(2.0))


def _vec_s(matrix):
    """Isometric flattening of a symmetric matrix: ``tr(XY) = vec_s(X).vec_s(Y)``."""
    return matrix[_ROW, _COL] * _SCALE


def _unvec_s(vector):
    lower = jnp.zeros((_IDENTITY_M, _IDENTITY_M)).at[_ROW, _COL].set(vector / _SCALE)
    return lower + lower.T - jnp.diag(jnp.diag(lower))


def _flatten(vector, matrix):
    return jnp.concatenate([vector.reshape(-1), _vec_s(matrix)])


def _unflatten(flat):
    return flat[:_IDENTITY_M].reshape(_IDENTITY_M, 1), _unvec_s(flat[_IDENTITY_M:])


def _reference_loss(mean, covariance):
    """Negative ELBO of a Bernoulli (logit) SVGP-style bound."""
    latent_mean = (_DESIGN @ mean).reshape(-1)
    latent_variance = _RESIDUAL_VARIANCE + jnp.sum(
        (_DESIGN @ covariance) * _DESIGN, axis=1
    )
    nodes = (
        latent_mean[:, None] + jnp.sqrt(latent_variance)[:, None] * _GH_NODES[None, :]
    )
    log_likelihood = -jnp.logaddexp(0.0, -_LABELS[:, None] * nodes)
    expected_log_likelihood = jnp.sum(log_likelihood @ _GH_WEIGHTS)

    _, logdet_covariance = jnp.linalg.slogdet(covariance)
    kl = 0.5 * (
        jnp.trace(_PRIOR_PRECISION @ covariance)
        + (mean.T @ _PRIOR_PRECISION @ mean).squeeze()
        - _IDENTITY_M
        - logdet_covariance
        - jnp.linalg.slogdet(_PRIOR_PRECISION)[1]
    )
    return -(expected_log_likelihood - kl)


def _reference_loss_of_moments(mean, root_covariance):
    return _reference_loss(mean, root_covariance @ root_covariance.T)


def _reference_loss_of_expectation(flat):
    return _reference_loss_of_moments(*moments_from_expectation(*_unflatten(flat)))


def _reference_loss_of_natural(flat):
    return _reference_loss_of_moments(*moments_from_natural(*_unflatten(flat)))


def _identity_point():
    return _random_moments(3, _IDENTITY_M)


# ---------------------------------------------------------------------------
# T1 -- coordinate-map round trips
# ---------------------------------------------------------------------------
@given(
    num_inducing=st.sampled_from([1, 2, 5]),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_moment_maps_round_trip(num_inducing: int, seed: int) -> None:
    """Every composition of the four maps is the identity at ``map_jitter=0``."""
    mean, root_covariance = _random_moments(seed, num_inducing)

    expectation = expectation_from_moments(mean, root_covariance)
    natural = natural_from_moments(mean, root_covariance)

    via_expectation = moments_from_expectation(*expectation)
    via_natural = moments_from_natural(*natural)
    via_both = moments_from_natural(*natural_from_moments(*via_expectation))
    natural_round_trip = natural_from_moments(
        *moments_from_expectation(*expectation_from_moments(*via_natural))
    )

    for recovered in (via_expectation, via_natural, via_both):
        np.testing.assert_allclose(
            np.float64(recovered[0]), np.float64(mean), atol=1e-10
        )
        np.testing.assert_allclose(
            np.float64(recovered[1]), np.float64(root_covariance), atol=1e-10
        )

    np.testing.assert_allclose(
        np.float64(natural_round_trip[0]), np.float64(natural[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.float64(natural_round_trip[1]), np.float64(natural[1]), atol=1e-10
    )


# ---------------------------------------------------------------------------
# T2 -- the closed form for the expectation gradient
# ---------------------------------------------------------------------------
def test_expectation_gradient_matches_closed_form() -> None:
    r"""$\partial\ell/\partial\eta_1=\partial\ell/\partial m-2(\partial\ell/\partial S)m$."""
    mean, root_covariance = _identity_point()
    covariance = root_covariance @ root_covariance.T
    expectation = expectation_from_moments(mean, root_covariance)

    def loss_of_expectation(pair):
        return _reference_loss_of_moments(*moments_from_expectation(*pair))

    gradient = jax.grad(loss_of_expectation)(expectation)
    grad_mean, grad_covariance = jax.grad(_reference_loss, argnums=(0, 1))(
        mean, covariance
    )
    grad_covariance = _symmetrise(grad_covariance)

    np.testing.assert_allclose(
        np.float64(gradient[0]),
        np.float64(grad_mean - 2.0 * grad_covariance @ mean),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.float64(gradient[1]), np.float64(grad_covariance), atol=1e-12
    )


# ---------------------------------------------------------------------------
# T3 -- the Fisher identity
# ---------------------------------------------------------------------------
def test_fisher_identity() -> None:
    r"""$F$ is symmetric positive definite and $F^{-1}\partial_\theta\ell=\partial_\eta\ell$."""
    mean, root_covariance = _identity_point()
    expectation_flat = _flatten(*expectation_from_moments(mean, root_covariance))
    natural_flat = _flatten(*natural_from_moments(mean, root_covariance))

    def expectation_of_natural(flat):
        return _flatten(
            *expectation_from_moments(*moments_from_natural(*_unflatten(flat)))
        )

    fisher = jax.jacfwd(expectation_of_natural)(natural_flat)
    grad_natural = jax.grad(_reference_loss_of_natural)(natural_flat)
    grad_expectation = jax.grad(_reference_loss_of_expectation)(expectation_flat)

    assert float(jnp.max(jnp.abs(fisher - fisher.T))) < 1e-10
    assert float(jnp.linalg.eigvalsh(_symmetrise(fisher)).min()) > 0.0

    natural_gradient = jnp.linalg.solve(fisher, grad_natural)
    assert float(jnp.max(jnp.abs(natural_gradient - grad_expectation))) < 1e-10


# ---------------------------------------------------------------------------
# T4 -- the Fisher information is the covariance of the sufficient statistics
# ---------------------------------------------------------------------------
def test_fisher_equals_sufficient_statistic_covariance() -> None:
    """``F == Cov_q[t(u)]``, built in closed form by Isserlis' theorem."""
    mean, root_covariance = _identity_point()
    natural_flat = _flatten(*natural_from_moments(mean, root_covariance))

    def expectation_of_natural(flat):
        return _flatten(
            *expectation_from_moments(*moments_from_natural(*_unflatten(flat)))
        )

    fisher = jax.jacfwd(expectation_of_natural)(natural_flat)

    covariance = root_covariance @ root_covariance.T
    flat_mean = mean.reshape(-1)
    cross = jnp.einsum("j,ik->ijk", flat_mean, covariance) + jnp.einsum(
        "k,ij->ijk", flat_mean, covariance
    )
    quadratic = (
        jnp.einsum("ik,jl->ijkl", covariance, covariance)
        + jnp.einsum("il,jk->ijkl", covariance, covariance)
        + jnp.einsum("i,k,jl->ijkl", flat_mean, flat_mean, covariance)
        + jnp.einsum("i,l,jk->ijkl", flat_mean, flat_mean, covariance)
        + jnp.einsum("j,k,il->ijkl", flat_mean, flat_mean, covariance)
        + jnp.einsum("j,l,ik->ijkl", flat_mean, flat_mean, covariance)
    )
    cross_flat = cross[:, _ROW, _COL] * _SCALE
    quadratic_flat = (
        quadratic[_ROW, _COL][:, _ROW, _COL] * _SCALE[:, None] * _SCALE[None, :]
    )
    statistic_covariance = jnp.block(
        [[covariance, cross_flat], [cross_flat.T, quadratic_flat]]
    )

    np.testing.assert_allclose(
        np.float64(fisher), np.float64(statistic_covariance), atol=1e-10
    )


# ---------------------------------------------------------------------------
# T7 -- the map_jitter regression guard (written before the headline test)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_cls", _SALIMBENI_FAMILIES, ids=lambda cls: cls.__name__
)
def test_map_jitter_biases_the_conjugate_step(family_cls) -> None:
    r"""``map_jitter`` biases $S$ by $\approx\varepsilon\lVert S\rVert^2$.

    The jitter added to $-2\Theta_2 = S^{-1}$ is a bias, not a rounding effect, so the
    headline exactness bound of :func:`test_natgrad_conjugate_one_step_exact` fails by
    orders of magnitude at ``map_jitter=1e-6``. This test exists to stop anyone
    "tidying" the default to ``q.jitter``.
    """
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    mean, root_covariance = _random_moments(11, _NUM_INDUCING)
    family = family_cls(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root_covariance,
        jitter=jitter,
    )

    stepped, _ = _take_step(family, dataset, 1.0, map_jitter=1e-6)
    _, updated_root = _moments_of(stepped)
    _, optimal_covariance, _ = _conjugate_optimum(family, dataset)

    error = float(jnp.max(jnp.abs(updated_root @ updated_root.T - optimal_covariance)))
    assert error > 1e-8


# ---------------------------------------------------------------------------
# T5 -- the headline conjugate exactness test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_cls", _SALIMBENI_FAMILIES, ids=lambda cls: cls.__name__
)
def test_natgrad_conjugate_one_step_exact(family_cls) -> None:
    r"""One $\gamma=1$ full-batch step reaches the exact optimal $q$."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    mean, root_covariance = _random_moments(11, _NUM_INDUCING)
    family = family_cls(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root_covariance,
        jitter=jitter,
    )

    stepped, _ = _take_step(family, dataset, 1.0)
    updated_mean, updated_root = _moments_of(stepped)
    optimal_mean, optimal_covariance, _ = _conjugate_optimum(family, dataset)

    np.testing.assert_allclose(
        np.float64(updated_mean), np.float64(optimal_mean), atol=1e-10
    )
    np.testing.assert_allclose(
        np.float64(updated_root @ updated_root.T),
        np.float64(optimal_covariance),
        atol=1e-10,
    )

    assert float(elbo(paramax.unwrap(stepped), dataset)) >= float(
        elbo(paramax.unwrap(family), dataset)
    )

    twice_stepped, _ = _take_step(stepped, dataset, 1.0)
    second_mean, _ = _moments_of(twice_stepped)
    np.testing.assert_allclose(
        np.float64(second_mean), np.float64(updated_mean), atol=1e-10
    )


# ---------------------------------------------------------------------------
# T6 -- the whitened and unwhitened optima are the same distribution
# ---------------------------------------------------------------------------
def test_whitened_and_unwhitened_optima_agree() -> None:
    r"""$m^\star=\mu_z+L_z m_w^\star$ and $S^\star=L_z S_w^\star L_z^\top$."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    mean, root_covariance = _random_moments(11, _NUM_INDUCING)
    families = {
        cls.__name__: cls(
            posterior=posterior,
            inducing_inputs=inducing_inputs,
            variational_mean=mean,
            variational_root_covariance=root_covariance,
            jitter=jitter,
        )
        for cls in _SALIMBENI_FAMILIES
    }

    unwhitened = families["VariationalGaussian"]
    whitened = families["WhitenedVariationalGaussian"]
    _, _, root_gram, inducing_mean, _ = _kernel_matrices(unwhitened, dataset.X)

    optimal_mean, optimal_covariance, _ = _conjugate_optimum(unwhitened, dataset)
    whitened_mean, whitened_covariance, _ = _conjugate_optimum(whitened, dataset)

    np.testing.assert_allclose(
        np.float64(inducing_mean + root_gram @ whitened_mean),
        np.float64(optimal_mean),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.float64(root_gram @ whitened_covariance @ root_gram.T),
        np.float64(optimal_covariance),
        atol=1e-10,
    )

    stepped_unwhitened, _ = _take_step(unwhitened, dataset, 1.0)
    stepped_whitened, _ = _take_step(whitened, dataset, 1.0)
    np.testing.assert_allclose(
        np.float64(elbo(paramax.unwrap(stepped_whitened), dataset)),
        np.float64(elbo(paramax.unwrap(stepped_unwhitened), dataset)),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# T8 -- the natural-parameter target is minus half the posterior precision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_cls", _SALIMBENI_FAMILIES, ids=lambda cls: cls.__name__
)
def test_theta2_target_equals_minus_half_precision(family_cls) -> None:
    r"""$\Theta_2^{\rm tgt}=\partial\mathcal L_{\rm data}/\partial S-\tfrac12 K_{zz}^{-1}=-\tfrac12\Lambda$."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    mean, root_covariance = _random_moments(11, _NUM_INDUCING)
    family = family_cls(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root_covariance,
        jitter=jitter,
    )

    data_gradient = _data_term_covariance_gradient(family, dataset)
    gram, _, _, _, _ = _kernel_matrices(family, dataset.X)
    prior_precision = (
        jnp.eye(_NUM_INDUCING)
        if isinstance(family, WhitenedVariationalGaussian)
        else jnp.linalg.inv(gram)
    )
    target = data_gradient - 0.5 * prior_precision

    _, _, precision = _conjugate_optimum(family, dataset)
    np.testing.assert_allclose(
        np.float64(target), np.float64(-0.5 * precision), atol=1e-10
    )


def _data_term_covariance_gradient(family, data):
    r"""$\partial\mathcal L_{\rm data}/\partial S$ by autodiff through GPJax."""
    mean, root_covariance = _moments_of(family)
    expectation = expectation_from_moments(mean, root_covariance)

    def data_term(pair):
        trial_mean, trial_root = moments_from_expectation(*pair)
        trial = eqx.tree_at(
            lambda tree: (tree.variational_mean, tree.variational_root_covariance),
            family,
            (Real(trial_mean), LowerTriangular(trial_root)),
        )
        trial = paramax.unwrap(trial)
        full_size = data.n_total if data.n_total is not None else data.n
        scale = full_size / data.n
        return jnp.sum(variational_expectation(trial, data)) * scale

    return _symmetrise(jax.grad(data_term)(expectation)[1])


# ---------------------------------------------------------------------------
# T9 -- the step is a convex combination in the natural coordinates
# ---------------------------------------------------------------------------
def test_natgrad_step_is_a_convex_combination_in_theta() -> None:
    r"""$\Theta_2^{\rm new}=(1-\gamma)\Theta_2+\gamma\Theta_2^{\rm tgt}$."""
    posterior, dataset, inducing_inputs, jitter = _bernoulli_setup()
    mean, root_covariance = _random_moments(5, _NUM_INDUCING)
    family = VariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root_covariance,
        jitter=jitter,
    )
    natgrad_lr = 0.37

    gram, _, _, _, _ = _kernel_matrices(family, dataset.X)
    target = _data_term_covariance_gradient(family, dataset) - 0.5 * jnp.linalg.inv(
        gram
    )
    _, initial_natural_matrix = natural_from_moments(mean, root_covariance)

    stepped, _ = _take_step(family, dataset, natgrad_lr)
    _, updated_natural_matrix = natural_from_moments(*_moments_of(stepped))

    np.testing.assert_allclose(
        np.float64(updated_natural_matrix),
        np.float64((1.0 - natgrad_lr) * initial_natural_matrix + natgrad_lr * target),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# T10 -- a gamma = 1 step on a mini-batch hits the mini-batch optimum
# ---------------------------------------------------------------------------
def test_natgrad_minibatch_hits_minibatch_optimum() -> None:
    """The ``N/B`` factor lives inside the loss, not in the step size."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    batch_size = 8
    batch = Dataset(X=dataset.X[:batch_size], y=dataset.y[:batch_size])

    mean, root_covariance = _random_moments(11, _NUM_INDUCING)
    family = VariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root_covariance,
        jitter=jitter,
    )

    stepped, _ = _take_step(family, batch, 1.0)
    updated_mean, updated_root = _moments_of(stepped)
    optimal_mean, optimal_covariance, _ = _conjugate_optimum(family, batch)

    np.testing.assert_allclose(
        np.float64(updated_mean), np.float64(optimal_mean), atol=1e-10
    )
    np.testing.assert_allclose(
        np.float64(updated_root @ updated_root.T),
        np.float64(optimal_covariance),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# T11 -- monotone ELBO on a non-conjugate model
# ---------------------------------------------------------------------------
def test_natgrad_monotone_elbo_bernoulli() -> None:
    """Fifty full-batch steps at a small step size never decrease the ELBO."""
    posterior, dataset, inducing_inputs, jitter = _bernoulli_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )

    trace = [float(elbo(paramax.unwrap(family), dataset))]
    for _ in range(50):
        family, _ = _take_step(family, dataset, 0.1)
        trace.append(float(elbo(paramax.unwrap(family), dataset)))

    assert bool(jnp.all(jnp.diff(jnp.asarray(trace)) >= -1e-9))


# ---------------------------------------------------------------------------
# T12 -- the step-size backoff
# ---------------------------------------------------------------------------
def _expectation_gradient(family, data):
    r"""$\partial\ell/\partial\boldsymbol\eta$ of the negative ELBO, symmetrised."""
    mean, root_covariance = _moments_of(family)
    expectation = expectation_from_moments(mean, root_covariance)

    def loss_of_expectation(pair):
        trial_mean, trial_root = moments_from_expectation(*pair)
        trial = eqx.tree_at(
            lambda tree: (tree.variational_mean, tree.variational_root_covariance),
            family,
            (Real(trial_mean), LowerTriangular(trial_root)),
        )
        return _negative_elbo(paramax.unwrap(trial), data)

    gradient = jax.grad(loss_of_expectation)(expectation)
    return gradient[0], _symmetrise(gradient[1])


def test_natgrad_backoff_recovers_from_large_step() -> None:
    """A wildly over-sized step is shrunk until it lands back in the cone.

    The initial covariance is deliberately tight (``S_0 = 0.01 I``) so that the target
    precision is *smaller* than the current one. Extrapolating past the target then
    genuinely leaves the negative-definite cone, which the default zero-mean,
    identity-root initialisation does not do at any step size.
    """
    posterior, dataset, inducing_inputs, jitter = _bernoulli_setup()
    family = VariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_root_covariance=0.1 * jnp.eye(_NUM_INDUCING),
        jitter=jitter,
    )
    natgrad_lr = 100.0

    # The un-shrunk step must genuinely fail, or the test proves nothing.
    initial_natural = natural_from_moments(*_moments_of(family))
    gradient = _expectation_gradient(family, dataset)
    unshrunk = moments_from_natural(
        initial_natural[0] - natgrad_lr * gradient[0],
        initial_natural[1] - natgrad_lr * gradient[1],
    )
    assert not bool(jnp.all(jnp.isfinite(unshrunk[1])))

    # With no shrink attempts the NaN reaches the caller ...
    exhausted, _ = _take_step(family, dataset, natgrad_lr, max_backoff=0)
    assert not bool(jnp.all(jnp.isfinite(_moments_of(exhausted)[1])))

    # ... and with them, the step is shrunk back into the cone.
    stepped, loss_value = _take_step(family, dataset, natgrad_lr, max_backoff=20)
    updated_mean, updated_root = _moments_of(stepped)

    assert bool(jnp.all(jnp.isfinite(updated_mean)))
    assert bool(jnp.all(jnp.isfinite(updated_root)))
    assert bool(jnp.isfinite(loss_value))
    assert bool(jnp.isfinite(elbo(paramax.unwrap(stepped), dataset)))

    # Recover the step size actually taken: Theta_2^new = Theta_2 - s * dl/dH_2.
    updated_natural_matrix = natural_from_moments(updated_mean, updated_root)[1]
    difference = initial_natural[1] - updated_natural_matrix
    accepted_step_size = float(
        jnp.sum(difference * gradient[1]) / jnp.sum(gradient[1] * gradient[1])
    )
    assert accepted_step_size < natgrad_lr
    np.testing.assert_allclose(
        accepted_step_size,
        natgrad_lr * 0.5 ** round(np.log2(natgrad_lr / accepted_step_size)),
        rtol=1e-8,
    )


def test_first_valid_trial_selects_the_largest_admissible_step() -> None:
    """``_first_valid_trial`` returns exactly the first finite trial of the vmap."""
    # S = I, so the precision -2 * Theta2 is the identity and a step of size s in the
    # direction -I leaves the cone exactly when s >= 1/2.
    mean = jr.normal(jr.key(2), (4, 1))
    natural_vector, natural_matrix = natural_from_moments(mean, jnp.eye(4))
    gradient_vector = jnp.zeros_like(natural_vector)
    gradient_matrix = -jnp.eye(4)

    natgrad_lr, backoff, max_backoff = 8.0, 0.5, 5
    accepted = None
    accepted_index = None
    for index in range(max_backoff + 1):
        step_size = natgrad_lr * backoff**index
        candidate = moments_from_natural(
            natural_vector - step_size * gradient_vector,
            natural_matrix - step_size * gradient_matrix,
        )
        if bool(jnp.all(jnp.isfinite(candidate[1]))):
            accepted = candidate
            accepted_index = index
            break

    assert accepted is not None, "the reference sweep found no admissible step"
    assert accepted_index > 0, "the backoff was never exercised"
    selected = _first_valid_trial(
        natural_vector,
        natural_matrix,
        gradient_vector,
        gradient_matrix,
        jnp.asarray(natgrad_lr),
        0.0,
        backoff,
        max_backoff,
    )
    np.testing.assert_allclose(np.float64(selected[0]), np.float64(accepted[0]))
    np.testing.assert_allclose(np.float64(selected[1]), np.float64(accepted[1]))

    # With no shrink attempts left, the NaN propagates rather than being hidden.
    exhausted = _first_valid_trial(
        natural_vector,
        natural_matrix,
        gradient_vector,
        gradient_matrix,
        jnp.asarray(natgrad_lr),
        0.0,
        backoff,
        0,
    )
    assert not bool(jnp.all(jnp.isfinite(exhausted[1])))


# ---------------------------------------------------------------------------
# T13 -- jit, scan and carry-structure compatibility
# ---------------------------------------------------------------------------
def test_natgrad_step_is_jit_and_scan_compatible() -> None:
    """The step compiles, scans, and leaves the carry treedef untouched."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    variational, hyper = partition_variational(family)

    def step(partition, natgrad_lr):
        return natural_gradient_step(
            partition, hyper, dataset, _negative_elbo, natgrad_lr
        )

    eager, eager_loss = step(variational, jnp.asarray(0.5))
    compiled, compiled_loss = eqx.filter_jit(step)(variational, jnp.asarray(0.5))

    np.testing.assert_allclose(
        np.float64(_val(compiled.variational_mean)),
        np.float64(_val(eager.variational_mean)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.float64(compiled_loss), np.float64(eager_loss), atol=1e-12
    )

    def body(carry, _):
        return step(carry, jnp.asarray(0.5))

    scanned, history = jax.lax.scan(body, variational, None, length=3)
    assert history.shape == (3,)
    assert jtu.tree_structure(scanned) == jtu.tree_structure(variational)


@pytest.mark.filterwarnings("ignore:X is not of type float64")
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_natgrad_step_is_dtype_preserving() -> None:
    """A float32 family stays float32, so the ``lax.scan`` carry type is stable.

    The trial ladder is the trap: under ``jax_enable_x64`` the exponent
    ``jnp.arange(K + 1)`` is ``int64``, so an uncast ``backoff ** arange`` promotes the
    whole update to float64 and ``lax.scan`` rejects the carry.
    """
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    cast = lambda leaf: jnp.asarray(leaf, dtype=jnp.float32)
    family = jtu.tree_map(cast, family)
    dataset = Dataset(X=cast(dataset.X), y=cast(dataset.y))

    stepped, loss_value = _take_step(family, dataset, jnp.float32(0.1))
    updated_mean, updated_root = _moments_of(stepped)

    # The coordinates are the scan carry and must keep their dtype. The loss is a scan
    # output, so its dtype only has to be stable across iterations, and GPJax's own
    # objectives promote it to the default float type regardless.
    assert updated_mean.dtype == jnp.float32
    assert updated_root.dtype == jnp.float32
    assert bool(jnp.isfinite(loss_value))


# ---------------------------------------------------------------------------
# T14 -- graph-structured inducing inputs
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:X is not of type float64")
def test_natgrad_step_supports_graph_variational_gaussian() -> None:
    """The graph family steps without error and keeps its int64 inducing inputs.

    The objective is the prior KL rather than the ELBO: ``elbo`` routes through
    ``variational_expectation``, whose per-point ``vmap`` trips a pre-existing
    jaxtyping error in ``EigenKernelComputation._cross_covariance`` for a single
    graph node. That is orthogonal to natural gradients, so the smoke test avoids it.
    """
    graph = nx.barbell_graph(10, 0)
    laplacian = nx.laplacian_matrix(graph).toarray()
    num_nodes = graph.number_of_nodes()

    kernel = gpjax.kernels.GraphKernel(
        laplacian=laplacian, lengthscale=2.3, variance=3.2, smoothness=6.1
    )
    posterior = (
        gpjax.gps.Prior(mean_function=gpjax.mean_functions.Constant(), kernel=kernel)
        * gpjax.likelihoods.Bernoulli()
    )

    inducing_inputs = jnp.arange(0, num_nodes, 4).reshape(-1, 1).astype(jnp.int64)
    inputs = jnp.arange(num_nodes).reshape(-1, 1).astype(jnp.int64)
    outputs = jr.bernoulli(jr.key(3), 0.5, (num_nodes, 1)).astype(jnp.float64)
    dataset = Dataset(X=inputs, y=outputs)

    family = GraphVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs
    )
    stepped, loss_value = _take_step(
        family, dataset, 0.1, objective=lambda tree, _: tree.prior_kl()
    )

    assert bool(jnp.isfinite(loss_value))
    assert _val(stepped.inducing_inputs).dtype == jnp.int64
    np.testing.assert_array_equal(
        np.asarray(_val(stepped.inducing_inputs)), np.asarray(inducing_inputs)
    )


# ---------------------------------------------------------------------------
# T15 / T16 / T17 -- partitioning and guards
# ---------------------------------------------------------------------------
def _leaf_paths(tree):
    return {jtu.keystr(path) for path, _ in jtu.tree_flatten_with_path(tree)[0]}


def test_partition_variational_leaf_sets() -> None:
    """The split is exactly the two exponential-family coordinates."""
    posterior, _, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    variational, hyper = partition_variational(family)

    assert _leaf_paths(variational) == {
        ".variational_mean.value",
        ".variational_root_covariance._flat",
    }
    assert _leaf_paths(hyper) == {
        ".posterior.prior.kernel.lengthscale._unconstrained",
        ".posterior.prior.kernel.variance._unconstrained",
        ".posterior.prior.mean_function.constant",
        ".posterior.likelihood.obs_stddev._unconstrained",
        ".inducing_inputs.value",
    }

    recombined = eqx.combine(variational, hyper)
    assert jtu.tree_structure(recombined) == jtu.tree_structure(family)
    for original, restored in zip(
        jtu.tree_leaves(family), jtu.tree_leaves(recombined), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))


def test_partition_variational_rejects_unknown_family() -> None:
    """Families without exponential-family coordinates raise ``NotImplementedError``."""
    posterior, _, inducing_inputs, jitter = _conjugate_setup()
    family = CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    with pytest.raises(NotImplementedError, match="CollapsedVariationalGaussian"):
        partition_variational(family)


def test_natgrad_step_rejects_non_trainable_coordinates() -> None:
    """A frozen coordinate makes a natural-gradient step meaningless."""
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    family = eqx.tree_at(
        lambda tree: tree.variational_mean,
        family,
        paramax.non_trainable(family.variational_mean),
    )
    with pytest.raises(ValueError, match="variational_mean"):
        _take_step(family, dataset, 0.1)


# ---------------------------------------------------------------------------
# T18 / T19 -- factorisation budget and the absence of explicit inverses
# ---------------------------------------------------------------------------
def _count_cholesky_calls(monkeypatch, thunk) -> int:
    """Count dense Cholesky factorisations performed while evaluating ``thunk``."""
    counts = {"dense_cholesky": 0}
    original_dense_cholesky = jnp.linalg.cholesky

    def counting_dense_cholesky(matrix):
        counts["dense_cholesky"] += 1
        return original_dense_cholesky(matrix)

    monkeypatch.setattr(jnp.linalg, "cholesky", counting_dense_cholesky)
    thunk()
    return counts["dense_cholesky"]


@pytest.mark.parametrize("max_backoff", [0, 5, 20])
def test_natgrad_step_cholesky_budget(monkeypatch, max_backoff: int) -> None:
    """One step traces exactly six dense Choleskys, whatever ``max_backoff`` is.

    The six are: one in ``moments_from_expectation`` inside the loss, two in the
    ``elbo`` forward pass, one in the vmapped admissibility probe, and two in the
    single ``moments_from_natural`` at the accepted step size. ``jax.vmap`` traces its
    body once, so this Python-level count is independent of the number of trials --
    the width is pinned by :func:`test_natgrad_step_replicates_only_the_probe`.
    A regression to the ``xi(theta(eta))`` round trip that the design rejects would
    show up here as eight.
    """
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )

    count = _count_cholesky_calls(
        monkeypatch,
        lambda: _take_step(family, dataset, 1.0, max_backoff=max_backoff),
    )
    assert count == 6


def _cholesky_call_shapes(lowered_text: str) -> list[tuple[int, ...]]:
    """Leading-argument shapes of every ``cholesky`` call site in lowered StableHLO."""
    shapes = []
    for match in re.finditer(
        r"= call @cholesky\w*\([^)]*\) : \(tensor<([^>]*)>", lowered_text
    ):
        dimensions = match.group(1).split("x")[:-1]
        shapes.append(tuple(int(dimension) for dimension in dimensions))
    return shapes


@pytest.mark.parametrize("max_backoff", [0, 5])
def test_natgrad_step_replicates_only_the_probe(max_backoff: int) -> None:
    """Exactly one factorisation is replicated across the ``K + 1`` trial steps.

    Counting at the Python level cannot see the ``vmap`` width, so the budget is read
    off the lowered computation instead: the batched Cholesky must have a leading
    dimension of ``max_backoff + 1`` and every other Cholesky must be unbatched.
    """
    posterior, dataset, inducing_inputs, jitter = _conjugate_setup()
    family = VariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    variational, hyper = partition_variational(family)

    def step(partition):
        return natural_gradient_step(
            partition,
            hyper,
            dataset,
            _negative_elbo,
            jnp.asarray(1.0),
            max_backoff=max_backoff,
        )

    shapes = _cholesky_call_shapes(eqx.filter_jit(step).lower(variational).as_text())

    assert shapes, "no cholesky call sites found -- has the JAX lowering changed?"
    batched = [shape for shape in shapes if len(shape) > 2]
    unbatched = [shape for shape in shapes if len(shape) == 2]
    assert batched == [(max_backoff + 1, _NUM_INDUCING, _NUM_INDUCING)]
    assert unbatched == [(_NUM_INDUCING, _NUM_INDUCING)] * len(unbatched)
    assert len(shapes) == 7


def test_natural_gradients_module_has_no_explicit_inverse() -> None:
    """The coordinate maps must use triangular solves, never ``jnp.linalg.inv``."""
    source = Path(gpjax.natural_gradients.__file__).read_text()
    assert "jnp.linalg.inv" not in source


# ---------------------------------------------------------------------------
# T20 -- parameterisation invariance of the natural-gradient trace
# ---------------------------------------------------------------------------
def test_whitened_and_unwhitened_traces_agree() -> None:
    """Matched initialisations give the same ELBO trace in both parameterisations."""
    posterior, dataset, inducing_inputs, jitter = _bernoulli_setup()
    whitened_mean, whitened_root = _random_moments(13, _NUM_INDUCING)

    whitened = WhitenedVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=whitened_mean,
        variational_root_covariance=whitened_root,
        jitter=jitter,
    )
    _, _, root_gram, inducing_mean, _ = _kernel_matrices(whitened, dataset.X)
    unwhitened = VariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=inducing_mean + root_gram @ whitened_mean,
        variational_root_covariance=jnp.linalg.cholesky(
            root_gram @ whitened_root @ whitened_root.T @ root_gram.T
        ),
        jitter=jitter,
    )

    whitened_trace = []
    unwhitened_trace = []
    for _ in range(6):
        whitened, _ = _take_step(whitened, dataset, 0.3)
        unwhitened, _ = _take_step(unwhitened, dataset, 0.3)
        whitened_trace.append(float(elbo(paramax.unwrap(whitened), dataset)))
        unwhitened_trace.append(float(elbo(paramax.unwrap(unwhitened), dataset)))

    np.testing.assert_allclose(
        np.asarray(whitened_trace), np.asarray(unwhitened_trace), atol=1e-9
    )
