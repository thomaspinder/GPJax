# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

from collections.abc import Callable

import equinox as eqx
import gpjax as gpx
import gpjax.distributions
from gpjax.gps import JointModel
import gpjax.linalg.utils
from gpjax.parameters import (
    LowerTriangular,
    Real,
    _val,
)
import gpjax.variational_families
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
    GraphVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Array,
    Float,
)
import networkx as nx
import numpy as np
import numpyro.distributions as npd
from numpyro.distributions import Distribution as NumpyroDistribution
import pytest

from tests._dual_helpers import (
    DUAL_JITTER,
    build_dual,
    matched_variational_gaussian as _matched_variational_gaussian,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)


def test_abstract_variational_family():
    # Test that the abstract class cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractVariationalFamily()

    # Create a dummy variational family class with abstract methods implemented.
    class DummyPosterior:
        @property
        def __class__(self) -> type:
            return JointModel

    class DummyVariationalFamily(AbstractVariationalFamily):
        def predict(self, x: Float[Array, "N D"]) -> npd.MultivariateNormal:
            return npd.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(x.shape[1]))

    # Test that the dummy variational family can be instantiated.
    dummy_variational_family = DummyVariationalFamily(posterior=DummyPosterior())
    assert isinstance(dummy_variational_family, AbstractVariationalFamily)


# Functions to test variational family parameter shapes upon initialisation.
def vector_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a vector with n_inducing rows and 1 column."""
    return (n_inducing, 1)


def matrix_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a matrix with n_inducing rows and 1 column."""
    return (n_inducing, n_inducing)


# Functions to test variational parameter values upon initialisation.
def vector_val(val: float) -> Callable[[int], Float[Array, "n_inducing 1"]]:
    """Vector of shape (n_inducing, 1) filled with val."""

    def vector_val_fn(n_inducing: int):
        return val * jnp.ones(vector_shape(n_inducing))

    return vector_val_fn


def diag_matrix_val(
    val: float,
) -> Callable[[int], Float[Array, "n_inducing n_inducing"]]:
    """Diagonal matrix of shape (n_inducing, n_inducing) filled with val."""

    def diag_matrix_fn(n_inducing: int) -> Float[Array, "n_inducing n_inducing"]:
        return jnp.eye(n_inducing) * val

    return diag_matrix_fn


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        VariationalGaussian,
        WhitenedVariationalGaussian,
        DualVariationalGaussian,
    ],
)
def test_variational_gaussians(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    # Initialise variational family:
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )
    likelihood = gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)

    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)

    posterior = prior * likelihood
    q = variational_family(posterior=posterior, inducing_inputs=inducing_inputs)

    # Test init:
    assert q.num_inducing == n_inducing
    assert isinstance(q, AbstractVariationalFamily)

    if isinstance(q, DualVariationalGaussian):
        # The dual family stores sites, not moments, and both default to zero so that
        # q(u) = p(u) at initialisation.
        assert _val(q.dual_vector).shape == vector_shape(n_inducing)
        assert _val(q.dual_matrix).shape == matrix_shape(n_inducing)
        assert (_val(q.dual_vector) == 0.0).all()
        assert (_val(q.dual_matrix) == 0.0).all()
    else:
        assert q.variational_mean.unwrap().shape == vector_shape(n_inducing)
        assert q.variational_root_covariance.unwrap().shape == matrix_shape(n_inducing)
        assert (q.variational_mean.unwrap() == vector_val(0.0)(n_inducing)).all()
        assert (
            q.variational_root_covariance.unwrap() == diag_matrix_val(1.0)(n_inducing)
        ).all()

    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    # The dual family initialises exactly at q(u) = p(u), where the KL is zero up to
    # the round-off of tr(R^{-1} Kzz) - M; the moment families start strictly inside.
    assert kl >= (-1e-10 if isinstance(q, DualVariationalGaussian) else 0.0)

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize(
    "removed_name",
    ["NaturalVariationalGaussian", "ExpectationVariationalGaussian"],
)
def test_removed_families_are_gone(removed_name: str) -> None:
    """The natural/expectation parameterisations were superseded by `fit_natgrads`.

    They were parameterisation-only classes with no optimiser attached; natural-gradient
    steps are now taken directly on `VariationalGaussian` and
    `WhitenedVariationalGaussian`. This guards against them creeping back in.
    """
    assert not hasattr(gpjax.variational_families, removed_name)
    assert removed_name not in gpjax.variational_families.__all__


def test_psd_helper_is_gone() -> None:
    """`_psd`'s only callers lived inside the two deleted classes.

    Checked separately from the class names because `_psd` was private and never
    exported, so the `__all__` assertion above would be vacuous for it. The guard is
    against the dead helper returning alongside the classes, not a reservation of the
    name for all time.
    """
    assert not hasattr(gpjax.variational_families, "_psd")


@pytest.mark.parametrize("n_test", [10, 20])
@pytest.mark.parametrize("n_inducing", [10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        GraphVariationalGaussian,
    ],
)
def test_graph_variational_gaussian(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    G = nx.barbell_graph(100, 0)
    L = nx.laplacian_matrix(G).toarray()

    kernel = gpx.kernels.GraphKernel(
        laplacian=L,
        lengthscale=2.3,
        variance=3.2,
        smoothness=6.1,
    )
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Bernoulli()

    inducing_inputs = jnp.array(
        np.random.randint(low=1, high=100, size=(n_inducing, 1))
    ).astype(jnp.int64)

    test_inputs = jnp.array(np.random.randint(low=0, high=1, size=(n_test, 1))).astype(
        jnp.int64
    )

    posterior = prior * likelihood
    q = variational_family(posterior=posterior, inducing_inputs=inducing_inputs)
    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    assert kl >= 0.0

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_datapoints", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_collapsed_variational_gaussian(
    n_test: int, n_inducing: int, n_datapoints: int, point_dim: int
) -> None:
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.key(123), shape=x.shape) * 0.1
    x = jnp.hstack([x] * point_dim)
    D = gpx.Dataset(X=x, y=y)

    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    test_inputs = jnp.hstack([test_inputs] * point_dim)

    posterior = prior * gpx.likelihoods.Gaussian()

    variational_family = CollapsedVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
    )

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        CollapsedVariationalGaussian(
            posterior=prior * gpx.likelihoods.Bernoulli(),
            inducing_inputs=inducing_inputs,
        )

    # Test init
    assert variational_family.num_inducing == n_inducing
    assert (variational_family.inducing_inputs.unwrap() == inducing_inputs).all()
    assert variational_family.posterior.likelihood.obs_stddev.unwrap() == 1.0

    # Test predictions
    predictive_dist = variational_family(test_inputs, D)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


# ---------------------------------------------------------------------------
# Closed-form prior KL (issue #665)
#
# `WhitenedVariationalGaussian.prior_kl` and `VariationalGaussian.prior_kl`
# used to densify `S = sqrt sqrt^T` and hand it to the generic Gaussian KL,
# which then re-factorised a matrix whose Cholesky factor is already stored in
# `variational_root_covariance`. The tests below pin the numerical values and
# gradients of the old implementation and assert the number of Cholesky
# factorisations the new implementation is allowed to perform.
# ---------------------------------------------------------------------------


def _kl_variational_mean(num_inducing: int) -> Float[Array, "N 1"]:
    """Deterministic, non-trivial variational mean."""
    index = jnp.arange(num_inducing, dtype=jnp.float64)
    return (0.3 * jnp.cos(1.7 * index) - 0.15 * index).reshape(-1, 1)


def _kl_variational_root(num_inducing: int) -> Float[Array, "N N"]:
    """Deterministic, non-trivial lower-triangular root with positive diagonal."""
    row = jnp.arange(num_inducing, dtype=jnp.float64)[:, None]
    col = jnp.arange(num_inducing, dtype=jnp.float64)[None, :]
    off_diagonal = 0.2 * jnp.sin(0.9 * row + 1.3 * col)
    diagonal = 0.5 + 0.4 * jnp.cos(0.6 * jnp.arange(num_inducing, dtype=jnp.float64))
    return jnp.tril(off_diagonal, -1) + jnp.diag(diagonal)


def _build_kl_family(
    family: type[VariationalGaussian], num_inducing: int
) -> VariationalGaussian:
    """Build a fully deterministic variational family for KL regression tests."""
    kernel = gpx.kernels.RBF(lengthscale=jnp.array(0.7), variance=jnp.array(1.3))
    mean_function = gpx.mean_functions.Constant(jnp.array([0.4]))
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function)
    posterior = prior * gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)
    return family(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=_kl_variational_mean(num_inducing),
        variational_root_covariance=_kl_variational_root(num_inducing),
    )


# KL values recorded from the pre-#665 implementation (densify + generic KL),
# in float64. These are hard-coded literals so that the test is a genuine
# regression guard rather than a comparison against the code under test.
_RECORDED_PRIOR_KL = {
    ("WhitenedVariationalGaussian", 1): 0.055360515657826292,
    ("WhitenedVariationalGaussian", 3): 0.4558012885085363,
    ("WhitenedVariationalGaussian", 5): 2.2228894799193393,
    ("WhitenedVariationalGaussian", 12): 12.540706158684808,
    ("VariationalGaussian", 1): 0.051927405288060224,
    ("VariationalGaussian", 3): 0.89833938800292357,
    ("VariationalGaussian", 5): 3.0465143546546192,
    ("VariationalGaussian", 12): 35.666594348171607,
}

# Gradients recorded from the pre-#665 implementation at ``num_inducing=3``,
# taken with respect to the *unconstrained* parameters that `fit` optimises.
_RECORDED_PRIOR_KL_GRADS = {
    "WhitenedVariationalGaussian": {
        "variational_mean": np.array(
            [[0.3], [-0.18865334828865737], [-0.5900394577738384]]
        ),
        "variational_root_covariance": np.array(
            [
                0.15666538192549667,
                0.19476952617563908,
                0.00831613248665811,
                -0.12527973849920676,
                -0.21121593177991954,
                -0.430429665953372,
            ]
        ),
        "inducing_inputs": np.zeros((3, 1)),
    },
    "VariationalGaussian": {
        "variational_mean": np.array(
            [[-0.07687652189052713], [-0.452723814014418], [-0.7615217319894719]]
        ),
        "variational_root_covariance": np.array(
            [
                0.12042525327023475,
                0.14981022922180554,
                0.00633143799283199,
                -0.24853831088039383,
                -0.31926350826875094,
                -0.5011713127475177,
            ]
        ),
        "inducing_inputs": np.array(
            [
                [-9.6628829749081168e-05],
                [-2.0328302385444184e-04],
                [2.9991185360352303e-04],
            ]
        ),
    },
}


def _textbook_gaussian_kl(mean_q, cov_q, mean_p, cov_p):
    """KL[q || p] for two multivariate Gaussians, written from the textbook.

    Deliberately independent of any GPJax linear-algebra helper so that this
    reference cannot drift with changes to ``gpjax.distributions``.
    """
    dim = mean_q.shape[0]
    trace = jnp.trace(jnp.linalg.solve(cov_p, cov_q))
    diff = mean_p - mean_q
    mahalanobis = diff @ jnp.linalg.solve(cov_p, diff)
    _, log_det_p = jnp.linalg.slogdet(cov_p)
    _, log_det_q = jnp.linalg.slogdet(cov_q)
    return 0.5 * (trace + mahalanobis - dim + log_det_p - log_det_q)


def _reference_prior_kl(family):
    """Reference prior KL built from the dense covariance matrices."""
    variational_mean = _val(family.variational_mean).reshape(-1)
    variational_sqrt = _val(family.variational_root_covariance)
    cov_q = variational_sqrt @ variational_sqrt.T
    num_inducing = variational_sqrt.shape[-1]

    if isinstance(family, WhitenedVariationalGaussian):
        mean_p = jnp.zeros_like(variational_mean)
        cov_p = jnp.eye(num_inducing, dtype=variational_sqrt.dtype)
    else:
        inducing_inputs = _val(family.inducing_inputs)
        mean_p = family.posterior.prior.mean_function(inducing_inputs).reshape(-1)
        cov_p = family.posterior.prior.kernel.gram(inducing_inputs).as_matrix()
        cov_p = cov_p + jnp.eye(num_inducing, dtype=cov_p.dtype) * family.jitter

    return _textbook_gaussian_kl(variational_mean, cov_q, mean_p, cov_p)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [1, 3, 5, 12])
def test_prior_kl_matches_recorded_values(family, num_inducing):
    """The closed form must reproduce the pre-#665 KL values."""
    q = _build_kl_family(family, num_inducing)
    expected = _RECORDED_PRIOR_KL[(family.__name__, num_inducing)]
    np.testing.assert_allclose(np.float64(q.prior_kl()), expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [1, 2, 3, 5, 12])
def test_prior_kl_matches_textbook_reference(family, num_inducing):
    """The closed form must agree with an independent dense-matrix reference."""
    q = _build_kl_family(family, num_inducing)
    np.testing.assert_allclose(
        np.float64(q.prior_kl()),
        np.float64(_reference_prior_kl(q)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
def test_prior_kl_gradients_match_recorded_values(family):
    """Gradients must match those of the pre-#665 implementation."""
    q = _build_kl_family(family, 3)
    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    recorded = _RECORDED_PRIOR_KL_GRADS[family.__name__]

    np.testing.assert_allclose(
        np.asarray(grads.variational_mean.value),
        recorded["variational_mean"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads.variational_root_covariance._flat),
        recorded["variational_root_covariance"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads.inducing_inputs.value),
        recorded["inducing_inputs"],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [2, 5, 12])
def test_prior_kl_gradients_match_textbook_reference(family, num_inducing):
    """Gradients must agree with an independent dense-matrix reference."""
    q = _build_kl_family(family, num_inducing)
    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    reference_grads = eqx.filter_grad(_reference_prior_kl)(q)

    np.testing.assert_allclose(
        np.asarray(grads.variational_mean.value),
        np.asarray(reference_grads.variational_mean.value),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(grads.variational_root_covariance._flat),
        np.asarray(reference_grads.variational_root_covariance._flat),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(grads.inducing_inputs.value),
        np.asarray(reference_grads.inducing_inputs.value),
        rtol=1e-9,
        atol=1e-11,
    )


def _count_cholesky_calls(monkeypatch, thunk) -> dict[str, int]:
    """Count Cholesky factorisations performed while evaluating ``thunk``."""
    counts = {"cholesky_factor": 0, "dense_cholesky": 0}
    original_cholesky_factor = gpjax.linalg.utils.cholesky_factor
    original_dense_cholesky = jnp.linalg.cholesky

    def counting_cholesky_factor(operator):
        counts["cholesky_factor"] += 1
        return original_cholesky_factor(operator)

    def counting_dense_cholesky(matrix):
        counts["dense_cholesky"] += 1
        return original_dense_cholesky(matrix)

    # ``gpjax.variational_families`` is deliberately absent: its only
    # ``cholesky_factor`` call site lived in the removed
    # ``ExpectationVariationalGaussian``, so the module no longer imports the name.
    for module in (gpjax.linalg.utils, gpjax.distributions):
        monkeypatch.setattr(module, "cholesky_factor", counting_cholesky_factor)
    monkeypatch.setattr(jnp.linalg, "cholesky", counting_dense_cholesky)

    thunk()
    return counts


@pytest.mark.parametrize(
    ("family", "expected_dense_cholesky"),
    [(WhitenedVariationalGaussian, 0), (VariationalGaussian, 1)],
    ids=["WhitenedVariationalGaussian", "VariationalGaussian"],
)
def test_prior_kl_factorisation_count(monkeypatch, family, expected_dense_cholesky):
    """The KL must not re-factorise a matrix whose root it already holds.

    Before #665 the counts were two dense Choleskys for the whitened family and
    four for ``VariationalGaussian``. The whitened KL needs none;
    ``VariationalGaussian`` needs exactly one, of ``Kzz``.
    """
    q = _build_kl_family(family, 4)
    counts = _count_cholesky_calls(monkeypatch, q.prior_kl)

    assert counts["dense_cholesky"] == expected_dense_cholesky
    assert counts["cholesky_factor"] == 0


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
def test_prior_kl_is_jit_grad_and_vmap_compatible(family):
    q = _build_kl_family(family, 4)
    eager = q.prior_kl()

    jitted = eqx.filter_jit(lambda model: model.prior_kl())
    np.testing.assert_allclose(np.float64(jitted(q)), np.float64(eager), rtol=1e-12)

    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    assert jnp.all(jnp.isfinite(grads.variational_mean.value))
    assert jnp.all(jnp.isfinite(grads.variational_root_covariance._flat))

    def kl_from_mean(mean_value):
        model = eqx.tree_at(lambda t: t.variational_mean.value, q, mean_value)
        return model.prior_kl()

    zero_mean = jnp.zeros_like(_val(q.variational_mean))
    batch = jnp.stack(
        [_val(q.variational_mean), _val(q.variational_mean) + 0.5, zero_mean]
    )
    batched = jax.vmap(kl_from_mean)(batch)

    assert batched.shape == (3,)
    np.testing.assert_allclose(
        np.float64(batched[0]), np.float64(eager), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.float64(batched[2]),
        np.float64(kl_from_mean(zero_mean)),
        rtol=1e-12,
        atol=1e-14,
    )


def test_whitened_prior_kl_is_zero_when_q_equals_the_prior():
    """KL[N(0, I) || N(0, I)] must be zero."""
    q = _build_kl_family(WhitenedVariationalGaussian, 6)
    q = eqx.tree_at(
        lambda t: (t.variational_mean, t.variational_root_covariance),
        q,
        (Real(jnp.zeros((6, 1))), LowerTriangular(jnp.eye(6))),
    )
    np.testing.assert_allclose(np.float64(q.prior_kl()), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prior_kl_is_non_negative(family, seed):
    num_inducing = 6
    key_mean, key_root = jr.split(jr.key(seed))
    root = jnp.tril(jr.normal(key_root, (num_inducing, num_inducing)) * 0.2, -1)
    root = root + jnp.diag(0.3 + jnp.abs(jr.normal(key_root, (num_inducing,))))

    q = _build_kl_family(family, num_inducing)
    q = eqx.tree_at(
        lambda t: (t.variational_mean, t.variational_root_covariance),
        q,
        (Real(jr.normal(key_mean, (num_inducing, 1))), LowerTriangular(root)),
    )
    assert q.prior_kl() >= 0.0


# ---------------------------------------------------------------------------
# DualVariationalGaussian -- the t-SVGP site parameterisation (arXiv:2111.03412)
# ---------------------------------------------------------------------------
_DUAL_JITTER = DUAL_JITTER


def _build_dual_family(
    num_inducing: int,
    seed: int | None = None,
    jitter: float = _DUAL_JITTER,
    variance: float = 1.3,
) -> DualVariationalGaussian:
    """Build a dual family, optionally at random positive semi-definite sites."""
    kernel = gpx.kernels.RBF(
        lengthscale=jnp.array(0.7), variance=jnp.array(float(variance))
    )
    mean_function = gpx.mean_functions.Constant(jnp.array([0.4]))
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function)
    posterior = prior * gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)

    return build_dual(posterior, inducing_inputs, jitter=jitter, seed=seed)


@pytest.mark.parametrize("num_inducing", [1, 4, 9])
def test_dual_prior_kl_zero_at_initialisation(num_inducing: int) -> None:
    """Zero sites mean q(u) = p(u), so R = Kzz and the KL vanishes."""
    q = _build_dual_family(num_inducing)
    np.testing.assert_allclose(np.float64(q.prior_kl()), 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", list(range(10)))
def test_dual_prior_kl_non_negative(seed: int) -> None:
    q = _build_dual_family(6, seed=seed)
    assert q.prior_kl() >= -1e-12


@pytest.mark.parametrize("n_test", [1, 7])
@pytest.mark.parametrize("seed", [0, 3])
def test_dual_predict_matches_variational_gaussian_at_matched_moments(
    n_test: int, seed: int
) -> None:
    """The dual predictive must agree with the moment family it is equivalent to."""
    q_dual = _build_dual_family(5, seed=seed)
    q_moment = _matched_variational_gaussian(q_dual)
    test_inputs = jnp.linspace(-3.0, 3.0, n_test).reshape(-1, 1)

    dual_dist = q_dual.predict(test_inputs)
    moment_dist = q_moment.predict(test_inputs)

    np.testing.assert_allclose(
        np.asarray(dual_dist.mean),
        np.asarray(moment_dist.mean),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(dual_dist.covariance()),
        np.asarray(moment_dist.covariance()),
        rtol=1e-9,
        atol=1e-10,
    )


def test_dual_prior_kl_is_jit_grad_and_vmap_compatible() -> None:
    q = _build_dual_family(4, seed=1)
    eager = q.prior_kl()

    jitted = eqx.filter_jit(lambda model: model.prior_kl())
    np.testing.assert_allclose(np.float64(jitted(q)), np.float64(eager), rtol=1e-12)

    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    assert jnp.all(jnp.isfinite(grads.dual_vector.value))
    assert jnp.all(jnp.isfinite(grads.dual_matrix.value))

    def kl_from_vector(vector_value):
        model = eqx.tree_at(lambda t: t.dual_vector.value, q, vector_value)
        return model.prior_kl()

    zero_vector = jnp.zeros_like(_val(q.dual_vector))
    batch = jnp.stack([_val(q.dual_vector), _val(q.dual_vector) + 0.5, zero_vector])
    batched = jax.vmap(kl_from_vector)(batch)

    assert batched.shape == (3,)
    np.testing.assert_allclose(
        np.float64(batched[0]), np.float64(eager), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.float64(batched[2]),
        np.float64(kl_from_vector(zero_vector)),
        rtol=1e-12,
        atol=1e-14,
    )


def test_dual_working_matrices_reconstruct_r() -> None:
    """``Lr`` is lower triangular and satisfies ``Lr Lr^T = Kzz + Kzz L2 Kzz``."""
    q = _build_dual_family(6, seed=5)
    gram, _, root_working = q._working_matrices()
    dual_matrix = _val(q.dual_matrix)

    working = gram + gram @ dual_matrix @ gram
    np.testing.assert_allclose(
        np.asarray(root_working @ root_working.T), np.asarray(working), rtol=1e-10
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.triu(root_working, 1)), np.zeros((6, 6))
    )


@pytest.mark.parametrize(("variance", "num_inducing"), [(1e4, 80), (1e3, 50)])
def test_dual_working_matrices_survive_a_large_variance_kernel(
    variance: float, num_inducing: int
) -> None:
    """A badly scaled Kzz must not make chol(R) return NaN.

    Forming ``R = Kzz + Kzz L2 Kzz`` explicitly carries a rounding error of order
    ``||Kzz||^2 ||L2|| eps``, which for these settings dwarfs
    ``lambda_min(R) ~ jitter`` and made ``jnp.linalg.cholesky`` return NaN silently --
    poisoning every later ``fit_natgrads`` iterate. Factorising in the Kzz basis
    instead is unconditionally safe. The default 1e-6 jitter is used deliberately;
    the failure is invisible at the 1e-8 the other dual tests run at.
    """
    q = _build_dual_family(num_inducing, seed=6, jitter=1e-6, variance=float(variance))
    inputs = jnp.linspace(-3.0, 3.0, 9).reshape(-1, 1)

    _, _, root_working = q._working_matrices()
    assert jnp.all(jnp.isfinite(root_working))
    assert jnp.isfinite(q.prior_kl())
    assert jnp.all(jnp.isfinite(jnp.stack(q.marginals(inputs))))


def test_dual_family_cholesky_budget(monkeypatch) -> None:
    """Both routines factorise Kzz and R once each, and nothing else."""
    q = _build_dual_family(4, seed=2)
    inputs = jnp.linspace(-3.0, 3.0, 11).reshape(-1, 1)

    kl_counts = _count_cholesky_calls(monkeypatch, q.prior_kl)
    assert kl_counts["dense_cholesky"] == 2
    assert kl_counts["cholesky_factor"] == 0

    marginal_counts = _count_cholesky_calls(monkeypatch, lambda: q.marginals(inputs))
    assert marginal_counts["dense_cholesky"] == 2
    assert marginal_counts["cholesky_factor"] == 0


def test_dual_marginals_include_jitter() -> None:
    """``marginals`` must inflate every variance by exactly ``jitter``.

    ``VariationalGaussian.predict`` runs ``add_jitter`` on its output covariance, so
    the per-point variances ``elbo`` sees carry the same offset. Without it,
    ``dual_elbo`` misses ``elbo`` at matched moments by ``N * jitter / (2 sigma^2)``.
    """
    q = _build_dual_family(5, seed=4)
    inputs = jnp.linspace(-3.0, 3.0, 13).reshape(-1, 1)

    gram, root_gram, root_working = q._working_matrices()
    cross = q.posterior.prior.kernel.cross_covariance(_val(q.inducing_inputs), inputs)
    diagonal = jnp.diag(q.posterior.prior.kernel.gram(inputs).as_matrix())
    prior_projection = jsp.linalg.solve_triangular(root_gram, cross, lower=True)
    site_projection = jsp.linalg.solve_triangular(root_working, cross, lower=True)
    analytic = (
        diagonal
        - jnp.sum(jnp.square(prior_projection), axis=0)
        + jnp.sum(jnp.square(site_projection), axis=0)
    )
    del gram

    _, variance = q.marginals(inputs)
    np.testing.assert_allclose(
        np.asarray(variance - analytic), q.jitter, rtol=0.0, atol=1e-15
    )
