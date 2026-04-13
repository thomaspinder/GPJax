"""Tests for the Hilbert Space Gaussian Process (HSGP) kernel approximation."""

from gpjax.kernels.approximations.hsgp import HSGP
from gpjax.kernels.stationary import RBF, Matern12, Matern32, Matern52
from gpjax.linalg.operators import LowRank
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)

STATIONARY_KERNELS = [RBF, Matern12, Matern32, Matern52]


def _make_hsgp(
    kernel_class=RBF,
    num_basis_fns: int = 20,
    domain_half_width: float = 5.0,
    center: float = 0.0,
) -> HSGP:
    """Create an HSGP with sensible defaults for testing."""
    base_kernel = kernel_class(n_dims=1)
    return HSGP(
        base_kernel=base_kernel,
        num_basis_fns=num_basis_fns,
        domain_half_width=domain_half_width,
        center=center,
    )


# ──────────────────────────────────────────────────────────────────────
# Eigenvalues
# ──────────────────────────────────────────────────────────────────────


def test_eigenvalue_count():
    hsgp = _make_hsgp(num_basis_fns=10, domain_half_width=3.0)
    eigenvalues = hsgp.eigenvalues()
    assert eigenvalues.shape == (10,)


def test_eigenvalue_formula():
    """sqrt(lambda_j) = j * pi / (2 * L)."""
    half_width = 5.0
    num_basis = 4
    hsgp = _make_hsgp(num_basis_fns=num_basis, domain_half_width=half_width)
    eigenvalues = hsgp.eigenvalues()
    indices = jnp.arange(1, num_basis + 1)
    expected = indices * jnp.pi / (2.0 * half_width)
    npt.assert_allclose(eigenvalues, expected)


def test_eigenvalues_are_strictly_increasing():
    hsgp = _make_hsgp(num_basis_fns=20, domain_half_width=3.0)
    eigenvalues = hsgp.eigenvalues()
    assert jnp.all(jnp.diff(eigenvalues) > 0)


# ──────────────────────────────────────────────────────────────────────
# Eigenfunctions
# ──────────────────────────────────────────────────────────────────────


def test_eigenfunction_shape():
    hsgp = _make_hsgp(num_basis_fns=10, domain_half_width=3.0)
    inputs = jnp.linspace(-2, 2, 50)[:, None]
    basis_matrix = hsgp.eigenfunctions(inputs)
    assert basis_matrix.shape == (50, 10)


def test_eigenfunctions_are_approximately_orthonormal():
    """Eigenfunctions should be approximately orthonormal over [-L, L]."""
    half_width = 3.0
    num_basis = 5
    hsgp = _make_hsgp(num_basis_fns=num_basis, domain_half_width=half_width, center=0.0)
    num_quadrature_points = 10_000
    inputs = jnp.linspace(-half_width, half_width, num_quadrature_points)[:, None]
    basis_matrix = hsgp.eigenfunctions(inputs)
    spacing = 2.0 * half_width / num_quadrature_points
    gram_matrix = basis_matrix.T @ basis_matrix * spacing
    npt.assert_allclose(gram_matrix, jnp.eye(num_basis), atol=1e-2)


def test_eigenfunctions_vanish_at_boundaries():
    """Eigenfunctions should be zero at x = -L and x = L."""
    half_width = 3.0
    hsgp = _make_hsgp(num_basis_fns=5, domain_half_width=half_width, center=0.0)
    boundary_points = jnp.array([[-half_width], [half_width]])
    basis_at_boundary = hsgp.eigenfunctions(boundary_points)
    npt.assert_allclose(basis_at_boundary, 0.0, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Centering
# ──────────────────────────────────────────────────────────────────────


def test_explicit_center_is_stored():
    hsgp = HSGP(
        base_kernel=RBF(n_dims=1),
        num_basis_fns=5,
        domain_half_width=3.0,
        center=1.0,
    )
    assert hsgp._center == 1.0


def test_auto_center_uses_midpoint_of_input_range():
    hsgp = _make_hsgp(num_basis_fns=5, domain_half_width=3.0, center=None)
    inputs = jnp.linspace(2.0, 8.0, 50)[:, None]
    _ = hsgp.eigenfunctions(inputs)
    assert hsgp._center == pytest.approx(5.0)


def test_auto_center_persists_across_calls():
    """Once set, auto-center should not change on subsequent calls."""
    hsgp = _make_hsgp(num_basis_fns=5, domain_half_width=3.0, center=None)
    first_inputs = jnp.linspace(0, 10, 50)[:, None]
    second_inputs = jnp.linspace(-5, 5, 50)[:, None]
    _ = hsgp.eigenfunctions(first_inputs)
    center_after_first_call = hsgp._center
    _ = hsgp.eigenfunctions(second_inputs)
    assert hsgp._center == center_after_first_call


# ──────────────────────────────────────────────────────────────────────
# compute_basis
# ──────────────────────────────────────────────────────────────────────


def test_compute_basis_returns_correct_shapes():
    hsgp = _make_hsgp(num_basis_fns=10, domain_half_width=3.0)
    inputs = jnp.linspace(-2, 2, 50)[:, None]
    basis_matrix, sqrt_spectral_weights = hsgp.compute_basis(inputs)
    assert basis_matrix.shape == (50, 10)
    assert sqrt_spectral_weights.shape == (10,)


def test_compute_basis_sqrt_psd_is_positive():
    hsgp = _make_hsgp(num_basis_fns=10, domain_half_width=3.0)
    inputs = jnp.linspace(-2, 2, 50)[:, None]
    _, sqrt_spectral_weights = hsgp.compute_basis(inputs)
    assert jnp.all(sqrt_spectral_weights > 0)


# ──────────────────────────────────────────────────────────────────────
# Gram, cross-covariance, and diagonal
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("KernelClass", STATIONARY_KERNELS)
def test_gram_returns_psd_low_rank_matrix(KernelClass):
    hsgp = _make_hsgp(kernel_class=KernelClass)
    inputs = jnp.linspace(-3, 3, 30)[:, None]
    gram_operator = hsgp.gram(inputs)

    assert isinstance(gram_operator, LowRank)
    gram_dense = gram_operator.to_dense()
    assert gram_dense.shape == (30, 30)

    eigenvalues_of_gram, _ = jnp.linalg.eigh(gram_dense + 1e-6 * jnp.eye(30))
    assert jnp.all(eigenvalues_of_gram > 0)


@pytest.mark.parametrize("KernelClass", STATIONARY_KERNELS)
def test_cross_covariance_shape(KernelClass):
    hsgp = _make_hsgp(kernel_class=KernelClass)
    inputs_a = jnp.linspace(-3, 3, 30)[:, None]
    inputs_b = jnp.linspace(-2, 2, 20)[:, None]
    cross_covariance = hsgp.cross_covariance(inputs_a, inputs_b)
    assert cross_covariance.shape == (30, 20)


def test_gram_is_symmetric():
    hsgp = _make_hsgp(kernel_class=RBF)
    inputs = jnp.linspace(-3, 3, 30)[:, None]
    gram_dense = hsgp.gram(inputs).to_dense()
    npt.assert_allclose(gram_dense, gram_dense.T, atol=1e-12)


@pytest.mark.parametrize("KernelClass", STATIONARY_KERNELS)
def test_diagonal_matches_gram_diagonal(KernelClass):
    hsgp = _make_hsgp(kernel_class=KernelClass)
    inputs = jnp.linspace(-3, 3, 30)[:, None]
    diagonal_dense = hsgp.diagonal(inputs).to_dense()
    gram_dense = hsgp.gram(inputs).to_dense()
    npt.assert_allclose(jnp.diag(diagonal_dense), jnp.diag(gram_dense), atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Convergence to exact kernel
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("KernelClass", [RBF, Matern32, Matern52])
def test_gram_error_decreases_with_more_basis_functions(KernelClass):
    """With more basis functions, the HSGP Gram matrix should converge to exact."""
    base_kernel = KernelClass(n_dims=1)
    inputs = jnp.linspace(-1, 1, 30)[:, None]
    exact_gram = base_kernel.gram(inputs).to_dense()

    hsgp_coarse = _make_hsgp(kernel_class=KernelClass, num_basis_fns=10)
    hsgp_fine = _make_hsgp(kernel_class=KernelClass, num_basis_fns=80)

    error_coarse = jnp.linalg.norm(exact_gram - hsgp_coarse.gram(inputs).to_dense())
    error_fine = jnp.linalg.norm(exact_gram - hsgp_fine.gram(inputs).to_dense())

    assert error_fine < error_coarse


def test_rbf_gram_closely_matches_exact():
    """RBF with many basis functions should be very close to exact."""
    base_kernel = RBF(n_dims=1)
    inputs = jnp.linspace(-1, 1, 20)[:, None]
    exact_gram = base_kernel.gram(inputs).to_dense()

    hsgp = _make_hsgp(kernel_class=RBF, num_basis_fns=100)
    approximate_gram = hsgp.gram(inputs).to_dense()

    max_absolute_error = jnp.max(jnp.abs(exact_gram - approximate_gram))
    assert max_absolute_error < 0.01


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────


def test_nonstationary_kernel_is_rejected():
    from gpjax.kernels.nonstationary import Linear

    with pytest.raises(TypeError):
        HSGP(base_kernel=Linear(1), num_basis_fns=10, domain_half_width=3.0)


def test_pointwise_call_raises():
    hsgp = _make_hsgp(num_basis_fns=10, domain_half_width=3.0)
    with pytest.raises(RuntimeError):
        hsgp(jnp.array([1.0]), jnp.array([2.0]))


# ──────────────────────────────────────────────────────────────────────
# Integration with Prior/Posterior pipeline
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def training_data():
    """Synthetic sinusoidal regression dataset."""
    key = jr.key(42)
    num_points = 50
    inputs = jnp.linspace(-3, 3, num_points)[:, None]
    targets = jnp.sin(inputs) + 0.1 * jr.normal(key, (num_points, 1))

    from gpjax.dataset import Dataset

    return Dataset(X=inputs, y=targets), num_points


@pytest.fixture
def hsgp_posterior(training_data):
    """Conjugate posterior with an HSGP-RBF kernel."""
    from gpjax.gps import Prior
    from gpjax.likelihoods import Gaussian
    from gpjax.mean_functions import Zero

    _, num_points = training_data
    hsgp = _make_hsgp(kernel_class=RBF, num_basis_fns=20)
    prior = Prior(kernel=hsgp, mean_function=Zero())
    likelihood = Gaussian(num_datapoints=num_points)
    return prior * likelihood


def test_conjugate_mll_returns_finite_scalar(training_data, hsgp_posterior):
    from gpjax.objectives import conjugate_mll

    dataset, _ = training_data
    mll = conjugate_mll(hsgp_posterior, dataset)
    assert jnp.isfinite(mll)


def test_posterior_predict_returns_finite_moments(training_data, hsgp_posterior):
    dataset, _ = training_data
    test_inputs = jnp.linspace(-2.5, 2.5, 30)[:, None]
    prediction = hsgp_posterior.predict(test_inputs, dataset)

    assert jnp.all(jnp.isfinite(prediction.mean))
    assert jnp.all(jnp.isfinite(prediction.covariance()))


def test_conjugate_mll_is_differentiable(training_data, hsgp_posterior):
    """conjugate_mll with HSGP must be differentiable w.r.t. kernel parameters."""
    from flax import nnx
    from gpjax.objectives import conjugate_mll

    dataset, _ = training_data
    graphdef, state = nnx.split(hsgp_posterior)

    def negative_mll(state):
        model = nnx.merge(graphdef, state)
        return -conjugate_mll(model, dataset)

    gradients = jax.grad(negative_mll)(state)
    flat_gradients = jax.tree.leaves(gradients)
    for grad_leaf in flat_gradients:
        assert jnp.all(jnp.isfinite(grad_leaf)), f"Non-finite gradient: {grad_leaf}"
