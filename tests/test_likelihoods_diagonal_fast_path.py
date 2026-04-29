"""Tests for the _diagonal_scale helper in gpjax.likelihoods."""

from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import (
    Gaussian,
    _diagonal_scale,
)
import jax.numpy as jnp
import lineax as lx


def test_diagonal_scale_plain_diagonal_returns_inner():
    diag = jnp.array([1.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(diag)
    result = _diagonal_scale(op)
    assert result is op


def test_diagonal_scale_tagged_diagonal_returns_inner_diagonal():
    diag = jnp.array([1.0, 2.0, 3.0])
    inner = lx.DiagonalLinearOperator(diag)
    tagged = lx.TaggedLinearOperator(inner, lx.positive_semidefinite_tag)
    result = _diagonal_scale(tagged)
    assert isinstance(result, lx.DiagonalLinearOperator)
    assert result is inner


def test_diagonal_scale_dense_returns_none():
    dense = lx.MatrixLinearOperator(jnp.eye(3) * 2.0)
    assert _diagonal_scale(dense) is None


def test_diagonal_scale_tagged_dense_returns_none():
    dense = lx.MatrixLinearOperator(jnp.eye(3) * 2.0)
    tagged = lx.TaggedLinearOperator(dense, lx.positive_semidefinite_tag)
    assert _diagonal_scale(tagged) is None


def test_gaussian_predict_preserves_diagonal_scale():
    """Diagonal GaussianDistribution input must produce diagonal output."""
    mean = jnp.array([0.0, 1.0, 2.0])
    diag_scale = lx.DiagonalLinearOperator(jnp.array([0.5, 1.0, 2.0]))
    dist_in = GaussianDistribution(mean, diag_scale)

    likelihood = Gaussian(num_datapoints=3, obs_stddev=0.1)
    dist_out = likelihood.predict(dist_in)

    assert isinstance(dist_out, GaussianDistribution)
    # Fast path preserves diagonal scale
    assert isinstance(dist_out.scale, lx.DiagonalLinearOperator)
    # Values: diag = [0.5, 1.0, 2.0] + 0.1^2 = [0.51, 1.01, 2.01]
    expected_diag = jnp.array([0.5, 1.0, 2.0]) + 0.01
    assert jnp.allclose(lx.diagonal(dist_out.scale), expected_diag)


def test_gaussian_predict_tagged_diagonal_preserves_diagonal_scale():
    """Tagged-diagonal input must still hit the fast path."""
    mean = jnp.array([0.0, 1.0])
    diag = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
    tagged = lx.TaggedLinearOperator(diag, lx.positive_semidefinite_tag)
    dist_in = GaussianDistribution(mean, tagged)

    likelihood = Gaussian(num_datapoints=2, obs_stddev=0.5)
    dist_out = likelihood.predict(dist_in)

    assert isinstance(dist_out, GaussianDistribution)
    assert isinstance(dist_out.scale, lx.DiagonalLinearOperator)
    expected_diag = jnp.array([1.0, 4.0]) + 0.25
    assert jnp.allclose(lx.diagonal(dist_out.scale), expected_diag)


def test_gaussian_predict_dense_returns_gaussian_distribution_with_matrix_scale():
    """Dense-covariance input widens to GaussianDistribution with MatrixLinearOperator scale."""
    mean = jnp.array([0.0, 1.0])
    cov = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    dense = lx.MatrixLinearOperator(cov)
    dist_in = GaussianDistribution(mean, dense)

    likelihood = Gaussian(num_datapoints=2, obs_stddev=0.1)
    dist_out = likelihood.predict(dist_in)

    assert isinstance(dist_out, GaussianDistribution)
    # Dense branch: scale is a MatrixLinearOperator
    assert isinstance(dist_out.scale, lx.MatrixLinearOperator)
    expected_cov = cov + jnp.eye(2) * 0.01
    assert jnp.allclose(dist_out.covariance_matrix, expected_cov)
