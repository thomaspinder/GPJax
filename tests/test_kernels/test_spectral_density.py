"""Tests for the SpectralDensity class and kernel spectral density evaluation."""

from gpjax.kernels.stationary import RBF, Matern12, Matern32, Matern52
from gpjax.kernels.stationary.utils import SpectralDensity
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)


def test_spectral_density_has_sample():
    """SpectralDensity must expose sample() for RFF compatibility."""
    kernel = RBF(n_dims=1)
    sd = kernel.spectral_density
    assert isinstance(sd, SpectralDensity)
    samples = sd.sample(key=jr.key(0), sample_shape=(10, 1))
    assert samples.shape == (10, 1)


def test_spectral_density_callable():
    """SpectralDensity must be callable with (omega, variance, lengthscale)."""
    kernel = RBF(n_dims=1)
    sd = kernel.spectral_density
    omega = jnp.array([0.5, 1.0, 2.0])
    result = sd(omega, jnp.array(1.0), jnp.array(1.0))
    assert result.shape == (3,)
    assert jnp.all(result > 0)


def test_rbf_spectral_density_formula():
    """Verify the RBF spectral density against the known closed form.

    S(w) = variance * sqrt(2*pi) * lengthscale * exp(-0.5 * lengthscale^2 * w^2)
    """
    kernel = RBF(n_dims=1, variance=2.0, lengthscale=0.5)
    sd = kernel.spectral_density

    omega = jnp.array([0.0, 1.0, 3.0])
    variance = jnp.array(2.0)
    lengthscale = jnp.array(0.5)

    result = sd(omega, variance, lengthscale)
    expected = (
        variance
        * jnp.sqrt(2 * jnp.pi)
        * lengthscale
        * jnp.exp(-0.5 * lengthscale**2 * omega**2)
    )
    npt.assert_allclose(result, expected, atol=1e-12)


def test_rbf_spectral_density_peak_at_zero():
    """RBF spectral density peaks at omega=0 and decays monotonically."""
    kernel = RBF(n_dims=1)
    sd = kernel.spectral_density
    omega = jnp.linspace(0, 10, 100)
    values = sd(omega, jnp.array(1.0), jnp.array(1.0))
    # Peak at omega=0
    assert values[0] == jnp.max(values)
    # Monotonically decreasing
    assert jnp.all(jnp.diff(values) <= 0)


def test_matern12_spectral_density_formula():
    """Verify Matern12 (nu=1/2) spectral density.

    S(w) = variance * 2/ell * 1/(1/ell^2 + w^2)
    """
    kernel = Matern12(n_dims=1, variance=1.5, lengthscale=0.8)
    sd = kernel.spectral_density
    omega = jnp.array([0.0, 1.0, 5.0])
    v, ell = jnp.array(1.5), jnp.array(0.8)

    result = sd(omega, v, ell)
    expected = v * (2.0 / ell) / (1.0 / ell**2 + omega**2)
    npt.assert_allclose(result, expected, atol=1e-12)


def test_matern32_spectral_density_formula():
    """Verify Matern32 (nu=3/2) spectral density.

    S(w) = variance * 4*(sqrt(3)/ell)^3 / (3/ell^2 + w^2)^2
    """
    kernel = Matern32(n_dims=1, variance=2.0, lengthscale=1.5)
    sd = kernel.spectral_density
    omega = jnp.array([0.0, 1.0, 5.0])
    v, ell = jnp.array(2.0), jnp.array(1.5)

    result = sd(omega, v, ell)
    alpha = jnp.sqrt(3.0) / ell
    expected = v * 4.0 * alpha**3 / (3.0 / ell**2 + omega**2) ** 2
    npt.assert_allclose(result, expected, atol=1e-12)


def test_matern52_spectral_density_formula():
    """Verify Matern52 (nu=5/2) spectral density.

    S(w) = variance * (16/3)*(sqrt(5)/ell)^5 / (5/ell^2 + w^2)^3
    """
    kernel = Matern52(n_dims=1, variance=3.0, lengthscale=2.0)
    sd = kernel.spectral_density
    omega = jnp.array([0.0, 1.0, 5.0])
    v, ell = jnp.array(3.0), jnp.array(2.0)

    result = sd(omega, v, ell)
    alpha = jnp.sqrt(5.0) / ell
    expected = v * (16.0 / 3.0) * alpha**5 / (5.0 / ell**2 + omega**2) ** 3
    npt.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("KernelClass", [RBF, Matern12, Matern32, Matern52])
def test_spectral_density_positive(KernelClass):
    """All spectral densities must be positive for all omega."""
    kernel = KernelClass(n_dims=1)
    sd = kernel.spectral_density
    omega = jnp.linspace(0, 20, 200)
    values = sd(omega, jnp.array(1.0), jnp.array(1.0))
    assert jnp.all(values > 0)
