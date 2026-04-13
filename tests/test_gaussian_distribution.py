import importlib
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import lineax as lx


def _load_distributions():
    """Load gpjax.distributions without triggering the full gpjax.__init__."""
    gpjax_root = os.path.join(os.path.dirname(__file__), "..", "gpjax")

    # Ensure gpjax package exists in sys.modules as a namespace so
    # submodule imports like 'from gpjax.linalg import ...' work.
    if "gpjax" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "gpjax",
            os.path.join(gpjax_root, "__init__.py"),
            submodule_search_locations=[gpjax_root],
        )
        mod = type(sys)("gpjax")
        mod.__path__ = [gpjax_root]
        mod.__package__ = "gpjax"
        mod.__spec__ = spec
        sys.modules["gpjax"] = mod

    # Now import the submodules that distributions.py actually needs.
    importlib.import_module("gpjax.typing")
    importlib.import_module("gpjax.linalg")

    # Finally import distributions itself.
    return importlib.import_module("gpjax.distributions")


_dist_mod = _load_distributions()
GaussianDistribution = _dist_mod.GaussianDistribution


def test_mean():
    mu = jnp.array([1.0, 2.0])
    cov = lx.MatrixLinearOperator(jnp.eye(2))
    d = GaussianDistribution(loc=mu, scale=cov)
    assert jnp.allclose(d.mean, mu)


def test_variance():
    mu = jnp.zeros(3)
    cov = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0, 9.0]))
    d = GaussianDistribution(loc=mu, scale=cov)
    assert jnp.allclose(d.variance, jnp.array([1.0, 4.0, 9.0]))


def test_sample_shape():
    mu = jnp.zeros(2)
    cov = lx.MatrixLinearOperator(jnp.eye(2))
    d = GaussianDistribution(loc=mu, scale=cov)
    samples = d.sample(jax.random.key(0), sample_shape=(10,))
    assert samples.shape == (10, 2)


def test_log_prob_standard_normal():
    mu = jnp.zeros(2)
    cov = lx.MatrixLinearOperator(jnp.eye(2))
    d = GaussianDistribution(loc=mu, scale=cov)
    lp = d.log_prob(jnp.zeros(2))
    expected = -0.5 * 2 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(lp, expected, atol=1e-5)


def test_covariance_returns_dense():
    mu = jnp.zeros(2)
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    cov = lx.MatrixLinearOperator(A)
    d = GaussianDistribution(loc=mu, scale=cov)
    assert jnp.allclose(d.covariance(), A)
