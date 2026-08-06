"""Integration tests validating end-to-end Equinox migration."""

import equinox as eqx
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import paramax
import pytest

config.update("jax_enable_x64", True)

import gpjax as gpx

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)


def test_full_gp_training_roundtrip():
    """Train a conjugate GP end-to-end with the new API."""
    X = jnp.linspace(0, 1, 20)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    trained, history = gpx.fit(
        model=posterior,
        objective=nmll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=50,
        verbose=False,
    )

    # Prediction works
    pred = trained(X, D)
    assert pred.mean.shape == (20,)
    # Loss decreased
    assert history[-1] < history[0]


def test_parameter_freezing_with_non_trainable():
    """NonTrainable parameters should not change during training."""
    kernel = gpx.kernels.RBF()
    original_variance = paramax.unwrap(kernel).variance

    frozen_kernel = eqx.tree_at(
        lambda k: k.variance, kernel, replace_fn=paramax.non_trainable
    )

    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=frozen_kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    X = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    trained, _ = gpx.fit(
        model=posterior,
        objective=nmll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=20,
        verbose=False,
    )

    final_variance = paramax.unwrap(trained.prior.kernel).variance
    assert jnp.allclose(final_variance, original_variance)


def test_non_conjugate_gp_training():
    """Train a non-conjugate GP (Bernoulli likelihood) end-to-end."""
    key = jr.key(123)
    X = jr.uniform(key, shape=(30, 1))
    y = (jnp.sin(3 * X) > 0).astype(jnp.float64)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Bernoulli()
    posterior = prior * likelihood

    nmll = lambda p, d: -gpx.objectives.non_conjugate_mll(p, d)
    _trained, history = gpx.fit(
        model=posterior,
        objective=nmll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=30,
        verbose=False,
    )

    assert history[-1] < history[0]


def test_lbfgs_training():
    """Train a GP using L-BFGS optimiser."""
    X = jnp.linspace(0, 1, 20)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    _trained, final_loss = gpx.fit_lbfgs(model=posterior, objective=nmll, train_data=D)

    assert jnp.isfinite(final_loss)


def test_kernel_composition():
    """Composed kernels (sum/product) work in the full pipeline."""
    X = jnp.linspace(0, 1, 20)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF() + gpx.kernels.Matern32()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    trained, _history = gpx.fit(
        model=posterior,
        objective=nmll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=30,
        verbose=False,
    )

    pred = trained(X, D)
    assert pred.mean.shape == (20,)


def test_jit_prediction():
    """JIT-compiled prediction works (extract arrays inside jit)."""
    X = jnp.linspace(0, 1, 20)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    @jax.jit
    def predict_mean(model, x, data):
        dist = model(x, data)
        return dist.mean

    mean = predict_mean(posterior, X, D)
    assert mean.shape == (20,)
    assert jnp.all(jnp.isfinite(mean))


def test_grad_through_model():
    """Gradients flow through the full model."""
    X = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    params, static = eqx.partition(posterior, eqx.is_array)

    def loss(params):
        model = paramax.unwrap(eqx.combine(params, static))
        return -gpx.objectives.conjugate_mll(model, D)

    grads = jax.grad(loss)(params)
    # Check that at least some gradients are non-zero
    flat_grads = jax.tree.leaves(grads)
    has_nonzero = any(jnp.any(g != 0.0) for g in flat_grads)
    assert has_nonzero


def test_init_subclass_docstring_inheritance():
    """Verify __init_subclass__ works with eqx.Module for docstring inheritance."""

    class Base(eqx.Module):
        def foo(self):
            """Base docstring."""

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            for attr_name, attr_value in cls.__dict__.items():
                if callable(attr_value) and attr_value.__doc__ is None:
                    for parent in cls.mro()[1:]:
                        if hasattr(parent, attr_name):
                            parent_attr = getattr(parent, attr_name)
                            if parent_attr.__doc__:
                                attr_value.__doc__ = parent_attr.__doc__
                                break

    class Child(Base):
        def foo(self):
            pass

    assert Child.foo.__doc__ == "Base docstring."
