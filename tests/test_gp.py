from gpjax import Gaussian, RBF
from gpjax.likelihoods import Bernoulli
from gpjax.gps import Prior
from gpjax.gps.posteriors import Posterior, PosteriorApprox
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_conjugate_posterior():
    p = Prior(RBF())
    lik = Gaussian()
    post = p * lik
    assert isinstance(post, Posterior)


def test_non_conjugate_poster():
    posterior = Prior(RBF()) * Bernoulli()
    assert isinstance(posterior, PosteriorApprox)


@pytest.mark.parametrize('n', [1, 10])
def test_prior_samples(n):
    input_pts = jnp.linspace(-1., 1., 100).reshape(-1, 1)
    key = jr.PRNGKey(123)
    p = Prior(RBF())
    samples = p.sample(input_pts, key=key, n_samples=n)
    assert samples.shape == (n, 100)


def test_neg_ll():
    x = jnp.array([0.5, 1.0]).reshape(-1, 1)
    y = jnp.sin(x)
    prior = Prior(RBF())
    post = prior * Gaussian()
    nmll = post.neg_mll(x, y)
    assert jnp.round(nmll, 2) == 2.60


def test_nvars():
    posterior = Prior(RBF()) * Gaussian()
    assert posterior.n_vars == 3


# TODO: Why does this not work for n=1?
def test_predict_shape(n = 10):
    key = jr.PRNGKey(123)
    x = jr.uniform(key, shape=(10, 1))
    y = jnp.sin(x)
    xtest = jnp.linspace(0., 1., n).reshape(-1, 1)
    posterior = Prior(RBF()) * Gaussian()
    mu, sigma = posterior.predict(xtest, x, y)
    assert mu.shape == (n, 1)
    assert sigma.shape == (n, n)


@pytest.mark.parametrize('ntest', [10, 100])
@pytest.mark.parametrize('n', [1, 10])
def test_non_conjugate_init(n, ntest):
    key = jr.PRNGKey(123)
    posterior = Prior(RBF()) * Bernoulli()
    x = jr.uniform(key = key, shape=(10, 1), minval=-1., maxval=1.)
    y = jnp.sign(x)
    xtest = jnp.linspace(-1., 1., ntest).reshape(-1, 1)

    mll = posterior.neg_mll(x, y)
    assert mll.shape == ()
    assert posterior.nu.untransform.shape == x.shape
    assert posterior.latent_init
    mu, cov = posterior.predict(xtest, x, y)
    assert mu.shape == cov.shape
    assert mu.shape == (ntest, )