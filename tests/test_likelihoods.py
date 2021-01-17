import jax.numpy as jnp
import jax.random as jr
from gpjax.likelihoods import Bernoulli
import pytest


@pytest.mark.parametrize('n', [1, 10])
def test_bernoulli_shapes(n):
    key = jr.PRNGKey(123)
    likelihood = Bernoulli()
    fmean = jr.uniform(key=key, shape=(n, 1)) * -1
    fvar = jr.uniform(key=key, shape=(n, 1))
    mu, sigma = likelihood.predictive_moments(fmean, fvar)
    y = jnp.round(jr.uniform(key, shape=(n, )))
    lpd = likelihood.log_density(mu, y)
    print(lpd)
    assert mu.shape == (n, )
    assert sigma.shape == (n, )
    assert lpd.shape == (n, )
