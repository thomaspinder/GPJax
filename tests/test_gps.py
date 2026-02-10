# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

try:
    from beartype.roar import BeartypeCallHintParamViolation
    from jaxtyping import TypeCheckError

    ValidationErrors = (TypeError, BeartypeCallHintParamViolation, TypeCheckError)
except ImportError:
    ValidationErrors = TypeError

from collections.abc import Callable

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    AbstractPosterior,
    AbstractPrior,
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
    construct_posterior,
)
from gpjax.kernels import (
    RBF,
    AbstractKernel,
    Matern52,
)
from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
    Zero,
)
from jax import config
import jax.numpy as jnp
import jax.random as jr
from numpyro.distributions import Distribution as NumpyroDistribution
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_prior():
    # Abstract prior should not be able to be instantiated.
    with pytest.raises(TypeError):
        AbstractPrior()


def test_abstract_posterior():
    # Abstract posterior should not be able to be instantiated.
    with pytest.raises(TypeError):
        AbstractPosterior()


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_prior_with_diag(
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Check types.
    assert isinstance(prior, Prior)
    assert isinstance(prior, AbstractPrior)

    # Query a marginal distribution at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    marginal_distribution_diag = prior(inputs, return_covariance_type="diagonal")
    marginal_distribution_full = prior(inputs, return_covariance_type="dense")

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution_diag, GaussianDistribution)
    assert isinstance(marginal_distribution_diag, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution_diag.mean
    sigma = marginal_distribution_diag.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)
    # test that off diagonal elements are zero
    assert jnp.all((sigma - jnp.diag(jnp.diag(sigma))) == 0)
    # test that we return exactly the diagonal of the full covariance
    assert jnp.allclose(
        jnp.diag(sigma), jnp.diag(marginal_distribution_full.covariance())
    )


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_prior(
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Check types.
    assert isinstance(prior, Prior)
    assert isinstance(prior, AbstractPrior)

    # Query a marginal distribution at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    marginal_distribution = prior(inputs)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("num_test_datapoints", [1, 10, 200])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_conjugate_posterior_with_diag(
    num_datapoints: int,
    num_test_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Gaussian(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, ConjugatePosterior)

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_test_datapoints).reshape(-1, 1)
    marginal_distribution_diag = posterior(inputs, D, return_covariance_type="diagonal")
    marginal_distribution_full = posterior(inputs, D, return_covariance_type="dense")

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution_diag, GaussianDistribution)
    assert isinstance(marginal_distribution_diag, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution_diag.mean
    sigma = marginal_distribution_diag.covariance()
    assert mu.shape == (num_test_datapoints,)
    assert sigma.shape == (num_test_datapoints, num_test_datapoints)
    # test that off diagonal elements are zero
    assert jnp.all((sigma - jnp.diag(jnp.diag(sigma))) == 0)
    # test that we return exactly the diagonal of the full covariance
    assert jnp.allclose(
        jnp.diag(sigma), jnp.diag(marginal_distribution_full.covariance())
    )


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("num_test_datapoints", [1, 10, 200])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_conjugate_posterior(
    num_datapoints: int,
    num_test_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Gaussian(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, ConjugatePosterior)

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_test_datapoints).reshape(-1, 1)
    marginal_distribution = posterior(inputs, D)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_test_datapoints,)
    assert sigma.shape == (num_test_datapoints, num_test_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("num_test_datapoints", [1, 10, 200])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_nonconjugate_posterior_with_diag(
    num_datapoints: int,
    num_test_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Bernoulli(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = NonConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, NonConjugatePosterior)

    # Check latent values.
    latent_values = jr.normal(posterior.key, (num_datapoints, 1))
    assert (posterior.latent[...] == latent_values).all()

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_test_datapoints).reshape(-1, 1)
    marginal_distribution_diag = posterior(inputs, D, return_covariance_type="diagonal")
    marginal_distribution_full = posterior(inputs, D, return_covariance_type="dense")

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution_diag, GaussianDistribution)
    assert isinstance(marginal_distribution_diag, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution_diag.mean
    sigma = marginal_distribution_diag.covariance()
    assert mu.shape == (num_test_datapoints,)
    # We are still returning a full covariance, even though the off diagonal
    # should all be zeros...
    assert sigma.shape == (num_test_datapoints, num_test_datapoints)
    assert jnp.all((sigma - jnp.diag(jnp.diag(sigma))) == 0)
    # test that we return exactly the diagonal of the full covariance
    assert jnp.allclose(
        jnp.diag(sigma), jnp.diag(marginal_distribution_full.covariance())
    )


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("num_test_datapoints", [1, 10, 200])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_nonconjugate_posterior(
    num_datapoints: int,
    num_test_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Bernoulli(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = NonConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, NonConjugatePosterior)

    # Check latent values.
    latent_values = jr.normal(posterior.key, (num_datapoints, 1))
    assert (posterior.latent[...] == latent_values).all()

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_test_datapoints).reshape(-1, 1)
    marginal_distribution = posterior(inputs, D)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, NumpyroDistribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_test_datapoints,)
    assert sigma.shape == (num_test_datapoints, num_test_datapoints)


@pytest.mark.parametrize("likelihood", [Bernoulli, Gaussian])
@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_posterior_construct(
    likelihood: type[AbstractLikelihood],
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Construct the posterior via the three methods.
    likelihood: AbstractLikelihood = likelihood(num_datapoints=num_datapoints)
    posterior_mul = prior * likelihood
    posterior_rmul = likelihood * prior
    posterior_manual = construct_posterior(prior=prior, likelihood=likelihood)

    # Ensure that the posterior is the same type in all three cases.
    assert type(posterior_mul) is type(posterior_rmul)
    assert type(posterior_mul) is type(posterior_manual)

    # Ensure we have the correct likelihood and prior.
    assert posterior_mul.likelihood == likelihood
    assert posterior_mul.prior == prior

    # If the likelihood is Gaussian, then the posterior should be conjugate.
    if isinstance(likelihood, Gaussian):
        assert isinstance(posterior_mul, ConjugatePosterior)

    # If the likelihood is Bernoulli or Poisson, then the posterior should be non-conjugate.
    if isinstance(likelihood, (Bernoulli, Poisson)):
        assert isinstance(posterior_mul, NonConjugatePosterior)


@pytest.mark.parametrize("num_datapoints", [1, 5])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_prior_sample_approx(num_datapoints, kernel, mean_function):
    kern = kernel(n_dims=2, lengthscale=jnp.array([5.0, 1.0]), variance=0.1)
    p = Prior(kernel=kern, mean_function=mean_function())
    key = jr.key(123)

    with pytest.raises(ValueError):
        p.sample_approx(-1, key)
    with pytest.raises(ValueError):
        p.sample_approx(0, key)
    with pytest.raises(ValidationErrors):
        p.sample_approx(0.5, key)
    with pytest.raises(ValueError):
        p.sample_approx(1, key, -10)
    with pytest.raises(ValueError):
        p.sample_approx(1, key, 0)
    with pytest.raises(ValidationErrors):
        p.sample_approx(1, key, 0.5)

    sampled_fn = p.sample_approx(1, key, 100)
    assert isinstance(sampled_fn, Callable)  # check type

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    evals = sampled_fn(x)
    assert evals.shape == (num_datapoints, 1.0)  # check shape

    sampled_fn_2 = p.sample_approx(1, key, 100)
    evals_2 = sampled_fn_2(x)
    max_delta = jnp.max(jnp.abs(evals - evals_2))
    assert max_delta == 0.0  # samples same for same seed

    new_key = jr.key(12345)
    sampled_fn_3 = p.sample_approx(1, new_key, num_features=100)
    evals_3 = sampled_fn_3(x)
    max_delta = jnp.max(jnp.abs(evals - evals_3))
    assert max_delta > 0.01  # samples different for different seed

    # Check validty of samples using Monte-Carlo
    sampled_fn = p.sample_approx(10_000, key, 100)
    sampled_evals = sampled_fn(x)
    approx_mean = jnp.mean(sampled_evals, -1)
    approx_var = jnp.var(sampled_evals, -1)
    true_predictive = p(x)
    true_mean = true_predictive.mean
    true_var = jnp.diagonal(true_predictive.covariance())
    max_error_in_mean = jnp.max(jnp.abs(approx_mean - true_mean))
    max_error_in_var = jnp.max(jnp.abs(approx_var - true_var))
    assert max_error_in_mean < 0.02  # check that samples are correct
    assert max_error_in_var < 0.05  # check that samples are correct


@pytest.mark.parametrize("num_datapoints", [1, 5])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_conjugate_posterior_sample_approx(num_datapoints, kernel, mean_function):
    kern = kernel(lengthscale=jnp.array([5.0, 1.0]), variance=0.1)
    p = Prior(kernel=kern, mean_function=mean_function()) * Gaussian(
        num_datapoints=num_datapoints
    )
    key = jr.key(123)

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    y = (
        jnp.mean(jnp.sin(x), 1, keepdims=True)
        + jr.normal(key=key, shape=(num_datapoints, 1)) * 0.1
    )
    D = Dataset(X=x, y=y)

    # with pytest.raises(ValueError):
    # p.sample_approx(-1, D, key)
    # with pytest.raises(ValueError):
    # p.sample_approx(0, D, key)
    # with pytest.raises(ValidationErrors):
    # p.sample_approx(0.5, D, key)
    # with pytest.raises(ValueError):
    # p.sample_approx(1, D, key, -10)
    # with pytest.raises(ValueError):
    # p.sample_approx(1, D, key, 0)
    # with pytest.raises(ValidationErrors):
    # p.sample_approx(1, D, key, 0.5)

    sampled_fn = p.sample_approx(1, D, key, num_features=100)
    assert isinstance(sampled_fn, Callable)  # check type

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    evals = sampled_fn(x)
    assert evals.shape == (num_datapoints, 1.0)  # check shape

    sampled_fn_2 = p.sample_approx(1, D, key, num_features=100)
    evals_2 = sampled_fn_2(x)
    max_delta = jnp.max(jnp.abs(evals - evals_2))
    assert max_delta == 0.0  # samples same for same seed

    new_key = jr.key(12345)
    sampled_fn_3 = p.sample_approx(1, D, new_key, num_features=100)
    evals_3 = sampled_fn_3(x)
    max_delta = jnp.max(jnp.abs(evals - evals_3))
    assert max_delta > 0.01  # samples different for different seed

    # Check validty of samples using Monte-Carlo
    sampled_fn = p.sample_approx(10_000, D, key, num_features=100)
    sampled_evals = sampled_fn(x)
    approx_mean = jnp.mean(sampled_evals, -1)
    approx_var = jnp.var(sampled_evals, -1)
    true_predictive = p(x, train_data=D)
    true_mean = true_predictive.mean
    true_var = jnp.diagonal(true_predictive.covariance())
    max_error_in_mean = jnp.max(jnp.abs(approx_mean - true_mean))
    max_error_in_var = jnp.max(jnp.abs(approx_var - true_var))
    assert max_error_in_mean < 0.02  # check that samples are correct
    assert max_error_in_var < 0.05  # check that samples are correct


class TestMultiOutputPosteriorPredict:
    @pytest.fixture
    def mo_setup(self):
        from gpjax.kernels.multioutput.icm import ICMKernel
        from gpjax.likelihoods import MultiOutputGaussian
        from gpjax.parameters import CoregionalizationMatrix

        key = jr.PRNGKey(42)
        N, P = 20, 2
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack([jnp.sin(X.squeeze()), jnp.cos(X.squeeze())])
        data = Dataset(X=X, y=y)
        coreg = CoregionalizationMatrix(num_outputs=P, rank=1, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        prior = Prior(mean_function=Zero(), kernel=kernel)
        lik = MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik
        return posterior, data, N, P

    def test_predict_mean_shape(self, mo_setup):
        """Posterior mean is [M*P] (flat joint vector)."""
        posterior, data, _N, P = mo_setup
        M = 5
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, data)
        assert pred.mean.shape == (M * P,)

    def test_predict_covariance_shape(self, mo_setup):
        """Posterior covariance is [MP, MP]."""
        posterior, data, _N, P = mo_setup
        M = 5
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, data)
        assert pred.covariance().shape == (M * P, M * P)

    def test_predict_mean_finite(self, mo_setup):
        """Posterior mean is finite."""
        posterior, data, _N, _P = mo_setup
        Xtest = jnp.linspace(0, 1, 5).reshape(-1, 1)
        pred = posterior.predict(Xtest, data)
        assert jnp.all(jnp.isfinite(pred.mean))

    def test_predict_covariance_psd(self, mo_setup):
        """Posterior covariance is positive semi-definite."""
        posterior, data, _N, _P = mo_setup
        Xtest = jnp.linspace(0, 1, 5).reshape(-1, 1)
        pred = posterior.predict(Xtest, data)
        eigvals = jnp.linalg.eigvalsh(pred.covariance())
        assert jnp.all(eigvals >= -1e-5)

    def test_predict_at_training_recovers_data(self, mo_setup):
        """At training points, posterior mean is close to training data."""
        posterior, data, N, P = mo_setup
        pred = posterior.predict(data.X, data)
        # Mean is output-major [NP], reshape to [P, N] then transpose to [N, P]
        mean_reshaped = pred.mean.reshape(P, N).T
        residual = jnp.abs(mean_reshaped - data.y)
        assert jnp.mean(residual) < 1.0  # Loose bound


class TestMultiOutputValidation:
    def test_mo_likelihood_with_so_kernel_raises(self):
        """MultiOutputGaussian + single-output kernel raises ValueError."""
        from gpjax.likelihoods import MultiOutputGaussian

        kernel = RBF()
        prior = Prior(mean_function=Zero(), kernel=kernel)
        lik = MultiOutputGaussian(num_datapoints=10, num_outputs=2)
        with pytest.raises(ValueError, match="multi-output kernel"):
            prior * lik

    def test_mo_kernel_with_so_likelihood_raises(self):
        """Multi-output kernel + Gaussian (not MultiOutput) raises ValueError."""
        from gpjax.kernels.multioutput.icm import ICMKernel
        from gpjax.parameters import CoregionalizationMatrix

        key = jr.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=2, rank=1, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        prior = Prior(mean_function=Zero(), kernel=kernel)
        lik = Gaussian(num_datapoints=10)
        with pytest.raises(ValueError, match="MultiOutputGaussian"):
            prior * lik


if __name__ == "__main__":
    test_conjugate_posterior_sample_approx(10, RBF, Zero)
