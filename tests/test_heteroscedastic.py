# Copyright 2024 The GPJax Contributors.
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
import equinox as eqx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import (
    HeteroscedasticModel,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import (
    AbstractLikelihood,
    HeteroscedasticGaussian,
    LogNormalTransform,
    NoiseMoments,
    SoftplusTransform,
)
from gpjax.mean_functions import Zero
from gpjax.objectives import heteroscedastic_elbo
from gpjax.variational_families import (
    HeteroscedasticPrediction,
    HeteroscedasticVariationalFamily,
    VariationalGaussianInit,
)
from hypothesis import (
    given,
    strategies as st,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import paramax
import pytest

config.update("jax_enable_x64", True)

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)


@pytest.fixture
def prior() -> Prior:
    return Prior(kernel=RBF(), mean_function=Zero())


@pytest.fixture
def noise_prior() -> Prior:
    return Prior(kernel=RBF(), mean_function=Zero())


@pytest.fixture
def dataset() -> Dataset:
    x = jnp.linspace(-2.0, 2.0, 10)[:, None]
    y = jnp.sin(x)
    return Dataset(X=x, y=y)


class SoftplusHeteroscedastic(HeteroscedasticGaussian):
    def supports_tight_bound(self) -> bool:
        return False


def test_construct_model_routing(prior, noise_prior):
    likelihood = HeteroscedasticGaussian()
    with pytest.raises(ValueError, match="noise_prior"):
        _ = prior * likelihood

    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )
    assert isinstance(posterior, HeteroscedasticModel)
    assert posterior.noise_prior is noise_prior
    assert posterior.noise_model.prior is noise_prior

    chained_posterior = HeteroscedasticModel(
        prior=prior, likelihood=SoftplusHeteroscedastic(), noise_prior=noise_prior
    )
    assert isinstance(chained_posterior, HeteroscedasticModel)


def test_likelihood_callable_compatibility(noise_prior):
    # Test that passing jnp.exp uses LogNormalTransform
    lik_exp = HeteroscedasticGaussian(noise_transform=jnp.exp)
    assert isinstance(lik_exp.noise_transform, LogNormalTransform)

    # Test that passing a custom callable uses SoftplusTransform (default fallback logic)
    def custom_transform(x):
        return jnp.square(x)

    lik_custom = HeteroscedasticGaussian(noise_transform=custom_transform)
    assert isinstance(lik_custom.noise_transform, SoftplusTransform)


def test_heteroscedastic_gaussian_validation(noise_prior, dataset):
    lik = HeteroscedasticGaussian()
    # Construct a valid GaussianDistribution to satisfy jaxtyping
    scale = lx.MatrixLinearOperator(jnp.eye(10))
    dist = gpx.distributions.GaussianDistribution(loc=jnp.zeros(10), scale=scale)

    # Test predict raises ValueError if noise_dist is None
    with pytest.raises(
        ValueError, match="noise_dist must be provided for heteroscedastic prediction"
    ):
        lik.predict(dist, noise_dist=None)

    # Test expected_log_likelihood raises ValueError if moments are None
    with pytest.raises(ValueError, match="mean_g and variance_g must be provided"):
        lik.expected_log_likelihood(
            dataset.y, dataset.X, dataset.X, mean_g=None, variance_g=None
        )


@given(num_data=st.integers(min_value=1, max_value=100))
def test_log_normal_transform_moments(num_data: int):
    transform = LogNormalTransform()
    mean = jnp.array([[0.5] for _ in range(num_data)])
    variance = jnp.array([[0.1] for _ in range(num_data)])

    moments = transform.moments(mean, variance)

    expected_variance = jnp.exp(mean + 0.5 * variance)
    expected_log_variance = mean
    expected_inv_variance = jnp.exp(-mean + 0.5 * variance)

    assert jnp.allclose(moments.variance, expected_variance)
    assert jnp.allclose(moments.log_variance, expected_log_variance)
    assert jnp.allclose(moments.inv_variance, expected_inv_variance)


@given(
    mean=st.floats(
        min_value=-2.0,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    variance=st.floats(min_value=1e-3, max_value=3.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_softplus_transform_numerical_accuracy(mean: float, variance: float, seed: int):
    # Monte Carlo verification of SoftplusTransform moments over a range of inputs
    transform = SoftplusTransform(num_points=100)
    mean_array = jnp.array([[mean]])
    variance_array = jnp.array([[variance]])

    moments = transform.moments(mean_array, variance_array)

    key = jr.key(seed)
    samples = mean_array + jnp.sqrt(variance_array) * jr.normal(key, (1000000, 1))
    transformed_samples = jax.nn.softplus(samples)

    # E[sigma^2]
    mc_variance = jnp.mean(transformed_samples)
    # E[log(sigma^2)]
    mc_log_variance = jnp.mean(jnp.log(transformed_samples))
    # E[1/sigma^2]
    mc_inv_variance = jnp.mean(1.0 / transformed_samples)

    # Allow for some MC error and quadrature approximation error
    rtol = 0.15
    atol = 0.02  # absolute tolerance for values near zero
    assert jnp.allclose(moments.variance, mc_variance, rtol=rtol, atol=atol)
    assert jnp.allclose(moments.log_variance, mc_log_variance, rtol=rtol, atol=atol)
    assert jnp.allclose(moments.inv_variance, mc_inv_variance, rtol=rtol, atol=atol)


def test_heteroscedastic_variational_predict(prior, noise_prior, dataset):
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=HeteroscedasticGaussian(), noise_prior=noise_prior
    )
    variational = HeteroscedasticVariationalFamily(
        model=posterior, inducing_inputs=dataset.X, inducing_inputs_g=dataset.X[::2]
    )

    mf, vf, _mg, _vg = variational.predict(dataset.X)
    assert mf.shape == (dataset.n, 1)
    assert vf.shape == (dataset.n, 1)
    assert _mg.shape == (dataset.n, 1)
    assert _vg.shape == (dataset.n, 1)

    kl = variational.prior_kl()
    assert jnp.isfinite(kl)

    latent_f, latent_g = variational.predict_latents(dataset.X)
    assert latent_f.mean.shape[0] == dataset.n
    assert latent_g.mean.shape[0] == dataset.n


@given(
    n_inducing=st.integers(min_value=1, max_value=10),
    offset=st.floats(
        min_value=-0.5,
        max_value=0.5,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_variational_family_init_structure(n_inducing: int, offset: float):
    prior = Prior(kernel=RBF(), mean_function=Zero())
    noise_prior = Prior(kernel=RBF(), mean_function=Zero())
    likelihood = HeteroscedasticGaussian()
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )

    inducing_inputs = jnp.linspace(0.0, 1.0, n_inducing, dtype=jnp.float64).reshape(
        -1, 1
    )

    signal_init = VariationalGaussianInit(inducing_inputs=inducing_inputs)
    noise_inducing = inducing_inputs + jnp.asarray(offset, dtype=jnp.float64)
    noise_init = VariationalGaussianInit(inducing_inputs=noise_inducing)

    q = HeteroscedasticVariationalFamily(
        model=posterior, signal_init=signal_init, noise_init=noise_init
    )

    assert jnp.allclose(q.signal_variational.inducing_inputs.unwrap(), inducing_inputs)
    assert jnp.allclose(q.noise_variational.inducing_inputs.unwrap(), noise_inducing)

    # Test initialization inference (noise inferred from signal)
    q_inferred = HeteroscedasticVariationalFamily(
        model=posterior, signal_init=signal_init
    )
    assert jnp.allclose(
        q_inferred.noise_variational.inducing_inputs.unwrap(), inducing_inputs
    )


def test_variational_family_init_errors(prior, noise_prior):
    likelihood = HeteroscedasticGaussian()
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )

    # Case 1: No inputs provided
    with pytest.raises(
        ValueError, match="Either signal_init or inducing_inputs must be provided"
    ):
        HeteroscedasticVariationalFamily(model=posterior)


def test_variational_family_predict_return_type(prior, noise_prior):
    likelihood = HeteroscedasticGaussian()
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )

    n_inducing = 5
    inducing_inputs = jnp.linspace(0, 1, n_inducing).reshape(-1, 1)
    q = HeteroscedasticVariationalFamily(
        model=posterior, inducing_inputs=inducing_inputs
    )

    test_inputs = jnp.linspace(0.5, 0.6, 3).reshape(-1, 1)
    prediction = q.predict(test_inputs)

    assert isinstance(prediction, HeteroscedasticPrediction)
    assert hasattr(prediction, "mean_f")
    assert hasattr(prediction, "variance_f")
    assert hasattr(prediction, "mean_g")
    assert hasattr(prediction, "variance_g")

    # Check backward compatibility (unpacking)
    mf, _vf, _mg, _vg = prediction
    assert jnp.allclose(mf, prediction.mean_f)


def test_heteroscedastic_elbo_gradients(dataset, prior, noise_prior):
    def _build_variational(likelihood_cls: type[HeteroscedasticGaussian]):
        likelihood = likelihood_cls()
        posterior = HeteroscedasticModel(
            prior=prior, likelihood=likelihood, noise_prior=noise_prior
        )
        return HeteroscedasticVariationalFamily(
            model=posterior, inducing_inputs=dataset.X
        )

    for likelihood_cls in (HeteroscedasticGaussian, SoftplusHeteroscedastic):
        variational = _build_variational(likelihood_cls)
        params, static = eqx.partition(variational, eqx.is_array)

        def loss(p, static=static):
            model = paramax.unwrap(eqx.combine(p, static))
            return -heteroscedastic_elbo(model, dataset)

        loss_val = loss(params)
        loss_jit = jax.jit(loss)(params)
        _ = jax.grad(loss)(params)

        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(loss_jit)


def test_jit_prediction(prior, noise_prior, dataset):
    likelihood = HeteroscedasticGaussian()
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )
    q = HeteroscedasticVariationalFamily(model=posterior, inducing_inputs=dataset.X)

    # JIT compile the predict call via a wrapper function
    @jax.jit
    def predict_jit(x):
        return q.predict(x)

    mf, _vf, _mg, _vg = predict_jit(dataset.X)

    assert mf.shape == (dataset.n, 1)
    assert jnp.isfinite(mf).all()

    # JIT compile transforms (testing low-level JIT)
    log_transform = LogNormalTransform()
    moments_fn = jax.jit(log_transform.moments)

    mu = jnp.array([[0.0]])
    var = jnp.array([[1.0]])
    moments = moments_fn(mu, var)
    assert jnp.isfinite(moments.variance).all()

    # Test SoftplusTransform JIT
    softplus_transform = SoftplusTransform(num_points=20)
    moments_fn_soft = jax.jit(softplus_transform.moments)
    moments_soft = moments_fn_soft(mu, var)
    assert jnp.isfinite(moments_soft.variance).all()


def test_jit_likelihood_prediction(dataset, prior, noise_prior):
    # Separate test for likelihood prediction to keep things clean
    likelihood = HeteroscedasticGaussian()

    # JIT compile likelihood prediction
    # We pass arrays and reconstruct distributions inside to ensure Pytree safety
    def lik_predict(f_mean, f_cov, g_mean, g_cov):
        f = gpx.distributions.GaussianDistribution(
            f_mean, lx.MatrixLinearOperator(f_cov)
        )
        g = gpx.distributions.GaussianDistribution(
            g_mean, lx.MatrixLinearOperator(g_cov)
        )
        return likelihood.predict(f, g).mean

    lik_predict_jit = jax.jit(lik_predict)

    cov = jnp.eye(dataset.n)
    mu = jnp.zeros(dataset.n)
    res = lik_predict_jit(mu, cov, mu, cov)
    assert res.shape == (dataset.n,)


def test_predictive_variance_tracks_noise(prior, noise_prior):
    x = jnp.array([[-1.0], [1.0]])
    likelihood = HeteroscedasticGaussian()
    posterior = HeteroscedasticModel(
        prior=prior, likelihood=likelihood, noise_prior=noise_prior
    )

    variational = HeteroscedasticVariationalFamily(
        model=posterior,
        inducing_inputs=x,
        inducing_inputs_g=x,
        variational_mean_g=jnp.array([[-1.0], [1.5]]),
    )

    signal_dist, noise_dist = variational.predict_latents(x)
    predictive = likelihood.predict(signal_dist, noise_dist)
    diag_cov = jnp.diag(predictive.covariance_matrix)

    assert diag_cov[1] > diag_cov[0]


def test_noise_moments_pytree_registration():
    # Explicitly test Pytree registration for NoiseMoments
    nm = NoiseMoments(
        log_variance=jnp.array([1.0]),
        inv_variance=jnp.array([0.5]),
        variance=jnp.array([2.0]),
    )
    leaves, treedef = jax.tree_util.tree_flatten(nm)

    # Check structure
    assert len(leaves) == 3
    assert leaves[0] is nm.log_variance
    assert leaves[1] is nm.inv_variance
    assert leaves[2] is nm.variance

    # Check unflattening
    nm_restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(nm_restored, NoiseMoments)
    assert jnp.allclose(nm_restored.log_variance, nm.log_variance)
    assert jnp.allclose(nm_restored.inv_variance, nm.inv_variance)
    assert jnp.allclose(nm_restored.variance, nm.variance)


# ---------------------------------------------------------------------------
# link_function must not silently evaluate the noise at g = 0 (issue #670)
# ---------------------------------------------------------------------------


def test_link_function_requires_the_noise_latent(noise_prior):
    """Without `g` there is no conditional to return, so it must refuse.

    Previously this silently returned N(y | f, σ²(0)) — the prior-noise density
    rather than the Lázaro-Gredilla & Titsias (2011) conditional N(y | f, σ²(g)).
    """
    likelihood = HeteroscedasticGaussian()
    f = jnp.array([0.5, -0.2, 1.0])

    with pytest.raises(ValueError, match="noise latent"):
        likelihood.link_function(f)


def test_link_function_uses_the_supplied_noise_latent(noise_prior):
    """Given `g`, the scale must be σ(g) — the whole point of the likelihood."""
    likelihood = HeteroscedasticGaussian()
    f = jnp.array([0.5, -0.2, 1.0])

    for g_value in (-2.0, 0.0, 2.0):
        g = jnp.full_like(f, g_value)
        dist = likelihood.link_function(f, g)

        assert jnp.allclose(dist.loc, f)
        assert jnp.allclose(dist.scale, jnp.sqrt(likelihood.noise_transform(g)))


def test_link_function_scale_varies_with_the_noise_latent(noise_prior):
    """A heteroscedastic scale that ignores g is the bug in #670."""
    likelihood = HeteroscedasticGaussian()
    f = jnp.zeros(3)

    low = likelihood.link_function(f, jnp.full_like(f, -2.0)).scale
    high = likelihood.link_function(f, jnp.full_like(f, 2.0)).scale

    assert jnp.all(low < high)
    # And neither may coincide with the old σ(0) answer.
    at_zero = jnp.sqrt(likelihood.noise_transform(jnp.zeros_like(f)))
    assert not jnp.allclose(low, at_zero)
    assert not jnp.allclose(high, at_zero)


def test_heteroscedastic_subclass_cannot_silently_use_prior_noise(noise_prior):
    """The inherited `expected_log_likelihood` must not quietly use σ²(0).

    `AbstractLikelihood.expected_log_likelihood` calls `link_function(f)` with a
    single argument. A heteroscedastic subclass that does not override it would
    otherwise integrate against the prior-noise density without any warning.
    """

    class BareHeteroscedastic(HeteroscedasticGaussian):
        expected_log_likelihood = AbstractLikelihood.expected_log_likelihood

    likelihood = BareHeteroscedastic()
    y = jnp.zeros((3, 1))
    mean = jnp.zeros((3, 1))
    variance = jnp.ones((3, 1))

    with pytest.raises(ValueError, match="noise latent"):
        likelihood.expected_log_likelihood(y, mean, variance)


def test_sanctioned_heteroscedastic_path_is_unaffected(noise_prior):
    """`HeteroscedasticGaussian.expected_log_likelihood` still works via g moments."""
    likelihood = HeteroscedasticGaussian()
    y = jnp.zeros((3, 1))
    mean = jnp.zeros((3, 1))
    variance = jnp.ones((3, 1))
    mean_g = jnp.full((3, 1), 0.3)
    variance_g = jnp.full((3, 1), 0.1)

    result = likelihood.expected_log_likelihood(
        y, mean, variance, mean_g=mean_g, variance_g=variance_g
    )

    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))
