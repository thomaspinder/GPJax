import equinox as eqx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.objectives import (
    collapsed_elbo,
    conjugate_loocv,
    conjugate_mll,
    elbo,
    non_conjugate_mll,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import paramax
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)


def build_data(n_points: int, n_dims: int, key, binary: bool):
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points, n_dims))
    if binary:
        y = (
            0.5
            * jnp.sign(
                jnp.cos(
                    3 * x[:, 0].reshape(-1, 1)
                    + jr.normal(key, shape=(n_points, 1)) * 0.05
                )
            )
            + 0.5
        )
    else:
        y = (
            jnp.sin(x[:, 0]).reshape(-1, 1)
            + jr.normal(key=key, shape=(n_points, 1)) * 0.1
        )
    D = Dataset(X=x, y=y)
    return D


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_mll(n_points: int, n_dims: int, key_val: int):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian()
    post = p * likelihood

    # test simple call
    res_simple = -conjugate_mll(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    params, static = eqx.partition(post, eqx.is_array)

    def loss(params):
        posterior = paramax.unwrap(eqx.combine(params, static))
        return -conjugate_mll(posterior, D)

    res_wrapped = loss(params)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(params)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    grad = jax.grad(loss)
    _ = grad(params)


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_loocv(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)

    # Build model
    p = Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = Gaussian()
    post = p * likelihood

    # test simple call
    res_simple = -conjugate_loocv(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    params, static = eqx.partition(post, eqx.is_array)

    def loss(params):
        posterior = paramax.unwrap(eqx.combine(params, static))
        return -conjugate_loocv(posterior, D)

    res_wrapped = loss(params)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(params)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    _ = loss_grad(params)


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_non_conjugate_mll(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=True)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Bernoulli()
    post = (p * likelihood).init_latent(D.n)

    # test simple call
    res_simple = -non_conjugate_mll(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    params, static = eqx.partition(post, eqx.is_array)

    def loss(params):
        posterior = paramax.unwrap(eqx.combine(params, static))
        return -non_conjugate_mll(posterior, D)

    res_wrapped = loss(params)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(params)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    _ = loss_grad(params)


@pytest.mark.parametrize("n_points", [10, 20])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_collapsed_elbo(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)
    z = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points // 2, n_dims))

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian()
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=z
    )

    # test simple call
    res_simple = -collapsed_elbo(q, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # Data on the full dataset should be the same as the marginal likelihood
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=D.X
    )
    expected_value = -conjugate_mll(p * likelihood, D)
    actual_value = -collapsed_elbo(q, D)
    assert jnp.abs(actual_value - expected_value) / expected_value < 1e-6


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
@pytest.mark.parametrize("binary", [True, False])
def test_elbo(n_points, n_dims, key_val, binary: bool):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=binary)
    z = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points // 2, n_dims))

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Bernoulli() if binary else gpx.likelihoods.Gaussian()
    post = p * likelihood

    q = gpx.variational_families.VariationalGaussian(posterior=post, inducing_inputs=z)

    # test simple call
    res_simple = -elbo(q, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    params, static = eqx.partition(q, eqx.is_array)

    def loss(params):
        model = paramax.unwrap(eqx.combine(params, static))
        return -elbo(model, D)

    res_wrapped = loss(params)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(params)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    _ = loss_grad(params)


class TestMultiOutputConjugateMLL:
    @pytest.fixture
    def mo_setup(self):
        key = jax.random.PRNGKey(42)
        N, P = 20, 2
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack([jnp.sin(X.squeeze()), jnp.cos(X.squeeze())])
        data = Dataset(X=X, y=y)
        from gpjax.kernels.multioutput.icm import ICMKernel
        from gpjax.kernels.stationary import RBF
        from gpjax.likelihoods import MultiOutputGaussian
        from gpjax.mean_functions import Zero
        from gpjax.parameters import CoregionalizationMatrix

        coreg = CoregionalizationMatrix(num_outputs=P, rank=1, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        meanf = Zero()
        prior = Prior(mean_function=meanf, kernel=kernel)
        lik = MultiOutputGaussian(num_outputs=P)
        posterior = prior * lik
        return posterior, data

    def test_mll_returns_scalar(self, mo_setup):
        posterior, data = mo_setup
        mll = conjugate_mll(posterior, data)
        assert mll.shape == ()

    def test_mll_is_finite(self, mo_setup):
        posterior, data = mo_setup
        mll = conjugate_mll(posterior, data)
        assert jnp.isfinite(mll)

    def test_mll_is_negative(self, mo_setup):
        """Log-likelihood of real data should be finite and typically negative."""
        posterior, data = mo_setup
        mll = conjugate_mll(posterior, data)
        assert mll < 0.0

    def test_single_output_unchanged(self):
        """Existing single-output path is unaffected."""
        X = jnp.linspace(0, 1, 20).reshape(-1, 1)
        y = jnp.sin(X)
        data = Dataset(X=X, y=y)
        from gpjax.kernels.stationary import RBF
        from gpjax.mean_functions import Zero

        kernel = RBF()
        prior = Prior(mean_function=Zero(), kernel=kernel)
        lik = Gaussian()
        posterior = prior * lik
        mll = conjugate_mll(posterior, data)
        assert jnp.isfinite(mll)


def test_conjugate_loocv_multioutput_matches_brute_force():
    r"""Multi-output LOOCV (leave-one-scalar-out on the flattened NP system)
    must match an independent brute-force reference that drops row/col i from
    Sigma and re-solves.  The scalar-noise/raw-y bug mis-scores this."""
    from gpjax.kernels.multioutput.icm import ICMKernel
    from gpjax.kernels.stationary import RBF
    from gpjax.likelihoods import MultiOutputGaussian
    from gpjax.mean_functions import Zero
    from gpjax.parameters import CoregionalizationMatrix
    import numpyro.distributions as npd

    key = jr.key(0)
    N, P = 6, 2
    X = jnp.linspace(0.0, 1.0, N).reshape(-1, 1)
    y = jnp.column_stack([jnp.sin(X.squeeze()), jnp.cos(X.squeeze())])
    data = Dataset(X=X, y=y)

    coreg = CoregionalizationMatrix(num_outputs=P, rank=1, key=key)
    kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
    prior = Prior(mean_function=Zero(), kernel=kernel)
    lik = MultiOutputGaussian(num_outputs=P)
    posterior = paramax.unwrap(prior * lik)

    # --- Independent brute-force reference on the full [NP, NP] system ---
    mx = posterior.prior.mean_function(X)
    y_flat, mx_flat = posterior.likelihood.prepare_targets(y, mx)
    y_flat = y_flat.reshape(-1)
    mx_flat = mx_flat.reshape(-1)
    noise = posterior.likelihood.noise_vector(N)
    Kxx = posterior.prior.kernel.gram(X).as_matrix()
    Sigma = Kxx + jnp.eye(Kxx.shape[0]) * posterior.prior.jitter + jnp.diag(noise)

    NP_dim = Sigma.shape[0]
    total = 0.0
    for i in range(NP_dim):
        idx = jnp.array([j for j in range(NP_dim) if j != i])
        S_i = Sigma[jnp.ix_(idx, idx)]
        y_i = y_flat[idx]
        mx_i = mx_flat[idx]
        L_i = jnp.linalg.cholesky(S_i)
        alpha_i = jsp.linalg.cho_solve((L_i, True), y_i - mx_i)
        loo_mean_i = mx_flat[i] + Sigma[i, idx] @ alpha_i
        v_i = jsp.linalg.solve_triangular(L_i, Sigma[idx, i], lower=True)
        loo_var_i = Sigma[i, i] - jnp.dot(v_i, v_i)
        total += npd.Normal(loc=loo_mean_i, scale=jnp.sqrt(loo_var_i)).log_prob(
            y_flat[i]
        )

    # --- Closed-form LOOCV via the implementation ---
    closed_form = conjugate_loocv(posterior, data)

    assert jnp.allclose(closed_form, total, atol=1e-5)
