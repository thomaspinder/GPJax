import equinox as eqx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.linalg import add_jitter
from gpjax.natural_gradients import (
    natural_gradient_step,
    partition_variational,
)
from gpjax.objectives import (
    collapsed_elbo,
    conjugate_loocv,
    conjugate_mll,
    dual_elbo,
    elbo,
    non_conjugate_mll,
)
from gpjax.parameters import (
    PositiveReal,
    _val,
)
from gpjax.variational_families import DualVariationalGaussian
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
import numpy as np
import paramax
import pytest

from tests._dual_helpers import (
    DUAL_JITTER,
    matched_variational_gaussian as _matched_variational_gaussian,
    random_dual_sites as _random_dual_sites,
)

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


# ---------------------------------------------------------------------------
# dual_elbo -- the t-SVGP bound (Adam et al. 2021, arXiv:2111.03412)
# ---------------------------------------------------------------------------
_DUAL_JITTER = DUAL_JITTER


def _dual_setup(
    num_inducing: int = 5,
    num_data: int = 30,
    jitter: float = _DUAL_JITTER,
    binary: bool = False,
    inducing_at_inputs: bool = False,
):
    """A conjugate (or Bernoulli) SVGP setup with a non-zero mean function."""
    key_inputs, key_noise = jr.split(jr.key(42))
    inputs = jnp.sort(
        jr.uniform(key_inputs, (num_data, 1), minval=-2.0, maxval=2.0), axis=0
    )
    if binary:
        outputs = jr.bernoulli(key_noise, 0.5, (num_data, 1)).astype(jnp.float64)
        likelihood = gpx.likelihoods.Bernoulli(num_datapoints=num_data)
    else:
        outputs = jnp.sin(3.0 * inputs) + 0.1 * jr.normal(key_noise, (num_data, 1))
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=num_data, obs_stddev=jnp.array(0.37)
        )
    data = Dataset(X=inputs, y=outputs)

    kernel = gpx.kernels.RBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.3))
    mean_function = gpx.mean_functions.Constant(jnp.array(0.4))
    posterior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel) * likelihood

    inducing_inputs = (
        inputs
        if inducing_at_inputs
        else jnp.linspace(-2.0, 2.0, num_inducing).reshape(-1, 1)
    )
    return posterior, data, inducing_inputs, jitter


def _dual_natgrad_step(q_dual: DualVariationalGaussian, data: Dataset, rate: float):
    variational, hyper = partition_variational(q_dual)
    updated, _ = natural_gradient_step(
        variational,
        hyper,
        data,
        lambda model, batch: -dual_elbo(model, batch),
        jnp.asarray(rate),
    )
    return eqx.combine(updated, hyper)


def _kernel_hyper_gradient(objective_fn, family, data):
    """Gradient of ``objective_fn`` with respect to the kernel parameters only."""
    params, static = eqx.partition(family, eqx.is_array)

    def loss(trainable):
        return objective_fn(paramax.unwrap(eqx.combine(trainable, static)), data)

    grads = jax.grad(loss)(params)
    leaves = jtu.tree_leaves(grads.posterior.prior.kernel)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


@pytest.mark.parametrize("binary", [True, False])
def test_dual_elbo(binary: bool):
    posterior, data, inducing_inputs, jitter = _dual_setup(binary=binary)
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )

    res_simple = -dual_elbo(q, data)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    params, static = eqx.partition(q, eqx.is_array)

    def loss(params):
        model = paramax.unwrap(eqx.combine(params, static))
        return -dual_elbo(model, data)

    np.testing.assert_allclose(
        np.float64(loss(params)), np.float64(res_simple), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.float64(jax.jit(loss)(params)), np.float64(res_simple), atol=1e-12
    )

    grads = jax.grad(loss)(params)
    assert jnp.all(jnp.isfinite(grads.dual_vector.value))
    assert jnp.all(jnp.isfinite(grads.dual_matrix.value))


@pytest.mark.parametrize("sites", ["optimal", "random"])
@pytest.mark.parametrize("batch", ["full", "half"])
def test_dual_elbo_equals_elbo_at_matched_moments(sites: str, batch: str):
    """The two bounds are the same functional, so their values must coincide.

    The ``half`` arm evaluates both bounds on half the rows while the likelihood still
    declares the full ``num_datapoints``, so the mini-batch factor ``N / B`` is 2 on
    both sides. Without it the arm would only ever exercise ``N / B == 1`` and a
    mutation dropping the factor from ``dual_elbo`` would survive.
    """
    posterior, data, inducing_inputs, jitter = _dual_setup()
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )

    if sites == "optimal":
        q = _dual_natgrad_step(q, data, 1.0)
    else:
        dual_vector, dual_matrix = _random_dual_sites(3, q.num_inducing)
        q = eqx.tree_at(
            lambda t: (t.dual_vector, t.dual_matrix),
            q,
            (gpx.parameters.Real(dual_vector), gpx.parameters.Real(dual_matrix)),
        )

    if batch == "half":
        half = data.n // 2
        data = Dataset(X=data.X[:half], y=data.y[:half])
        assert posterior.likelihood.num_datapoints / data.n == 2.0

    np.testing.assert_allclose(
        np.float64(dual_elbo(q, data)),
        np.float64(elbo(_matched_variational_gaussian(q), data)),
        rtol=1e-10,
    )


def test_dual_elbo_applies_the_minibatch_scale():
    """``dual_elbo`` must scale the expectation term, and only it, by ``N / B``.

    Pinned directly rather than through the ``elbo`` comparison above, which would
    stay green if *both* bounds dropped the factor.
    """
    posterior, data, inducing_inputs, jitter = _dual_setup()
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    dual_vector, dual_matrix = _random_dual_sites(7, q.num_inducing)
    q = eqx.tree_at(
        lambda t: (t.dual_vector, t.dual_matrix),
        q,
        (gpx.parameters.Real(dual_vector), gpx.parameters.Real(dual_matrix)),
    )

    half = data.n // 2
    batch = Dataset(X=data.X[:half], y=data.y[:half])
    scale = posterior.likelihood.num_datapoints / batch.n

    mean, variance = q.marginals(batch.X)
    expectation = jnp.sum(
        posterior.likelihood.expected_log_likelihood(
            batch.y, mean[:, None], variance[:, None]
        )
    )
    expected = scale * expectation - q.prior_kl()

    np.testing.assert_allclose(
        np.float64(dual_elbo(q, batch)), np.float64(expected), rtol=1e-12
    )


def test_dual_elbo_equals_titsias_collapsed_bound():
    """One rho = 1 step on a conjugate model reproduces the Titsias bound.

    The family jitter is pushed to 1e-12 because ``marginals`` adds it to every
    marginal variance, which biases the bound by ``N * jitter / (2 sigma^2)`` -- at the
    usual 1e-8 that is 1.1e-6, four orders of magnitude outside the tolerance here.
    """
    jitter = 1e-12
    posterior, data, inducing_inputs, _ = _dual_setup(jitter=jitter)
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    q = _dual_natgrad_step(q, data, 1.0)

    kernel = posterior.prior.kernel
    gram = add_jitter(kernel.gram(inducing_inputs).as_matrix(), jitter)
    cross = kernel.cross_covariance(inducing_inputs, data.X)
    diagonal = jnp.diag(kernel.gram(data.X).as_matrix())
    noise_variance = _val(posterior.likelihood.obs_stddev) ** 2
    residual = (data.y - posterior.prior.mean_function(data.X)).squeeze(-1)

    root_gram = jnp.linalg.cholesky(gram)
    design = jsp.linalg.cho_solve((root_gram, True), cross)
    nystrom = cross.T @ design
    covariance = nystrom + noise_variance * jnp.eye(data.n)
    root_covariance = jnp.linalg.cholesky(covariance)
    quadratic = jnp.sum(
        residual
        * jsp.linalg.cho_solve((root_covariance, True), residual[:, None]).squeeze(-1)
    )
    log_marginal = -0.5 * (
        data.n * jnp.log(2 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(root_covariance)))
        + quadratic
    )
    trace_term = jnp.sum(diagonal - jnp.sum(cross * design, axis=0)) / (
        2 * noise_variance
    )

    np.testing.assert_allclose(
        np.float64(dual_elbo(q, data)),
        np.float64(log_marginal - trace_term),
        atol=1e-8,
    )


def test_dual_elbo_hyper_gradients_differ_away_from_optimum():
    """Away from a converged E-step the two bounds have different theta-gradients.

    This is the regression test against caching ``(m, S)`` on the family: a cached
    implementation reproduces every *value* assertion in this module and only fails
    here.
    """
    posterior, data, inducing_inputs, jitter = _dual_setup()
    dual_vector, dual_matrix = _random_dual_sites(3, inducing_inputs.shape[0])
    q = DualVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        dual_vector=dual_vector,
        dual_matrix=dual_matrix,
        jitter=jitter,
    )

    dual_gradient = _kernel_hyper_gradient(dual_elbo, q, data)
    moment_gradient = _kernel_hyper_gradient(
        elbo, _matched_variational_gaussian(q), data
    )

    assert not jnp.allclose(dual_gradient, moment_gradient)
    relative_difference = jnp.max(jnp.abs(dual_gradient - moment_gradient)) / jnp.max(
        jnp.abs(moment_gradient)
    )
    assert relative_difference > 0.1


def test_dual_elbo_hyper_gradients_agree_at_converged_estep():
    """At a stationary q the extra term vanishes and the gradients coincide.

    The E-step really does have to be run to convergence: at 66 steps the two
    gradients agree to 1.5e-14, but at 6 steps they still differ by 1.6e-4.
    """
    posterior, data, inducing_inputs, jitter = _dual_setup(binary=True)
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    for _ in range(66):
        q = _dual_natgrad_step(q, data, 0.8)

    dual_gradient = _kernel_hyper_gradient(dual_elbo, q, data)
    moment_gradient = _kernel_hyper_gradient(
        elbo, _matched_variational_gaussian(q), data
    )
    np.testing.assert_allclose(
        np.asarray(dual_gradient), np.asarray(moment_gradient), atol=1e-8
    )


def test_dual_elbo_dominates_when_inducing_equal_inputs():
    """With Z = X the stored sites are theta-free, so the dual bound dominates.

    Adam et al.'s Eq. (29)-(31) needs theta-free exact sites, which only happens in the
    dense conjugate limit. No dominance is asserted in the sparse case, where the
    flanked sites still move with theta through K_zx.
    """
    posterior, data, inducing_inputs, jitter = _dual_setup(
        num_data=12, jitter=1e-6, inducing_at_inputs=True
    )
    q = DualVariationalGaussian(
        posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
    )
    q = _dual_natgrad_step(q, data, 1.0)
    q_moment = _matched_variational_gaussian(q)

    base_lengthscale = _val(posterior.prior.kernel.lengthscale)
    for shift in (-0.6, -0.3, 0.0, 0.3, 0.6):
        lengthscale = PositiveReal(base_lengthscale * jnp.exp(shift))
        where = lambda t: t.posterior.prior.kernel.lengthscale
        dual_value = dual_elbo(eqx.tree_at(where, q, lengthscale), data)
        moment_value = elbo(eqx.tree_at(where, q_moment, lengthscale), data)
        assert dual_value >= moment_value - 1e-5
