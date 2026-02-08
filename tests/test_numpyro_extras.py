from flax import nnx
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpyro.distributions as dist
from numpyro.handlers import (
    seed,
    trace,
)

from gpjax.numpyro_extras import (
    register_parameters,
    resolve_prior,
    tree_path_to_name,
)
from gpjax.parameters import (
    LowerTriangular,
    NonNegativeReal,
    PositiveReal,
    Real,
    SigmoidBounded,
)


def valid_shapes(min_dims=0, max_dims=2):
    return st.integers(min_dims, max_dims).flatmap(
        lambda d: st.lists(st.integers(1, 3), min_size=d, max_size=d).map(tuple)
    )


def real_arrays(shape=None, min_value=None, max_value=None):
    return arrays(
        dtype=np.float64,
        shape=shape if shape is not None else valid_shapes(),
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ).map(jnp.array)


def lower_triangular_matrices(n=2):
    return arrays(
        dtype=np.float64,
        shape=(n, n),
        elements=st.floats(min_value=-2.0, max_value=2.0, width=64),
    ).map(lambda x: jnp.tril(jnp.array(x)))


class FlexibleMockModel(nnx.Module):
    def __init__(
        self,
        pos_val,
        real_val,
        non_neg_val,
        sigmoid_val,
        lower_val,
        vec_val,
        pos_prior=None,
        real_prior=None,
        non_neg_prior=None,
        sigmoid_prior=None,
        lower_prior=None,
        vec_prior=None,
    ):
        self.pos = PositiveReal(pos_val, prior=pos_prior)
        self.real = Real(real_val, prior=real_prior)
        self.non_neg = NonNegativeReal(non_neg_val, prior=non_neg_prior)
        self.sigmoid = SigmoidBounded(sigmoid_val, prior=sigmoid_prior)
        self.lower = LowerTriangular(lower_val, prior=lower_prior)
        self.vec = Real(vec_val, prior=vec_prior)


@given(
    pos_val=real_arrays(shape=(1,), min_value=1e-3, max_value=10.0),
    real_val=real_arrays(shape=(1,), min_value=-10.0, max_value=10.0),
    non_neg_val=real_arrays(shape=(1,), min_value=0.0, max_value=10.0),
    sigmoid_val=real_arrays(shape=(1,), min_value=1e-3, max_value=0.999),
    lower_val=lower_triangular_matrices(n=2),
    vec_val=real_arrays(shape=(2,), min_value=-10.0, max_value=10.0),
)
def test_no_priors_no_sampling(
    pos_val, real_val, non_neg_val, sigmoid_val, lower_val, vec_val
):
    model = FlexibleMockModel(
        pos_val, real_val, non_neg_val, sigmoid_val, lower_val, vec_val
    )

    def model_fn():
        return register_parameters(model)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    # Should be empty because no priors were attached or passed
    assert len(tr) == 0


@given(
    pos_val=real_arrays(shape=(1,), min_value=1e-3, max_value=10.0),
    real_val=real_arrays(shape=(1,), min_value=-10.0, max_value=10.0),
    non_neg_val=real_arrays(shape=(1,), min_value=0.0, max_value=10.0),
    sigmoid_val=real_arrays(shape=(1,), min_value=1e-3, max_value=0.999),
    lower_val=lower_triangular_matrices(n=2),
    vec_val=real_arrays(shape=(2,), min_value=-10.0, max_value=10.0),
)
def test_explicit_priors_sampling(
    pos_val, real_val, non_neg_val, sigmoid_val, lower_val, vec_val
):
    model = FlexibleMockModel(
        pos_val, real_val, non_neg_val, sigmoid_val, lower_val, vec_val
    )

    # Define priors compatible with shapes
    priors = {
        "pos": dist.LogNormal(0.0, 1.0).expand(pos_val.shape).to_event(pos_val.ndim),
        "real": dist.Normal(0.0, 1.0).expand(real_val.shape).to_event(real_val.ndim),
        "non_neg": dist.LogNormal(0.0, 1.0)
        .expand(non_neg_val.shape)
        .to_event(non_neg_val.ndim),
        "sigmoid": dist.Uniform(0.0, 1.0)
        .expand(sigmoid_val.shape)
        .to_event(sigmoid_val.ndim),
        # For LowerTriangular, user must provide a prior over the full matrix shape
        # OR a transformed prior. Here we simulate providing a prior over the full shape
        # just to ensure the site is registered.
        "lower": dist.Normal(0.0, 1.0).expand(lower_val.shape).to_event(lower_val.ndim),
        "vec": dist.Normal(0.0, 1.0).expand(vec_val.shape).to_event(vec_val.ndim),
    }

    def model_fn():
        return register_parameters(model, priors=priors)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    assert "pos" in tr
    assert "real" in tr
    assert "non_neg" in tr
    assert "sigmoid" in tr
    assert "lower" in tr
    assert "vec" in tr


@given(
    pos_val=real_arrays(shape=(1,), min_value=1e-3, max_value=10.0),
    real_val=real_arrays(shape=(1,), min_value=-10.0, max_value=10.0),
    non_neg_val=real_arrays(shape=(1,), min_value=0.0, max_value=10.0),
    sigmoid_val=real_arrays(shape=(1,), min_value=1e-3, max_value=0.999),
    lower_val=lower_triangular_matrices(n=2),
    vec_val=real_arrays(shape=(2,), min_value=-10.0, max_value=10.0),
)
def test_attached_priors_sampling(
    pos_val, real_val, non_neg_val, sigmoid_val, lower_val, vec_val
):
    # Create priors
    pos_prior = dist.LogNormal(0.0, 1.0).expand(pos_val.shape).to_event(pos_val.ndim)
    real_prior = dist.Normal(0.0, 1.0).expand(real_val.shape).to_event(real_val.ndim)
    # Attach only to a subset to verify mixed behavior
    model = FlexibleMockModel(
        pos_val,
        real_val,
        non_neg_val,
        sigmoid_val,
        lower_val,
        vec_val,
        pos_prior=pos_prior,
        real_prior=real_prior,
    )

    def model_fn():
        return register_parameters(model)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    assert "pos" in tr
    assert "real" in tr
    assert "non_neg" not in tr
    assert "vec" not in tr


@given(
    pos_val=real_arrays(shape=(1,), min_value=1e-3, max_value=10.0),
)
def test_prior_precedence(pos_val):
    # Attached prior
    attached_prior = dist.Gamma(2.0, 1.0).expand(pos_val.shape).to_event(pos_val.ndim)

    # Explicit prior (different)
    explicit_prior = dist.Exponential(1.0).expand(pos_val.shape).to_event(pos_val.ndim)

    # Model with attached prior
    # We need dummy values for others
    dummy_real = jnp.array([0.0])
    dummy_lower = jnp.eye(2)
    dummy_vec = jnp.zeros(2)

    model = FlexibleMockModel(
        pos_val,
        dummy_real,
        dummy_real,
        dummy_real,
        dummy_lower,
        dummy_vec,
        pos_prior=attached_prior,
    )

    priors = {"pos": explicit_prior}

    def model_fn():
        return register_parameters(model, priors=priors)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    # Check that the sampled site corresponds to the explicit prior
    # We can check the distribution object
    # Structure might be Independent(Expanded(Exponential)) or Independent(Exponential)
    d = tr["pos"]["fn"]
    while hasattr(d, "base_dist"):
        d = d.base_dist
    assert isinstance(d, dist.Exponential)


def test_register_parameters_nested_prefix():
    class NestedModel(nnx.Module):
        def __init__(self):
            self.inner = FlexibleMockModel(
                jnp.array([1.0]),
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                jnp.eye(2),
                jnp.zeros(2),
            )

    model = NestedModel()
    # Explicit prior for nested
    priors = {"outer.inner.pos": dist.LogNormal(0.0, 1.0).expand((1,)).to_event(1)}

    def model_fn():
        return register_parameters(model, prefix="outer", priors=priors)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    assert "outer.inner.pos" in tr
    assert "outer.inner.real" not in tr


def test_tree_path_to_name_with_getattr_key():
    path = (jtu.GetAttrKey("kernel"), jtu.GetAttrKey("lengthscale"))
    assert tree_path_to_name(path) == "kernel.lengthscale"


def test_tree_path_to_name_with_dict_key():
    path = (jtu.DictKey(key="params"), jtu.DictKey(key="variance"))
    assert tree_path_to_name(path) == "params.variance"


def test_tree_path_to_name_with_sequence_key():
    path = (jtu.GetAttrKey("layers"), jtu.SequenceKey(idx=0), jtu.GetAttrKey("weight"))
    assert tree_path_to_name(path) == "layers.0.weight"


def test_tree_path_to_name_with_prefix():
    path = (jtu.GetAttrKey("kernel"), jtu.GetAttrKey("variance"))
    assert tree_path_to_name(path, prefix="model") == "model.kernel.variance"


def test_tree_path_to_name_empty_path():
    path = ()
    assert tree_path_to_name(path) == ""


def test_tree_path_to_name_empty_path_with_prefix():
    path = ()
    assert tree_path_to_name(path, prefix="model") == "model."


def test_tree_path_to_name_mixed_keys():
    path = (
        jtu.GetAttrKey("nested"),
        jtu.DictKey(key="sub"),
        jtu.SequenceKey(idx=2),
    )
    assert tree_path_to_name(path) == "nested.sub.2"


def test_resolve_prior_explicit_takes_precedence():
    explicit_prior = dist.Normal(0.0, 1.0)
    attached_prior = dist.Gamma(1.0, 1.0)
    param = PositiveReal(jnp.array([1.0]), prior=attached_prior)
    priors = {"my_param": explicit_prior}

    result = resolve_prior("my_param", param, priors)
    assert result is explicit_prior


def test_resolve_prior_falls_back_to_attached():
    attached_prior = dist.LogNormal(0.0, 1.0)
    param = PositiveReal(jnp.array([1.0]), prior=attached_prior)
    priors = {}

    result = resolve_prior("my_param", param, priors)
    assert result is attached_prior


def test_resolve_prior_returns_none_when_no_prior():
    param = Real(jnp.array([0.0]))
    priors = {}

    result = resolve_prior("my_param", param, priors)
    assert result is None


def test_resolve_prior_explicit_for_different_name_no_attached():
    explicit_prior = dist.Normal(0.0, 1.0)
    param = Real(jnp.array([0.0]))
    priors = {"other_param": explicit_prior}

    result = resolve_prior("my_param", param, priors)
    assert result is None


def test_register_parameters_conjugate_posterior():
    """Integration test: register_parameters on a real ConjugatePosterior.

    Verifies that:
    - Nested modules (kernel, likelihood) are traversed correctly
    - Shared nnx.Variable references (lengthscale shared between RBF and Periodic)
      result in a single sample site
    - nnx.List inside CombinationKernel is traversed properly
    - All parameters with priors are sampled
    - conjugate_mll can be evaluated with the sampled parameters
    """
    import gpjax as gpx

    lengthscale_prior = dist.LogNormal(0.0, 1.0)
    variance_prior = dist.LogNormal(0.0, 1.0)
    period_prior = dist.LogNormal(0.0, 0.5)
    noise_prior = dist.LogNormal(0.0, 1.0)

    lengthscale = PositiveReal(1.0, prior=lengthscale_prior)
    variance = PositiveReal(1.0, prior=variance_prior)
    period = PositiveReal(1.0, prior=period_prior)
    noise = NonNegativeReal(1.0, prior=noise_prior)

    rbf = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)
    periodic = gpx.kernels.Periodic(lengthscale=lengthscale, period=period)
    kernel = rbf * periodic

    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=10, obs_stddev=noise)
    posterior = prior * likelihood

    def model_fn():
        return register_parameters(posterior)

    with seed(rng_seed=42):
        tr = trace(model_fn).get_trace()

    # Shared lengthscale should appear once (shared Variable → single site)
    lengthscale_sites = [k for k in tr if "lengthscale" in k]
    assert len(lengthscale_sites) == 1, (
        f"Expected 1 lengthscale site, got {len(lengthscale_sites)}: {lengthscale_sites}"
    )

    # variance, period, and obs_stddev should each appear once
    variance_sites = [k for k in tr if "variance" in k]
    assert len(variance_sites) == 1

    period_sites = [k for k in tr if "period" in k]
    assert len(period_sites) == 1

    obs_stddev_sites = [k for k in tr if "obs_stddev" in k]
    assert len(obs_stddev_sites) == 1

    # Total: 4 sampled sites (lengthscale, variance, period, obs_stddev)
    assert len(tr) == 4, f"Expected 4 sample sites, got {len(tr)}: {list(tr.keys())}"


def test_register_parameters_conjugate_posterior_mll():
    """Integration test: sampled parameters flow through conjugate_mll."""
    import jax.random as jr

    import gpjax as gpx

    key = jr.key(0)
    X = jr.uniform(key, shape=(10, 1))
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    lengthscale = PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))
    variance = PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))
    noise = NonNegativeReal(0.5, prior=dist.LogNormal(0.0, 1.0))

    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=10, obs_stddev=noise)
    posterior = prior * likelihood

    mll_value = None

    def model_fn():
        nonlocal mll_value
        p = register_parameters(posterior)
        mll_value = gpx.objectives.conjugate_mll(p, D)
        return mll_value

    with seed(rng_seed=42):
        tr = trace(model_fn).get_trace()

    assert len(tr) == 3  # lengthscale, variance, obs_stddev
    assert mll_value is not None
    assert jnp.isfinite(mll_value)
