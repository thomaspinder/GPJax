import io

import equinox as eqx
from jax import config
import jax
import jax.numpy as jnp
import numpy as np
import paramax
import pytest
from rich.console import Console
from rich.table import Table

# Match the rest of the suite: stable float64 numerics.
config.update("jax_enable_x64", True)

import gpjax as gpx
from gpjax import summary
from gpjax.gps import Prior
from gpjax.kernels import RBF, Linear
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.parameters import (
    LowerTriangular,
    NonNegativeReal,
    PositiveReal,
    Real,
    SigmoidBounded,
)


def _frozen_prior():
    """A prior whose lengthscale has been frozen via paramax.non_trainable."""
    prior = Prior(mean_function=Zero(), kernel=RBF())
    return eqx.tree_at(
        lambda m: m.kernel.lengthscale, prior, replace_fn=paramax.non_trainable
    )


def test_paramrecord_fields_exist():
    rec = summary.ParamRecord(
        name="kernel.lengthscale",
        cls="PositiveReal",
        value=jnp.array(1.0),
        bijector="Softplus",
        prior="-",
        trainable=True,
        shape=(),
        dtype="f64",
    )
    assert rec.name == "kernel.lengthscale"
    assert rec.trainable is True
    assert rec.prior == "-"


def test_is_param_leaf_stops_at_parameters():
    assert summary._is_param_leaf(PositiveReal(1.0)) is True
    assert summary._is_param_leaf(jnp.array(1.0)) is False


def test_is_frozen_true_for_frozen_parameter():
    frozen = _frozen_prior()
    assert summary._is_frozen(frozen.kernel.lengthscale) is True
    assert summary._is_frozen(frozen.kernel.variance) is False


def test_is_frozen_false_for_plain_parameter():
    assert summary._is_frozen(PositiveReal(1.0)) is False


@pytest.mark.parametrize(
    ("param", "expected"),
    [
        (PositiveReal(2.0), "Softplus"),
        (NonNegativeReal(0.5), "Softplus"),
        (Real(-3.0), "Identity"),
        (LowerTriangular(jnp.eye(2)), "LowerCholesky"),
    ],
)
def test_bijector_name_mapping(param, expected):
    assert summary._bijector_name(param) == expected


def test_bijector_name_sigmoid_bounded_uses_bounds():
    param = SigmoidBounded(0.5, low=0.0, high=2.0)
    assert summary._bijector_name(param) == "Sigmoid[0, 2]"


def test_bijector_name_falls_back_to_identity_without_constraint():
    # A NonTrainable-wrapped bare array has no `_constraint`.
    wrapped = paramax.non_trainable(jnp.array(1.0))
    assert summary._bijector_name(wrapped) == "Identity"


def test_short_dtype_abbreviates():
    assert summary._short_dtype(jnp.float64) == "f64"
    assert summary._short_dtype(jnp.float32) == "f32"
    assert summary._short_dtype(jnp.int32) == "i32"


def test_format_value_scalar_uses_precision():
    assert summary._format_value(jnp.array(1.0), max_array=4, precision=3) == "1"
    assert (
        summary._format_value(jnp.array(0.123456), max_array=4, precision=3) == "0.123"
    )


def test_format_value_array_truncates():
    out = summary._format_value(jnp.arange(6.0), max_array=4, precision=3)
    assert out == "[0, 1, 2, 3, ...]"


def test_format_value_short_array_not_truncated():
    out = summary._format_value(jnp.array([1.0, 2.0]), max_array=4, precision=3)
    assert out == "[1, 2]"


def _record_by_name(records, name):
    matches = [r for r in records if r.name == name]
    assert matches, f"{name!r} not in {[r.name for r in records]}"
    return matches[0]


def test_collect_isotropic_rbf_prior():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    records = summary._collect(prior)
    names = {r.name for r in records}
    assert "kernel.lengthscale" in names
    assert "kernel.variance" in names

    ls = _record_by_name(records, "kernel.lengthscale")
    assert ls.cls == "PositiveReal"
    assert ls.bijector == "Softplus"
    assert ls.trainable is True
    assert ls.shape == ()
    assert ls.dtype == "f64"


def test_collect_renders_bare_array_leaf():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    rec = _record_by_name(summary._collect(prior), "mean_function.constant")
    assert rec.cls == "Array"
    assert rec.bijector == "Identity"
    assert rec.trainable is True


def test_collect_composite_kernel_paths():
    prior = Prior(mean_function=Zero(), kernel=RBF() + Linear())
    names = {r.name for r in summary._collect(prior)}
    assert "kernel.kernels[0].lengthscale" in names
    assert "kernel.kernels[1].variance" in names


def test_collect_marks_and_unwraps_frozen_parameter():
    records = summary._collect(_frozen_prior())
    ls = _record_by_name(records, "kernel.lengthscale")
    assert ls.trainable is False
    # paramax.unwrap returns a concrete constrained value even when frozen.
    assert float(np.asarray(ls.value)) > 0.0


def test_collect_variational_family_lower_cholesky():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    posterior = prior * Gaussian(num_datapoints=5)
    q = gpx.variational_families.VariationalGaussian(
        posterior=posterior, inducing_inputs=jnp.linspace(-3, 3, 5).reshape(-1, 1)
    )
    rec = _record_by_name(summary._collect(q), "variational_root_covariance")
    assert rec.cls == "LowerTriangular"
    assert rec.bijector == "LowerCholesky"
    assert rec.shape == (5, 5)


def test_format_value_traced_value_placeholder():
    captured = {}

    @jax.jit
    def f(x):
        captured["s"] = summary._format_value(x, max_array=4, precision=3)
        return x

    f(jnp.arange(3.0))
    assert captured["s"].startswith("<traced")
    assert "[3]" in captured["s"]


def test_collect_frozen_bare_array_keeps_array_class():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    frozen = eqx.tree_at(
        lambda m: m.mean_function.constant, prior, replace_fn=paramax.non_trainable
    )
    rec = _record_by_name(summary._collect(frozen), "mean_function.constant")
    assert rec.cls == "Array"
    assert rec.trainable is False
    assert rec.bijector == "Identity"
