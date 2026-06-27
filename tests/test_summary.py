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
