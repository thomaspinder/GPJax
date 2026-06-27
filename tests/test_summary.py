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
