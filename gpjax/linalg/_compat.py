"""Deprecated wrappers mapping old GPJax linalg types to Lineax equivalents."""

import warnings

import jax
import jax.numpy as jnp
import lineax as lx


def Dense(array):
    warnings.warn(
        "gpjax.linalg.Dense is deprecated, use lineax.MatrixLinearOperator directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lx.MatrixLinearOperator(array)


def Diagonal(diagonal):
    warnings.warn(
        "gpjax.linalg.Diagonal is deprecated, use lineax.DiagonalLinearOperator directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lx.DiagonalLinearOperator(diagonal)


def Identity(shape, dtype=jnp.float64):
    warnings.warn(
        "gpjax.linalg.Identity is deprecated, use lineax.IdentityLinearOperator directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, tuple) and len(shape) == 2:
        shape = (shape[0],)
    return lx.IdentityLinearOperator(jax.ShapeDtypeStruct(shape, dtype))


def Triangular(array, lower=True):
    warnings.warn(
        "gpjax.linalg.Triangular is deprecated, use lineax.MatrixLinearOperator with tags.",
        DeprecationWarning,
        stacklevel=2,
    )
    tag = lx.lower_triangular_tag if lower else lx.upper_triangular_tag
    return lx.MatrixLinearOperator(array, tags=tag)
