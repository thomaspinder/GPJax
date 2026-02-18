"""Tests for the LowRank linear operator."""

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt

config.update("jax_enable_x64", True)

from gpjax.linalg.operations import diag
from gpjax.linalg.operators import LowRank


class TestLowRankProperties:
    def test_shape(self):
        W = jnp.ones((10, 3))
        op = LowRank(W)
        assert op.shape == (10, 10)

    def test_rank(self):
        W = jnp.ones((10, 3))
        op = LowRank(W)
        assert op.rank == 3

    def test_dtype(self):
        W = jnp.ones((10, 3), dtype=jnp.float64)
        op = LowRank(W)
        assert op.dtype == jnp.float64

    def test_to_dense(self):
        key = jr.key(0)
        W = jr.normal(key, (10, 3))
        op = LowRank(W)
        npt.assert_allclose(op.to_dense(), W @ W.T, atol=1e-12)

    def test_transpose_is_self(self):
        W = jnp.ones((10, 3))
        op = LowRank(W)
        assert op.T is op

    def test_diag_dispatch(self):
        key = jr.key(0)
        W = jr.normal(key, (10, 3))
        op = LowRank(W)
        expected = jnp.sum(W**2, axis=1)
        npt.assert_allclose(diag(op), expected, atol=1e-12)


class TestLowRankPyTree:
    def test_flatten_unflatten_roundtrip(self):
        key = jr.key(0)
        W = jr.normal(key, (10, 3))
        op = LowRank(W)
        leaves, treedef = jax.tree.flatten(op)
        restored = treedef.unflatten(leaves)
        npt.assert_allclose(restored.factor, op.factor)

    def test_jit_compatible(self):
        key = jr.key(0)
        W = jr.normal(key, (10, 3))
        op = LowRank(W)

        @jax.jit
        def fn(op):
            return op.to_dense()

        result = fn(op)
        npt.assert_allclose(result, W @ W.T, atol=1e-12)

    def test_grad_through_to_dense(self):
        key = jr.key(0)
        W = jr.normal(key, (10, 3))

        @jax.grad
        def fn(W):
            op = LowRank(W)
            return jnp.sum(op.to_dense())

        grads = fn(W)
        assert jnp.all(jnp.isfinite(grads))
