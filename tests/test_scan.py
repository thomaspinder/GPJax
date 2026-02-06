"""Tests for gpjax.scan module (vscan)."""

from gpjax.scan import vscan
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


def _cumsum_fn(carry, x):
    """Simple scan function: cumulative sum."""
    carry = carry + x
    return carry, carry


class TestVscan:
    """Tests for the vscan function."""

    def test_basic_cumulative_sum(self):
        """vscan should produce same results as jax.lax.scan."""
        xs = jnp.arange(10)
        carry_v, ys_v = vscan(_cumsum_fn, 0, xs)
        carry_l, ys_l = jax.lax.scan(_cumsum_fn, 0, xs)

        assert jnp.allclose(carry_v, carry_l)
        assert jnp.allclose(ys_v, ys_l)

    def test_output_shape(self):
        """Output arrays should have shape matching input length."""
        xs = jnp.ones(20)
        carry, ys = vscan(_cumsum_fn, 0.0, xs)

        assert ys.shape == (20,)
        assert carry.shape == ()

    def test_reverse(self):
        """reverse=True should scan inputs in reverse order."""
        xs = jnp.arange(5, dtype=jnp.float32)
        _, _ys_fwd = vscan(_cumsum_fn, 0.0, xs, reverse=False, log_rate=1)
        _, ys_rev = vscan(_cumsum_fn, 0.0, xs, reverse=True, log_rate=1)

        # Compare against jax.lax.scan for correctness
        _, ys_rev_ref = jax.lax.scan(_cumsum_fn, 0.0, xs, reverse=True)
        assert jnp.allclose(ys_rev, ys_rev_ref)

    def test_unroll(self):
        """unroll parameter should not change results."""
        xs = jnp.arange(12, dtype=jnp.float32)
        _, ys_1 = vscan(_cumsum_fn, 0.0, xs, unroll=1)
        _, ys_4 = vscan(_cumsum_fn, 0.0, xs, unroll=4)

        assert jnp.allclose(ys_1, ys_4)

    @pytest.mark.parametrize("log_rate", [1, 5, 10, 50])
    def test_log_rate(self, log_rate: int):
        """Different log_rate values should not change results."""
        xs = jnp.arange(100, dtype=jnp.float32)
        carry, ys = vscan(_cumsum_fn, 0.0, xs, log_rate=log_rate)
        carry_ref, ys_ref = jax.lax.scan(_cumsum_fn, 0.0, xs)

        assert jnp.allclose(carry, carry_ref)
        assert jnp.allclose(ys, ys_ref)

    def test_log_value_false(self):
        """log_value=False should not change results."""
        xs = jnp.arange(10, dtype=jnp.float32)
        carry, ys = vscan(_cumsum_fn, 0.0, xs, log_value=False)
        carry_ref, ys_ref = jax.lax.scan(_cumsum_fn, 0.0, xs)

        assert jnp.allclose(carry, carry_ref)
        assert jnp.allclose(ys, ys_ref)

    def test_pytree_inputs(self):
        """vscan should work with pytree inputs and carries."""

        def fn(carry, x):
            new_carry = {"sum": carry["sum"] + x["val"], "count": carry["count"] + 1}
            return new_carry, carry["sum"] + x["val"]

        xs = {"val": jnp.arange(5, dtype=jnp.float32)}
        init = {"sum": jnp.float32(0.0), "count": jnp.int32(0)}

        carry, ys = vscan(fn, init, xs)
        assert carry["count"] == 5
        assert jnp.allclose(carry["sum"], 10.0)
        assert ys.shape == (5,)

    def test_single_element(self):
        """vscan should handle single-element input."""
        xs = jnp.array([42.0])
        carry, ys = vscan(_cumsum_fn, 0.0, xs, log_rate=1)

        assert jnp.allclose(carry, 42.0)
        assert ys.shape == (1,)

    def test_multidimensional_carry_and_output(self):
        """vscan should work with multidimensional carry and output."""

        def fn(carry, x):
            new_carry = carry + x
            return new_carry, new_carry

        xs = jr.normal(jr.key(0), (10, 3))
        init = jnp.zeros(3)
        # log_value=False because the tqdm postfix formatter can't format arrays
        carry, ys = vscan(fn, init, xs, log_value=False)

        carry_ref, ys_ref = jax.lax.scan(fn, init, xs)
        assert jnp.allclose(carry, carry_ref)
        assert jnp.allclose(ys, ys_ref)
        assert ys.shape == (10, 3)

    def test_matches_lax_scan_with_length(self):
        """Explicit length parameter should work correctly."""
        xs = jnp.arange(10, dtype=jnp.float32)
        carry, ys = vscan(_cumsum_fn, 0.0, xs, length=10)
        carry_ref, ys_ref = jax.lax.scan(_cumsum_fn, 0.0, xs, length=10)

        assert jnp.allclose(carry, carry_ref)
        assert jnp.allclose(ys, ys_ref)
