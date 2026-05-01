"""Shared fixtures and helpers for ASV benchmarks.

The contract every benchmark in this directory follows:

1. Float64 is enabled globally here (mirroring tests/conftest.py).
2. Every timed call must end with realise(result) so the GPU/dispatcher
   actually completes the work before the timer stops. JAX is async;
   without this we time dispatch, not work.
3. No conftest.py exists in this directory — beartype and hypothesis
   must NOT activate during benchmark runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


KEY = jr.key(0)


def realise(result):
    """Block until every JAX array leaf in ``result`` has finished computing.

    Pass anything: a JAX array, a tuple, a dict, an Equinox Module — anything
    a JAX PyTree can hold. Non-array leaves are ignored.
    """
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        result,
    )
    return result


def make_inputs(n: int, d: int = 1, key=KEY) -> jnp.ndarray:
    """Standard input grid: uniform on [0, 1]^d, shape (n, d)."""
    return jr.uniform(key, (n, d), dtype=jnp.float64)


def make_outputs(n: int, q: int = 1, key=KEY) -> jnp.ndarray:
    """Standard output array, shape (n, q)."""
    return jr.normal(key, (n, q), dtype=jnp.float64)
