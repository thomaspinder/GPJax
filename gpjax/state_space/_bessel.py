"""Numerically stable scaled modified Bessel functions for state-space GPs.

Leaf module imported by both `sde.py` and `kernels.py`.

See plans/2026-04-21-state-space-gps-design.md §Bessel stability.

Three regimes are dispatched via nested `jax.lax.cond` so that only the taken
branch contributes under `jit` and `grad`:

- `c > truncation_order`        : forward recurrence seeded by `i0e`/`i1e`
- `1e-8 <= c <= truncation_order` : Miller's downward recurrence rescaled by
  `i0e(c)`
- `c < 1e-8`                    : power-series ``Ĩ_k(c) ≈ (c/2)^k / k! · e^{-c}``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


def _large_c_scaled_ive(c, truncation_order: int):
    """Forward recurrence for Ĩ_k(c) = e^{-c} I_k(c), seeded by i0e/i1e.

    The recurrence

        Ĩ_{k+1}(c) = Ĩ_{k-1}(c) - (2k/c) Ĩ_k(c)

    inherits stability from the unscaled identity since the e^{-c} factor
    cancels. For ``c > truncation_order`` the values ``I_k(c)`` are comparable
    in magnitude across ``k``, so upward (forward) recurrence is numerically
    stable here; downward (Miller) recurrence is reserved for
    ``c <= truncation_order``, where ``I_k(c)`` decays in ``k`` and forward
    recurrence would amplify round-off.
    """
    c = jnp.asarray(c)
    bessel_at_order_0 = jsp.i0e(c)
    bessel_at_order_1 = jsp.i1e(c)

    if truncation_order == 0:
        return jnp.stack([bessel_at_order_0])
    if truncation_order == 1:
        return jnp.stack([bessel_at_order_0, bessel_at_order_1])

    def recurrence_step(carry, current_order):
        bessel_at_lower, bessel_at_current = carry
        bessel_at_upper = (
            bessel_at_lower - (2.0 * current_order / c) * bessel_at_current
        )
        return (bessel_at_current, bessel_at_upper), bessel_at_upper

    middle_orders = jnp.arange(1, truncation_order)
    _, higher_order_values = jax.lax.scan(
        recurrence_step,
        (bessel_at_order_0, bessel_at_order_1),
        middle_orders,
    )
    return jnp.concatenate(
        [jnp.stack([bessel_at_order_0, bessel_at_order_1]), higher_order_values]
    )


def _miller_downward_scaled_ive(c, truncation_order: int):
    """Miller's downward recurrence for Ĩ_k(c), rescaled by i0e(c).

    Start at `k_max = 2 * truncation_order + 20` with seeds
    (ratio_{k_max} = 0, ratio_{k_max-1} = 1); apply downward recurrence
        ratio_{k-1} = (2k / c) * ratio_k + ratio_{k+1}
    until ratio_0; rescale all ratios by the exact Ĩ_0(c) = i0e(c) / ratio_0.
    `k_max` is static (compile-time) so the loop unrolls under jit; `c` stays
    traced.
    """
    c = jnp.asarray(c)
    k_max = 2 * truncation_order + 20

    def body(k_iteration, state):
        ratios, ratio_k_plus_1, ratio_k = state
        # Downward index: k_iteration 1 → current k = k_max-1; recurrence
        # computes ratio at (current_k - 1) from ratio_k, ratio_{k+1}.
        current_k = k_max - k_iteration
        ratio_k_minus_1 = (2.0 * current_k / c) * ratio_k + ratio_k_plus_1
        ratios = ratios.at[current_k - 1].set(ratio_k_minus_1)
        return (ratios, ratio_k, ratio_k_minus_1)

    initial_ratios = jnp.zeros(k_max + 1, dtype=c.dtype)
    initial_ratios = initial_ratios.at[k_max].set(jnp.asarray(0.0, c.dtype))
    initial_ratios = initial_ratios.at[k_max - 1].set(jnp.asarray(1.0, c.dtype))
    ratios, _, _ = jax.lax.fori_loop(
        1,
        k_max,
        body,
        (initial_ratios, jnp.asarray(0.0, c.dtype), jnp.asarray(1.0, c.dtype)),
    )
    rescale = jsp.i0e(c) / ratios[0]
    return ratios[: truncation_order + 1] * rescale


def _tiny_c_series_scaled_ive(c, truncation_order: int):
    """Power-series Ĩ_k(c) ≈ (c/2)^k / k! · e^{-c}. Exact for c → 0."""
    c = jnp.asarray(c)
    bessel_orders = jnp.arange(truncation_order + 1)
    log_factorial = jsp.gammaln(bessel_orders + 1.0)
    # Use log-space to avoid underflow at k_max.
    return jnp.exp(bessel_orders * jnp.log(c / 2.0) - log_factorial - c)


def _stable_scaled_ive(c, truncation_order: int):
    """Compute Ĩ_k(c) = e^{-c} I_k(c) for k = 0, …, truncation_order.

    Three regimes, selected via nested `jax.lax.cond` so only the taken branch
    contributes under jit and grad:

    - ``c > truncation_order``         : forward recurrence seeded by i0e/i1e
    - ``1e-8 <= c <= truncation_order`` : Miller downward recurrence + i0e rescale
    - ``c < 1e-8``                     : power-series Ĩ_k(c) ≈ (c/2)^k/k! · e^{-c}

    `truncation_order` is static (Python int), so both branch closures compile
    to fixed-shape computations.
    """
    c = jnp.asarray(c)

    def large_branch(c_):
        return _large_c_scaled_ive(c_, truncation_order)

    def small_branch(c_):
        return jax.lax.cond(
            c_ <= 1e-8,
            lambda cc: _tiny_c_series_scaled_ive(cc, truncation_order),
            lambda cc: _miller_downward_scaled_ive(cc, truncation_order),
            c_,
        )

    return jax.lax.cond(c > float(truncation_order), large_branch, small_branch, c)
