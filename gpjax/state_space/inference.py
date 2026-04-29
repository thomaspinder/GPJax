"""Square-root Kalman filter and RTS smoother for state-space GPs.

See plans/2026-04-21-state-space-gps-design.md §Stage 2 and §Stage 3.
"""

from __future__ import annotations

import jax.numpy as jnp


def _sign_normalise(R):
    """Flip row signs so the diagonal of R is non-negative without changing R.T @ R.

    `jnp.linalg.qr` does not guarantee sign on the R-factor diagonal (LAPACK's
    `geqrf` returns arbitrary signs). This breaks square-root Kalman updates,
    where ``R[0, 0]`` is the innovation standard deviation.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2 (`_sign_normalise`).
    """
    row_signs = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
    return row_signs[:, None] * R
