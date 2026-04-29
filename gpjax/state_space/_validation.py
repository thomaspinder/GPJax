"""Validation helpers for state-space kernels and datasets."""

from __future__ import annotations

import jax.numpy as jnp
import paramax


def _validate_temporal_kernel(kernel) -> None:
    """Raise ``ValueError`` unless ``kernel`` is configured for a 1-D temporal input.

    State-space GPs require kernels with scalar lengthscale, no active_dims
    selection (or trivial ``slice(None)``), and ``n_dims`` either ``None`` or 1.
    """
    lengthscale = paramax.unwrap(kernel.lengthscale)
    if jnp.asarray(lengthscale).ndim != 0:
        raise ValueError(
            "State-space kernels require a scalar lengthscale (1-D temporal input). "
            f"Got lengthscale of shape {jnp.asarray(lengthscale).shape}."
        )

    n_dims = getattr(kernel, "n_dims", None)
    if n_dims is not None and n_dims != 1:
        raise ValueError(
            "State-space kernels require n_dims == 1 (1-D temporal input). "
            f"Got n_dims={n_dims}."
        )

    active_dims = getattr(kernel, "active_dims", None)
    if isinstance(active_dims, list) and len(active_dims) > 1:
        raise ValueError(
            "State-space kernels require single-axis active_dims (1-D temporal input). "
            f"Got active_dims={active_dims}."
        )
