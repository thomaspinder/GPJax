"""Tests for the Orthogonal Additive Kernel."""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
