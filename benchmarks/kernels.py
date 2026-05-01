"""Kernel micro-benchmarks (gram, cross_covariance, diagonal).

Coverage is intentionally minimal in this commit. Full kernel/N/D matrix
is added in a later task once the plumbing is verified.
"""

from __future__ import annotations

import gpjax as gpx

from benchmarks._setup import make_inputs, realise


class GramSuite:
    """Cost of K(X, X) for representative kernels."""

    params = ([100],)
    param_names = ("n",)

    def setup(self, n):
        self.X = make_inputs(n)
        self.kernel = gpx.kernels.RBF()
        # Warm: trigger any one-time tracing so time_gram measures
        # warmed steady-state, not first-call compile.
        realise(self.kernel.gram(self.X))

    def time_gram(self, n):
        realise(self.kernel.gram(self.X))
