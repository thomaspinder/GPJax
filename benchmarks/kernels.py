"""Kernel micro-benchmarks: gram, cross_covariance, diagonal."""

from __future__ import annotations

import gpjax as gpx
import jax.random as jr

from benchmarks._setup import make_inputs, realise

_KERNELS = {
    "RBF": gpx.kernels.RBF,
    "Matern12": gpx.kernels.Matern12,
    "Matern32": gpx.kernels.Matern32,
    "Matern52": gpx.kernels.Matern52,
    "Periodic": gpx.kernels.Periodic,
    "Linear": gpx.kernels.Linear,
    "Polynomial": gpx.kernels.Polynomial,
}


class GramSuite:
    """K(X, X) over kernel x N x D."""

    params = (
        list(_KERNELS.keys()),
        [100, 500, 2000],
        [1, 5],
    )
    param_names = ("kernel", "n", "d")

    def setup(self, kernel, n, d):
        self.X = make_inputs(n, d)
        self.kernel = _KERNELS[kernel]()
        realise(self.kernel.gram(self.X))  # warm

    def time_gram(self, kernel, n, d):
        realise(self.kernel.gram(self.X))


class CrossCovarianceSuite:
    """K(X, Y) for representative kernels."""

    params = (["RBF", "Matern52"], [100, 500])
    param_names = ("kernel", "n")

    def setup(self, kernel, n):
        self.X = make_inputs(n, key=jr.key(1))
        self.Y = make_inputs(n, key=jr.key(2))
        self.kernel = _KERNELS[kernel]()
        realise(self.kernel.cross_covariance(self.X, self.Y))

    def time_cross_covariance(self, kernel, n):
        realise(self.kernel.cross_covariance(self.X, self.Y))


class DiagonalSuite:
    """diag K(X) - should be O(N), not O(N^2)."""

    params = (["RBF", "Matern52"], [100, 2000])
    param_names = ("kernel", "n")

    def setup(self, kernel, n):
        self.X = make_inputs(n)
        self.kernel = _KERNELS[kernel]()
        realise(self.kernel.diagonal(self.X))

    def time_diagonal(self, kernel, n):
        realise(self.kernel.diagonal(self.X))
