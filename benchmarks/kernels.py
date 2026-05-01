"""Kernel micro-benchmarks: gram, cross_covariance, diagonal."""

from __future__ import annotations

import gpjax as gpx
from gpjax.kernels.approximations import RFF
from gpjax.kernels.multioutput.icm import ICMKernel
from gpjax.kernels.multioutput.lcm import LCMKernel
from gpjax.parameters import CoregionalizationMatrix
import jax.random as jr

from benchmarks._setup import ALIGNED_NS, make_inputs, realise

_KERNELS = {
    "RBF": gpx.kernels.RBF,
    "Matern12": gpx.kernels.Matern12,
    "Matern32": gpx.kernels.Matern32,
    "Matern52": gpx.kernels.Matern52,
    "Periodic": gpx.kernels.Periodic,
    "Linear": gpx.kernels.Linear,
    "Polynomial": gpx.kernels.Polynomial,
}

# Number of RFF basis functions. Larger M -> closer to exact, slower.
# 128 is a common practical default for D=1 stationary kernels.
_RFF_NUM_BASIS = 128


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


class RffGramSuite:
    """Gram cost: exact stationary kernel vs RFF approximation at matched N.

    Aligns with ALIGNED_NS so the dashboard shows the RFF/exact crossover
    on the same N axis as the sparse-vs-full objective comparison.
    """

    _BASE = ("RBF", "Matern52")
    _APPROX = ("exact", "rff")
    params = (list(_BASE), list(_APPROX), list(ALIGNED_NS))
    param_names = ("base", "approx", "n")

    def setup(self, base, approx, n):
        self.X = make_inputs(n, d=1)
        base_kernel = _KERNELS[base](n_dims=1)
        if approx == "exact":
            self.kernel = base_kernel
        else:
            self.kernel = RFF(
                base_kernel=base_kernel,
                num_basis_fns=_RFF_NUM_BASIS,
                key=jr.key(0),
            )
        realise(self.kernel.gram(self.X).as_matrix())

    def time_gram(self, base, approx, n):
        realise(self.kernel.gram(self.X).as_matrix())


class MultiOutputGramSuite:
    """Multi-output gram realisation: ICM vs LCM at varying latent rank Q.

    ICM (Q=1) keeps Kronecker structure; LCM with Q>=2 falls back to a
    materialised dense N*P x N*P matrix. Calling .as_matrix() makes the
    structural difference visible — the cost users pay when they solve.
    """

    # ("ICM", None) means the single-component Kronecker path.
    # ("LCM", Q) means an LCM with Q latent kernels (dense fallback for Q>=2).
    _MODELS = (("ICM", None), ("LCM", 2), ("LCM", 4))
    params = ([f"{kind}_Q{q or 1}" for kind, q in _MODELS], [200])
    param_names = ("model", "n")

    def setup(self, model, n):
        kind, q_str = model.split("_Q")
        Q = int(q_str)
        P = 4
        self.X = make_inputs(n, d=1)
        if kind == "ICM":
            coreg = CoregionalizationMatrix(num_outputs=P, rank=2, key=jr.key(11))
            self.kernel = ICMKernel(
                base_kernel=gpx.kernels.RBF(), coregionalization_matrix=coreg
            )
        else:
            kernels = [gpx.kernels.RBF() for _ in range(Q)]
            mats = [
                CoregionalizationMatrix(num_outputs=P, rank=2, key=jr.key(20 + i))
                for i in range(Q)
            ]
            self.kernel = LCMKernel(kernels=kernels, coregionalization_matrices=mats)
        realise(self.kernel.gram(self.X).as_matrix())

    def time_gram(self, model, n):
        realise(self.kernel.gram(self.X).as_matrix())
