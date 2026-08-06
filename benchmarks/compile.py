"""JIT compile-time tracking.

Unlike time_* benchmarks (which measure warmed steady-state), these
benchmarks measure the *first* call of a freshly-jitted function - the
compile cost. Each iteration:

  setup() -> jax.clear_caches() and rebuild the jitted function
  track_compile_*() -> time the first call

Because each setup creates a new function object, JAX has nothing cached
for it; the first call inside track_compile_* triggers tracing+lowering+
compilation.

The conjugate_mll vs elbo pair is the headline comparison here: variational
posteriors carry more pytree structure through the trace, so the compile
gap (not just steady-state cost) matters for iteration speed.
"""

from __future__ import annotations

import time

import gpjax as gpx
from gpjax import objectives
from gpjax.variational_families import VariationalGaussian
import jax

from benchmarks._setup import M_INDUCING, make_inputs, make_outputs, realise


class CompileSuite:
    def setup(self):
        jax.clear_caches()

        n = 200
        X = make_inputs(n)
        y = make_outputs(n)
        self.data = gpx.Dataset(X=X, y=y)
        kernel = gpx.kernels.RBF()
        mean = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(kernel=kernel, mean_function=mean)
        likelihood = gpx.likelihoods.Gaussian()
        self.posterior = prior * likelihood
        self.q = VariationalGaussian(
            posterior=self.posterior, inducing_inputs=X[:M_INDUCING]
        )

        self.jitted_mll = jax.jit(objectives.conjugate_mll)
        self.jitted_elbo = jax.jit(objectives.elbo)
        self.jitted_gram = jax.jit(lambda X: kernel.gram(X).as_matrix())
        self.X_warm = X

    def track_compile_conjugate_mll(self):
        t0 = time.perf_counter()
        realise(self.jitted_mll(self.posterior, self.data))
        return time.perf_counter() - t0

    track_compile_conjugate_mll.unit = "seconds"

    def track_compile_elbo(self):
        t0 = time.perf_counter()
        realise(self.jitted_elbo(self.q, self.data))
        return time.perf_counter() - t0

    track_compile_elbo.unit = "seconds"

    def track_compile_kernel_gram(self):
        t0 = time.perf_counter()
        realise(self.jitted_gram(self.X_warm))
        return time.perf_counter() - t0

    track_compile_kernel_gram.unit = "seconds"
