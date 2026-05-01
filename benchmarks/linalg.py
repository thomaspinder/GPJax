"""Linalg micro-benchmarks: Cholesky, logdet, jitter, structured operators."""

from __future__ import annotations

from gpjax.linalg import BlockDiag, Kronecker, add_jitter, cholesky_factor, logdet
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from benchmarks._setup import KEY, realise


def _spd(n: int, key=KEY) -> jnp.ndarray:
    """A well-conditioned SPD matrix of shape (n, n)."""
    A = jr.normal(key, (n, n), dtype=jnp.float64)
    return A @ A.T + n * jnp.eye(n, dtype=jnp.float64)


class CholeskySuite:
    params = ([100, 500, 2000],)
    param_names = ("n",)

    def setup(self, n):
        self.K = lx.MatrixLinearOperator(_spd(n), tags=lx.positive_semidefinite_tag)
        realise(cholesky_factor(self.K).as_matrix())

    def time_cholesky_factor(self, n):
        realise(cholesky_factor(self.K).as_matrix())


class LogdetSuite:
    params = ([100, 500, 2000],)
    param_names = ("n",)

    def setup(self, n):
        self.K = lx.MatrixLinearOperator(_spd(n), tags=lx.positive_semidefinite_tag)
        realise(logdet(self.K))

    def time_logdet(self, n):
        realise(logdet(self.K))


class AddJitterSuite:
    params = ([100, 2000],)
    param_names = ("n",)

    def setup(self, n):
        self.K = _spd(n)

    def time_add_jitter(self, n):
        realise(add_jitter(self.K))


class BlockDiagMvSuite:
    """Matches the operator shape from PR #634."""

    params = ([(8, 64), (16, 128)],)
    param_names = ("blocks_x_size",)

    def setup(self, blocks_x_size):
        n_blocks, block_size = blocks_x_size
        blocks = [_spd(block_size, key=jr.key(i)) for i in range(n_blocks)]
        self.op = BlockDiag([lx.MatrixLinearOperator(b) for b in blocks])
        self.v = jr.normal(jr.key(99), (n_blocks * block_size,), dtype=jnp.float64)
        realise(self.op.mv(self.v))

    def time_blockdiag_mv(self, blocks_x_size):
        realise(self.op.mv(self.v))


class KroneckerMvSuite:
    """Same #634 motivation: A kron B applied to a vector."""

    params = ([(16, 16), (32, 32), (64, 64)],)
    param_names = ("a_x_b_size",)

    def setup(self, a_x_b_size):
        a, b = a_x_b_size
        A = lx.MatrixLinearOperator(_spd(a, key=jr.key(7)))
        B = lx.MatrixLinearOperator(_spd(b, key=jr.key(8)))
        self.op = Kronecker(A, B)
        self.v = jr.normal(jr.key(9), (a * b,), dtype=jnp.float64)
        realise(self.op.mv(self.v))

    def time_kronecker_mv(self, a_x_b_size):
        realise(self.op.mv(self.v))
