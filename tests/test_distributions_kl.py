"""Tests for `gpjax.distributions._kl_divergence`.

Regression cover for issue #664: the KL divergence performed four O(N³)
Cholesky factorisations where two suffice, because `logdet` re-factorised
covariances whose factors were already in hand.
"""

import gpjax.distributions as distributions_module
from gpjax.distributions import (
    GaussianDistribution,
    _kl_divergence,
)
from gpjax.linalg.custom_operators import BlockDiag, Kronecker
import gpjax.linalg.utils as linalg_utils
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, Float
import lineax as lx
import pytest

# ---------------------------------------------------------------------------
# Fixed problem data. The covariances are hard-coded (rather than regenerated
# from a PRNG key) so the reference KL values below stay valid regardless of
# any change to JAX's random number generation.
# ---------------------------------------------------------------------------

MU_Q = jnp.array([0.5, -1.0, 2.0, 0.25])
MU_P = jnp.array([-0.5, 0.75, 1.0, -2.0])

SIGMA_Q = jnp.array(
    [
        [
            6.423345932061903,
            -1.5657236260856189,
            1.7868650232265757,
            2.5178265294419315,
        ],
        [
            -1.5657236260856189,
            5.533719240771048,
            -0.772030041877254,
            -1.6540599776143663,
        ],
        [1.7868650232265757, -0.772030041877254, 7.552230276375627, 1.0754558214469359],
        [
            2.5178265294419315,
            -1.6540599776143663,
            1.0754558214469359,
            6.930344303910331,
        ],
    ]
)

SIGMA_P = jnp.array(
    [
        [
            9.320493791092119,
            -0.68510143454854,
            -1.0611906030751206,
            2.7369839112022785,
        ],
        [
            -0.68510143454854,
            7.234547691937459,
            -1.0370790565138517,
            -0.8781737432082408,
        ],
        [
            -1.0611906030751206,
            -1.0370790565138517,
            5.104728457374599,
            0.21682659649216027,
        ],
        [
            2.7369839112022785,
            -0.8781737432082408,
            0.21682659649216027,
            6.312547386760501,
        ],
    ]
)

DIAG_Q = jnp.array([1.0, 2.5, 0.3, 4.0])
DIAG_P = jnp.array([0.7, 1.2, 2.0, 0.9])

_STRUCT = jax.ShapeDtypeStruct((4,), jnp.float64)


def _scale(kind: str, which: str) -> lx.AbstractLinearOperator:
    matrix = SIGMA_Q if which == "q" else SIGMA_P
    diagonal = DIAG_Q if which == "q" else DIAG_P
    if kind == "dense":
        return lx.MatrixLinearOperator(matrix)
    if kind == "diagonal":
        return lx.DiagonalLinearOperator(diagonal)
    if kind == "identity":
        return lx.IdentityLinearOperator(_STRUCT)
    if kind == "block_diag":
        # Two disjoint principal submatrices of a PD matrix are themselves
        # PD, so this is a genuine (non-dense-equivalent) covariance operator.
        return BlockDiag(
            [
                lx.MatrixLinearOperator(matrix[:2, :2]),
                lx.MatrixLinearOperator(matrix[2:, 2:]),
            ]
        )
    if kind == "kronecker":
        return Kronecker(
            lx.MatrixLinearOperator(matrix[:2, :2]),
            lx.MatrixLinearOperator(matrix[2:, 2:]),
        )
    raise ValueError(kind)


def _dense_equivalent(kind: str, which: str) -> Float[Array, "4 4"]:
    """Densifies a structured `_scale` operator for independent cross-checks."""
    matrix = SIGMA_Q if which == "q" else SIGMA_P
    if kind == "block_diag":
        return jsp.linalg.block_diag(matrix[:2, :2], matrix[2:, 2:])
    if kind == "kronecker":
        return jnp.kron(matrix[:2, :2], matrix[2:, 2:])
    raise ValueError(kind)


def _distributions(kind_q: str, kind_p: str):
    q = GaussianDistribution(MU_Q, _scale(kind_q, "q"))
    p = GaussianDistribution(MU_P, _scale(kind_p, "p"))
    return q, p


# Values captured from the implementation *before* the issue #664 fix, in
# float64. They pin the numerics: the fix must only remove redundant work.
REFERENCE_KL = {
    ("dense", "dense"): 0.8126422619336267,
    ("diagonal", "diagonal"): 6.763412478671601,
    ("identity", "identity"): 5.0625,
    ("dense", "diagonal"): 12.294647261400954,
    ("diagonal", "dense"): 2.428489493791848,
    # Independently computed (see `_dense_equivalent`) via the textbook
    # dense formula, not via `_kl_divergence` itself -- issue #709.
    ("block_diag", "block_diag"): 0.8534692623379325,
    ("kronecker", "kronecker"): 0.179706013056113,
}


# ---------------------------------------------------------------------------
# Factorisation counting
# ---------------------------------------------------------------------------


@pytest.fixture
def cholesky_calls(monkeypatch):
    """Count every `cholesky_factor` invocation reachable from `_kl_divergence`.

    `cholesky_factor` is looked up in two places on this code path: as a module
    global of `gpjax.distributions`, and as a module global of
    `gpjax.linalg.utils` (which is where `logdet`'s generic fallback finds it).
    Both bindings are patched so the count covers the whole call tree.
    """
    calls: list[str] = []
    real_cholesky_factor = linalg_utils.cholesky_factor

    def counting_cholesky_factor(op):
        calls.append(type(op).__name__)
        return real_cholesky_factor(op)

    monkeypatch.setattr(linalg_utils, "cholesky_factor", counting_cholesky_factor)
    monkeypatch.setattr(
        distributions_module, "cholesky_factor", counting_cholesky_factor
    )
    return calls


def test_kl_dense_performs_exactly_two_factorisations(cholesky_calls):
    """Dense KL needs one Cholesky per covariance, not two (issue #664)."""
    q, p = _distributions("dense", "dense")
    _kl_divergence(q, p)
    assert len(cholesky_calls) == 2, (
        f"expected 2 Cholesky factorisations, got {len(cholesky_calls)}: "
        f"{cholesky_calls}"
    )


def test_kl_mixed_dense_diagonal_performs_exactly_two_factorisations(cholesky_calls):
    q, p = _distributions("dense", "diagonal")
    _kl_divergence(q, p)
    assert len(cholesky_calls) == 2, cholesky_calls


def test_kl_diagonal_performs_exactly_two_factorisations(cholesky_calls):
    q, p = _distributions("diagonal", "diagonal")
    _kl_divergence(q, p)
    assert len(cholesky_calls) == 2, cholesky_calls


# ---------------------------------------------------------------------------
# Numerical equivalence with the pre-fix implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind_q", "kind_p"), list(REFERENCE_KL))
def test_kl_matches_prefix_reference_value(kind_q, kind_p):
    q, p = _distributions(kind_q, kind_p)
    result = _kl_divergence(q, p)
    expected = REFERENCE_KL[(kind_q, kind_p)]
    assert result.dtype == jnp.float64
    assert jnp.abs(result - expected) < 1e-10


def test_kl_matches_closed_form_formula():
    """Independent check against the textbook formula built from dense algebra."""
    q, p = _distributions("dense", "dense")
    diff = MU_P - MU_Q
    sigma_p_inv = jnp.linalg.inv(SIGMA_P)
    expected = 0.5 * (
        diff @ sigma_p_inv @ diff
        - 4
        - jnp.linalg.slogdet(SIGMA_Q)[1]
        + jnp.linalg.slogdet(SIGMA_P)[1]
        + jnp.trace(sigma_p_inv @ SIGMA_Q)
    )
    assert jnp.abs(_kl_divergence(q, p) - expected) < 1e-10


# ---------------------------------------------------------------------------
# BlockDiag / Kronecker support (issue #709)
#
# `kl_divergence` calls `lx.linear_solve(..., solver=lx.Triangular())`, which
# queries `lx.has_unit_diagonal` on the Cholesky factor. `BlockDiag` and
# `Kronecker` are registered for the other Lineax operator predicates but were
# missing this one, so the solve raised `NotImplementedError` for any
# covariance backed by either operator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["block_diag", "kronecker"])
def test_kl_matches_dense_equivalent(kind):
    """Cross-check against a manually densified covariance, independent of
    the hardcoded `REFERENCE_KL` constants."""
    q, p = _distributions(kind, kind)
    dense_q = GaussianDistribution(
        MU_Q, lx.MatrixLinearOperator(_dense_equivalent(kind, "q"))
    )
    dense_p = GaussianDistribution(
        MU_P, lx.MatrixLinearOperator(_dense_equivalent(kind, "p"))
    )
    expected = _kl_divergence(dense_q, dense_p)
    assert jnp.abs(_kl_divergence(q, p) - expected) < 1e-10


@pytest.mark.parametrize("kind", ["block_diag", "kronecker"])
def test_kl_via_public_method_works_for_structured_operators(kind):
    """Acceptance check for issue #709: the public API must not raise."""
    q, p = _distributions(kind, kind)
    result = q.kl_divergence(p)
    assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["dense", "diagonal", "identity", "block_diag", "kronecker"]
)
def test_kl_of_distribution_with_itself_is_zero(kind):
    q, _ = _distributions(kind, kind)
    assert jnp.abs(_kl_divergence(q, q)) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_kl_is_non_negative(seed):
    key_q, key_p, key_mu_q, key_mu_p = jr.split(jr.key(seed), 4)
    n = 5

    def random_psd(key):
        a = jr.normal(key, (n, n))
        return a @ a.T + n * jnp.eye(n)

    q = GaussianDistribution(
        jr.normal(key_mu_q, (n,)), lx.MatrixLinearOperator(random_psd(key_q))
    )
    p = GaussianDistribution(
        jr.normal(key_mu_p, (n,)), lx.MatrixLinearOperator(random_psd(key_p))
    )
    assert _kl_divergence(q, p) >= -1e-10


def test_kl_via_public_method_matches_private_function():
    q, p = _distributions("dense", "dense")
    assert jnp.allclose(q.kl_divergence(p), _kl_divergence(q, p))


# ---------------------------------------------------------------------------
# JAX transformations
# ---------------------------------------------------------------------------


def _kl_from_arrays(mu_q, sigma_q, mu_p, sigma_p):
    q = GaussianDistribution(mu_q, lx.MatrixLinearOperator(sigma_q))
    p = GaussianDistribution(mu_p, lx.MatrixLinearOperator(sigma_p))
    return _kl_divergence(q, p)


def test_kl_is_jittable():
    jitted = jax.jit(_kl_from_arrays)
    result = jitted(MU_Q, SIGMA_Q, MU_P, SIGMA_P)
    assert jnp.abs(result - REFERENCE_KL[("dense", "dense")]) < 1e-10


def test_kl_is_differentiable():
    grad_fn = jax.grad(_kl_from_arrays, argnums=(0, 1))
    grad_mu, grad_sigma = grad_fn(MU_Q, SIGMA_Q, MU_P, SIGMA_P)
    assert grad_mu.shape == MU_Q.shape
    assert grad_sigma.shape == SIGMA_Q.shape
    assert jnp.all(jnp.isfinite(grad_mu))
    assert jnp.all(jnp.isfinite(grad_sigma))
    # KL is minimised at q == p, so the gradient there vanishes.
    zero_grad_mu, zero_grad_sigma = grad_fn(MU_Q, SIGMA_Q, MU_Q, SIGMA_Q)
    assert jnp.allclose(zero_grad_mu, 0.0, atol=1e-10)
    assert jnp.allclose(zero_grad_sigma, 0.0, atol=1e-10)


def test_kl_is_vmappable():
    means = jnp.stack([MU_Q, MU_P, jnp.zeros(4)])
    covs = jnp.stack([SIGMA_Q, SIGMA_P, jnp.eye(4) * 2.0])
    batched = jax.vmap(_kl_from_arrays, in_axes=(0, 0, None, None))(
        means, covs, MU_P, SIGMA_P
    )
    assert batched.shape == (3,)
    assert jnp.abs(batched[0] - REFERENCE_KL[("dense", "dense")]) < 1e-10
    # Second element is KL[p || p] == 0.
    assert jnp.abs(batched[1]) < 1e-10


def test_kl_is_jittable_for_diagonal_scale():
    def kl_diag(diag_q, diag_p):
        q = GaussianDistribution(MU_Q, lx.DiagonalLinearOperator(diag_q))
        p = GaussianDistribution(MU_P, lx.DiagonalLinearOperator(diag_p))
        return _kl_divergence(q, p)

    result = jax.jit(kl_diag)(DIAG_Q, DIAG_P)
    assert jnp.abs(result - REFERENCE_KL[("diagonal", "diagonal")]) < 1e-10


def test_kl_is_jittable_for_block_diag_scale():
    def kl_block_diag(block_1_q, block_2_q, block_1_p, block_2_p):
        q = GaussianDistribution(
            MU_Q,
            BlockDiag(
                [lx.MatrixLinearOperator(block_1_q), lx.MatrixLinearOperator(block_2_q)]
            ),
        )
        p = GaussianDistribution(
            MU_P,
            BlockDiag(
                [lx.MatrixLinearOperator(block_1_p), lx.MatrixLinearOperator(block_2_p)]
            ),
        )
        return _kl_divergence(q, p)

    result = jax.jit(kl_block_diag)(
        SIGMA_Q[:2, :2], SIGMA_Q[2:, 2:], SIGMA_P[:2, :2], SIGMA_P[2:, 2:]
    )
    assert jnp.abs(result - REFERENCE_KL[("block_diag", "block_diag")]) < 1e-10


def test_kl_is_jittable_for_kronecker_scale():
    def kl_kronecker(factor_1_q, factor_2_q, factor_1_p, factor_2_p):
        q = GaussianDistribution(
            MU_Q,
            Kronecker(
                lx.MatrixLinearOperator(factor_1_q), lx.MatrixLinearOperator(factor_2_q)
            ),
        )
        p = GaussianDistribution(
            MU_P,
            Kronecker(
                lx.MatrixLinearOperator(factor_1_p), lx.MatrixLinearOperator(factor_2_p)
            ),
        )
        return _kl_divergence(q, p)

    result = jax.jit(kl_kronecker)(
        SIGMA_Q[:2, :2], SIGMA_Q[2:, 2:], SIGMA_P[:2, :2], SIGMA_P[2:, 2:]
    )
    assert jnp.abs(result - REFERENCE_KL[("kronecker", "kronecker")]) < 1e-10
