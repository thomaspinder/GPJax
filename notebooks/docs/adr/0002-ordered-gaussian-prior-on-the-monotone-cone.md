---
status: accepted
---

# Ordered-Gaussian prior on the monotone cone, not a truncated MVN

The paper places a truncated Gaussian N_C(m_Γ, V_Γ) on the Bernstein
coefficients, restricted to the monotone cone C = {γ : (I_p ⊗ D)γ ⪰ 0}, and
samples it with Botev's minimax-tilting algorithm inside Gibbs. Under NUTS we
instead sample an ordered-transformed Gaussian (base N(0, 2), NumPyro
`OrderedTransform`): identical support, different measure on the cone.

## Consequences

- Near-flat coefficient paths (rows sitting close to the cone boundary) carry
  less prior mass under the exp-increment ordered transform than under a
  boundary-hugging truncated Gaussian. A 2×2 seam experiment (ordered vs
  half-normal-increment parametrisations × per-node vs common ALD scales)
  showed **no measurable difference** in surface recovery, PIT calibration,
  divergences, or population-path bias on the validation panel, so the
  spec-approved ordered parametrisation was kept.
- Numeric prior hyperparameters are our choice throughout — the paper states
  none (no m_Γ, V_Γ, Σ family, lengthscale treatment, ALD-scale prior, or
  MCMC configuration anywhere, including appendices).
