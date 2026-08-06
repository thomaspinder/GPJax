---
status: accepted
---

# NUTS instead of the paper's bespoke Gibbs sampler

The paper estimates the model with a hand-built Gibbs sampler: Kozumi–Kobayashi
exponential-mixture augmentation of the asymmetric-Laplace likelihood, Botev
(2017) minimax-tilting truncated-MVN draws for the monotone Bernstein
coefficients, and a precision sampler for the common time effect. None of
these steps exist in NumPyro, and porting them would dominate a didactic
notebook. We run NUTS on the full joint posterior instead: the ALD density is
used directly (no augmentation needed outside Gibbs), and the monotone cone is
reparametrized away with an ordered transform.

## Consequences

- Identical model support; different sampler. Algorithm-level fidelity to the
  paper is a non-goal.
- The ordered-transform route implies a different prior *measure* on the
  monotone cone than the paper's truncated MVN (same support). Recorded
  separately once the prior block is finalized.
