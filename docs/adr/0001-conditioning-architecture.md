# ADR-0001: The v1.0 conditioning architecture

- **Status:** accepted
- **Date:** 2026-08-06
- **Deciders:** Thomas Pinder (design settled in an architecture-review +
  grilling session; full decision tree recorded there)

## Context

An architecture review (2026-08-06) found that no module owned "condition a
GP on data": the derivation *stabilise → factor → solve → predictive
moments* was written out at eleven call sites across `gps.py`,
`objectives.py`, `variational_families.py`, and `models/oilmm.py`. Fixes
reached some copies and missed others (the diagonal-predict fix landed in two
of eleven), jitter had two owners (`Prior.jitter` vs `Posterior.jitter`), so
`predict` and `conjugate_mll` could factorise *different matrices* for the
same model, and `conjugate_mll` — the oracle for the Kalman MLL and
`collapsed_elbo` — had no value-level test of its own.

## Decision

One deep conditioning module, with the public API as its veneer, rolled out
universally at v1.0:

- `prior * likelihood` returns a **`JointModel`** — the joint p(f, y), the
  trainable object (`ConjugateModel`, `NonConjugateModel` with a lazily-sized
  latent, `HeteroscedasticModel` owning the noise-process prior).
- `model.condition(D)` (operator sugar `model | D`) returns a **`Posterior`**:
  an immutable pytree caching the factorisation, with the predictive,
  `log_marginal_likelihood`, `loo`, and pathwise `sample_approx` as views of
  it. One abstract interface; per-mode internal implementations.
- `prior.jitter` is the single stabilisation knob, applied exactly once,
  inside conditioning, through `linalg.stabilised_cholesky` (the seed of the
  linalg deepening).
- `predict(x, D)` survives as documented one-line sugar. Objectives become
  one-line views. `return_covariance_type` is renamed `covariance`.
- Likelihoods are pure conditionals: `num_datapoints` deleted (`Dataset`
  carries a static `n_total`, stamped by `get_batch`, from which the
  minibatch ELBO scale is derived) and `noise_prior` moved to
  `HeteroscedasticModel` (removing the `likelihoods -> gps` circular import).
- Deleted as failed deletion-tests: `AbstractPrior` (one-adapter seam),
  `AbstractPosterior` (splits into JointModel/Posterior), the
  `LatentPosterior` and `ChainedPosterior` markers.
- Universality is a property of the **v1.0 release**, assembled from stacked
  PRs: safety net → core conditioning → variational universalisation. The
  variational PR is sequenced **after** the natural-gradients stack
  (#714–#730) merges, since that stack rewrites `variational_families.py`;
  until then the families keep their `posterior` field name and their own
  predict derivations (renamed mechanically only).

The safety net landed first, in its own PR: closed-form oracles for the MLL,
predict, and LOOCV; cross-derivation equivalence pins; and an integration
harness whose failures actually raise.

## Rejected alternatives

- **Dropping the JointModel object** ("just `prior.condition(data,
  likelihood)`"): the trainable/derived split is load-bearing. `fit` needs a
  named pytree of everything trainable; the Posterior carries derived cache
  that a gradient step must never touch. Any container invented to make
  `fit` ergonomic *is* the JointModel under another name.
- **Staged or partial public rollout**: a patchy API (some objects
  conditioning, others not) was ruled out; stacking PRs into one v1.0 release
  achieves universality without a big-bang branch.
- **Prior inside the likelihood**: the codebase already ran this experiment —
  `HeteroscedasticGaussian.noise_prior` produced circular ownership across
  three modules. Priors live in models; likelihoods stay conditional
  families.
- **Keeping `num_datapoints`**: only four real reads existed; nothing
  validated the value, so a wrong one silently mis-scaled the ELBO; no major
  GP library puts dataset size on likelihoods (research:
  `plans/2026-08-06-drop-num-datapoints-research.md`).
- **Naming the joint `Model`** (rejected in favour of `JointModel`): the
  prefix self-documents the maths and answers "why can't I predict with this
  yet?"; "model" remains in the name for practitioners.

## Consequences

- Locality: one home for the algebra; the two-owner jitter bug is
  structurally impossible; `sample_approx` reuses the same factor as
  `predict` and refuses multi-output loudly instead of broadcasting wrongly.
- The evidence is cached with the factorisation, so repeated
  predict-then-score workflows stop re-factorising.
- Breaking changes at v1.0 are recorded in `docs/migration.md`; the
  vocabulary lives in `CONTEXT.md`.
- Variational universalisation (landed as the stack's final PR): every
  Gaussian-output family conditions through the module's sparse/collapsed
  modes, deleting the five copied predictive derivations and the duplicated
  Titsias bound. The family-side `jitter` knob is gone — `Prior.jitter` is
  applied inside conditioning for families exactly as for joint models — so
  the collapsed-ELBO/MLL equivalence test that was strictly xfailed at
  non-default jitter now passes, and the families' `posterior` field is
  renamed `model` (it holds the joint, not a posterior).
- Follow-ups tracked for the stack: the linalg structure-preserving
  deepening, one training loop with stepper adapters, the compute-engine
  seam, sharing the `K_zz` factor between `prior_kl` and `condition()`
  within a single ELBO step (today each factorises its own; XLA CSE merges
  them under `jit`), and the heteroscedastic family's NamedTuple predict
  (candidate-3 debt — it still derives its predictive through its two
  sub-families rather than a conditioning mode of its own).
