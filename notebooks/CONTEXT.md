# Soft-Noncrossing Panel Quantile Regression (notebook)

The bounded context of the marimo notebook recreating Huber, Poon & Zhu
(arXiv:2608.04664) with GPJax + NumPyro. Deliberately separate from the root
`CONTEXT.md` (the GPJax library language): this directory migrates out of the
repo once GPJax v1.0 merges to `main`.

## Language

### Model

**Quantile grid**:
The finite set of quantile levels 𝒯 on which coefficient paths are estimated.
_Avoid_: quantile levels, tau grid

**Coefficient path**:
A function τ ↦ coefficient (intercept, macro slope, or climate response) for
one country, treated as a stochastic process in τ.
_Avoid_: coefficient curve, quantile process

**Population path**:
The cross-country mean coefficient path, represented in a Bernstein basis and
componentwise nondecreasing in τ by construction.
_Avoid_: mean path, central path, grand mean

**Unit deviation**:
A country's smooth Gaussian-process departure from the population path; its
scale governs the probability of quantile crossing.
_Avoid_: random effect, country effect

**Soft noncrossing**:
Noncrossing delivered probabilistically — a monotone population path plus
small, smooth unit deviations makes crossings unlikely (Theorem 1), never
impossible.
_Avoid_: monotonicity constraint, noncrossing constraint

**Common time effect**:
The quantile-invariant latent AR(1) factor f_t that absorbs aggregate shocks,
driven by observed global predictors and normalised to sum to zero.
_Avoid_: time fixed effect, global factor, time dummy

**Working likelihood**:
The product of asymmetric-Laplace terms over the quantile grid — a composite
likelihood in which each observation is counted once per grid node, so its
total weight scales with grid size.
_Avoid_: likelihood (unqualified), pseudo-likelihood

### Climate application

**Climate shock**:
One of four standardised disturbances entering eq. (9) as separate regressors:
local/global × level/volatility temperature shocks.
_Avoid_: climate variable, weather shock

**Global component**:
The PPP-GDP-weighted cross-country average of a country-level shock series;
the **local component** is a country's residual from it.
_Avoid_: world shock, aggregate shock

**DIRF**:
Distributional impulse response function — the climate coefficient path
itself, θ^s_{i,h}(τ), read across the quantile grid.
_Avoid_: IRF (unqualified), quantile response

**Baseline scenario**:
The counterfactual with all four climate shocks set to zero; a shock scenario
raises exactly one shock by one standard deviation.
_Avoid_: no-shock world, control

**Growth-at-risk (GaR)**:
A lower-tail conditional quantile of cumulative future output growth under a
scenario; ΔGaR is the scenario-minus-baseline shift.
_Avoid_: VaR, downside quantile

**Expected shortfall (ES)**:
The average of conditional quantiles at or below the tail level α (0.05),
approximated by the grid nodes in that tail.
_Avoid_: CVaR, tail mean
