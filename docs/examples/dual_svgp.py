# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dual Parameterisation of Sparse GPs (t-SVGP)
#
# Download this notebook: {nb-download}`dual_svgp.ipynb`
#
# This is the applied companion to the
# [natural gradients notebook](natural_gradients.py). That notebook derives
# the whole geometry this one puts to work: the exponential-family view of
# $q(\mathbf{u})$, the Fisher identity that makes the natural gradient free
# to compute, the site — or *dual* — reparameterisation of
# {cite:t}`adam2021dual`, its EP heritage, the two silent convention traps in
# the source material, the tied update and why it never inverts anything,
# the positive-semidefinite cone-safety argument for the site branch, and
# the claims table for `dual_elbo` versus `elbo` as an M-step objective.
# **Read that one first.** Nothing below re-derives any of it; this notebook
# assumes it and asks instead: does GPJax's implementation actually deliver
# what the theory promises, on real models, with real numbers? A sibling
# notebook, [natgrads.py](natgrads.py), runs the same kind of check for the
# moment-storage branch, `VariationalGaussian`.
#
# Four checks, in order. A conjugate regression where `DualVariationalGaussian`
# plus one `natural_gradient_step` at $\rho=1$ reproduces the
# {cite:t}`titsias2009` optimum exactly. A non-conjugate classification
# problem where the site branch and the moment branch are driven through
# matched steps and compared directly, which locates precisely where the
# $\rho=\gamma$ identity's log-concavity condition fails and `beta_floor`
# starts to matter. The banana benchmark from the natural-gradients notebook,
# run again with all three optimisers — Adam alone, natural gradients on
# $(\mathbf{m},\mathbf{L})$, and t-SVGP's dual branch — with timings. And,
# last, the M-step claims table exercised in practice: a bound slice showing
# where `dual_elbo` dominates `elbo` and where it does not, and a full
# variational-EM training loop comparing the two as M-step objectives.
#
# **One notational reminder, carried over from the natural-gradients
# notebook.** $\boldsymbol{\theta}$ is the kernel hyperparameters, not a
# natural parameter; the natural parameter of $q(\mathbf{u})$ is
# $\boldsymbol{\eta}$, its expectation parameter is $\boldsymbol{\mu}$, and
# $\boldsymbol{\lambda} = (\boldsymbol{\lambda}_1, \boldsymbol{\Lambda}_2)$
# is the site — the pair `DualVariationalGaussian` stores as `dual_vector`
# and `dual_matrix`. $\mathbf{a}_i = \mathbf{K}_{zz}^{-1}\mathbf{k}_z(x_i)$
# throughout.

# %%
# Enable Float64 for more stable matrix inversions.
import time

import equinox as eqx
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import paramax
from utils import clean_legend, use_mpl_style

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    import gpjax.kernels as jk
    from gpjax.natural_gradients import (
        natural_gradient_step,
        partition_variational,
    )
    from gpjax.objectives import dual_elbo, elbo
    from gpjax.parameters import Real
    from gpjax.variational_families import (
        DualVariationalGaussian,
        VariationalGaussian,
    )

key = jr.key(123)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


def negative_elbo(model, data):
    """The loss for a family that stores moments; GPJax optimisers descend."""
    return -elbo(model, data)


def negative_dual_elbo(model, data):
    """The loss for a family that stores sites."""
    return -dual_elbo(model, data)


# %% [markdown]
# ## What `DualVariationalGaussian` stores
#
# In brief, because the natural-gradients notebook has the derivation: $q$
# is the prior reweighted by an unnormalised Gaussian site,
# $q(\mathbf{u}) \propto p_{\boldsymbol{\theta}}(\mathbf{u})\,
# \exp(\boldsymbol{\lambda}_1^\top\tilde{\mathbf{u}} -
# \tfrac12\tilde{\mathbf{u}}^\top\boldsymbol{\Lambda}_2\tilde{\mathbf{u}})$
# with $\tilde{\mathbf{u}} = \mathbf{u} - \boldsymbol{\mu}_z$, giving moments
# $\mathbf{S} = (\mathbf{K}_{zz}^{-1} + \boldsymbol{\Lambda}_2)^{-1}$,
# $\mathbf{m} = \boldsymbol{\mu}_z + \mathbf{S}\boldsymbol{\lambda}_1$.
# `DualVariationalGaussian` stores $\boldsymbol{\lambda}_1$ as `dual_vector`
# and $\boldsymbol{\Lambda}_2$ as `dual_matrix`, both zero by default — so
# $q = p$ at initialisation — and exposes the pair above via `.moments()`.
# GPJax stores the **flanked, precision** convention (never the un-flanked
# sums, never $-\tfrac12\boldsymbol{\Lambda}_2$); that is the rule the
# natural-gradients notebook derives and states, and every demo below
# depends on it silently, the way any user of the API does.
#
# Where do $\boldsymbol{\lambda}_1$ and $\boldsymbol{\Lambda}_2$ come from at
# the optimum? One local likelihood approximation per data point, EP-style,
# with the site values read off Bonnet's and Price's theorems rather than
# fitted by moment matching:
#
# $$\alpha_i = \frac{\partial}{\partial m_i}\,\mathbb{E}_{q(f_i)}\!\left[\log p(y_i\mid f_i)\right], \qquad \beta_i = -2\,\frac{\partial}{\partial v_i}\,\mathbb{E}_{q(f_i)}\!\left[\log p(y_i\mid f_i)\right],$$
#
# so a single `jax.grad` of the likelihood's existing
# `expected_log_likelihood` gives both, for closed-form and quadrature
# likelihoods alike. Here is that call pattern, checked against the
# Gaussian closed form $\alpha_i = (y_i-m_i)/\sigma^2$,
# $\beta_i = 1/\sigma^2$.

# %%
key, alpha_beta_key = jr.split(key)
check_response = jr.normal(alpha_beta_key, (5, 1))
check_mean = jnp.linspace(-1.0, 1.0, 5)
check_variance = jnp.linspace(0.2, 0.9, 5)
check_stddev = 0.37
check_likelihood = gpx.likelihoods.Gaussian(obs_stddev=check_stddev)


def total_expected_log_likelihood(mean, variance):
    """Summed variational expectation, as a function of the marginal moments."""
    return jnp.sum(
        check_likelihood.expected_log_likelihood(
            check_response, mean[:, None], variance[:, None]
        )
    )


bonnet_alpha, price_derivative = jax.grad(
    total_expected_log_likelihood, argnums=(0, 1)
)(check_mean, check_variance)
price_beta = -2.0 * price_derivative

closed_form_alpha = (check_response.squeeze(-1) - check_mean) / check_stddev**2
closed_form_beta = jnp.full_like(check_mean, 1.0 / check_stddev**2)

print(
    "max |alpha - (y - m) / sigma^2| : "
    f"{jnp.max(jnp.abs(bonnet_alpha - closed_form_alpha)):.3e}"
)
print(
    "max |beta - 1 / sigma^2|        : "
    f"{jnp.max(jnp.abs(price_beta - closed_form_beta)):.3e}"
)
print(f"beta                            : {price_beta[0]:.6f} (= 1 / {check_stddev}^2)")

# %% [markdown]
# Both match the closed form to machine precision. This is the ingredient
# every demo below is built from: `expected_log_likelihood` plus one
# `jax.grad` call, tied across data points into the $M$- and
# $M\times M$-sized `dual_vector`/`dual_matrix` update the natural-gradients
# notebook derives in full.

# %% [markdown]
# ## Conjugate models: one step is enough
#
# With a Gaussian likelihood the site targets do not depend on $q$ at all:
#
# $$\alpha_i = \frac{y_i - m_i}{\sigma^2}, \quad \beta_i = \frac{1}{\sigma^2} \qquad\Longrightarrow\qquad g_{1,i} = \alpha_i + \beta_i\left(m_i - \mu(x_i)\right) = \frac{y_i - \mu(x_i)}{\sigma^2}, \quad g_{2,i} = \frac{1}{\sigma^2} .$$
#
# The $m_i$ cancels, so the tied update's target does not move either, and
# $\rho=1$ lands on the fixed point from anywhere in one step:
#
# $$\boldsymbol{\lambda}_1^\star = \frac{1}{\sigma^2}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zx}(\mathbf{y}-\boldsymbol{\mu}_x), \qquad \boldsymbol{\Lambda}_2^\star = \frac{1}{\sigma^2}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zx}\mathbf{K}_{xz}\mathbf{K}_{zz}^{-1},$$
#
# whereupon $\mathbf{R}^\star = \mathbf{K}_{zz} + \sigma^{-2}\mathbf{K}_{zx}\mathbf{K}_{xz}$
# is exactly the inverse of Titsias' $\boldsymbol{\Sigma}$ and $(\mathbf{m}^\star,\mathbf{S}^\star)$
# is the {cite:t}`titsias2009` optimal $q(\mathbf{u})$ verbatim — the site
# instantiation of the general "$\gamma=1$ is exact" argument the
# natural-gradients notebook proves for any storage convention. The mean
# function below is deliberately non-zero: the sites act on the *centred*
# process, so $\mathbf{y}-\boldsymbol{\mu}_x$ is where that matters.

# %%
num_data = 200
noise_stddev = 0.3
observation_variance = noise_stddev**2
prior_constant = 0.4
regression_lengthscale = 0.5
regression_jitter = 1e-8

key, input_key, noise_key = jr.split(key, 3)
regression_inputs = jr.uniform(input_key, (num_data, 1), minval=-3.0, maxval=3.0)
regression_outputs = jnp.sin(2.0 * regression_inputs) + noise_stddev * jr.normal(
    noise_key, (num_data, 1)
)
regression_data = gpx.Dataset(X=regression_inputs, y=regression_outputs)

num_inducing = 20
regression_inducing = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)


def conjugate_model(lengthscale):
    """The conjugate joint model (prior * likelihood) at a given RBF lengthscale."""
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(prior_constant)),
        kernel=jk.RBF(lengthscale=lengthscale),
        jitter=regression_jitter,
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=noise_stddev)


def site_family(lengthscale, inducing_inputs, sites=None):
    """A dual family, optionally carrying a frozen pair of sites."""
    family = DualVariationalGaussian(
        model=conjugate_model(lengthscale),
        inducing_inputs=inducing_inputs,
    )
    if sites is None:
        return family
    return eqx.tree_at(
        lambda tree: (tree.dual_vector, tree.dual_matrix),
        family,
        (Real(sites[0]), Real(sites[1])),
    )


def moment_family(lengthscale, inducing_inputs, moments):
    """A moment family carrying a frozen $(m, S)$."""
    mean, covariance = moments
    return VariationalGaussian(
        model=conjugate_model(lengthscale),
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=jnp.linalg.cholesky(covariance),
    )


def exact_sites(lengthscale, inducing_inputs, dataset):
    """One rho = 1 conjugate step from lambda = 0: the exactly optimal sites."""
    variational, hyper = partition_variational(
        site_family(lengthscale, inducing_inputs)
    )
    variational, _ = natural_gradient_step(
        variational, hyper, dataset, negative_dual_elbo, 1.0
    )
    fitted = paramax.unwrap(eqx.combine(variational, hyper))
    return (fitted.dual_vector, fitted.dual_matrix), fitted.moments()


# %%
# The Titsias optimum in closed form, against the same jittered K_zz the family uses.
initial_dual = site_family(regression_lengthscale, regression_inducing)
regression_prior = paramax.unwrap(initial_dual).model.prior
regression_kernel = regression_prior.kernel
regression_mean_function = regression_prior.mean_function

Kzz = regression_kernel.gram(regression_inducing).as_matrix()
Kzz = Kzz + regression_jitter * jnp.eye(num_inducing)
Kzx = regression_kernel.cross_covariance(regression_inducing, regression_inputs)
centred_outputs = regression_outputs - regression_mean_function(regression_inputs)

titsias_precision = Kzz + Kzx @ Kzx.T / observation_variance
optimal_mean = (
    regression_mean_function(regression_inducing)
    + Kzz
    @ jnp.linalg.solve(titsias_precision, Kzx @ centred_outputs)
    / observation_variance
)
optimal_covariance = Kzz @ jnp.linalg.solve(titsias_precision, Kzz)

# The collapsed (Titsias) bound, which the dual ELBO must reproduce at that optimum.
nystrom = Kzx.T @ jnp.linalg.solve(Kzz, Kzx)
marginal_covariance = nystrom + observation_variance * jnp.eye(num_data)
_, marginal_logdet = jnp.linalg.slogdet(marginal_covariance)
marginal_quadratic = centred_outputs.squeeze(-1) @ jnp.linalg.solve(
    marginal_covariance, centred_outputs.squeeze(-1)
)
prior_variance_diagonal = jnp.diag(
    regression_kernel.gram(regression_inputs).as_matrix()
)
sparsity_gap = jnp.sum(prior_variance_diagonal - jnp.diag(nystrom)) / (
    2 * observation_variance
)
collapsed_bound = (
    -0.5 * (num_data * jnp.log(2 * jnp.pi) + marginal_logdet + marginal_quadratic)
    - sparsity_gap
)

# %%
# One dual natural-gradient step at rho = 1, from lambda = 0.
dual_variational, dual_hyper = partition_variational(initial_dual)
stepped_variational, loss_before = natural_gradient_step(
    dual_variational, dual_hyper, regression_data, negative_dual_elbo, 1.0
)
stepped_dual = paramax.unwrap(eqx.combine(stepped_variational, dual_hyper))
stepped_mean, stepped_covariance = stepped_dual.moments()

# A second step must be a no-op.
twice_stepped_variational, _ = natural_gradient_step(
    stepped_variational, dual_hyper, regression_data, negative_dual_elbo, 1.0
)
twice_stepped_dual = paramax.unwrap(eqx.combine(twice_stepped_variational, dual_hyper))
twice_stepped_mean, twice_stepped_covariance = twice_stepped_dual.moments()

stepped_bound = dual_elbo(stepped_dual, regression_data)

print(f"ELBO before the step        : {-loss_before:12.6f}")
print(f"dual_elbo after one step    : {float(stepped_bound):12.6f}")
print(f"Titsias collapsed bound     : {float(collapsed_bound):12.6f}")
print(
    f"max |m - m*|                : {jnp.max(jnp.abs(stepped_mean - optimal_mean)):.3e}"
)
print(
    "max |S - S*|                : "
    f"{jnp.max(jnp.abs(stepped_covariance - optimal_covariance)):.3e}"
)
print(
    "max |m_2 - m_1| (fixed pt)  : "
    f"{jnp.max(jnp.abs(twice_stepped_mean - stepped_mean)):.3e}"
)
print(
    "max |S_2 - S_1| (fixed pt)  : "
    f"{jnp.max(jnp.abs(twice_stepped_covariance - stepped_covariance)):.3e}"
)
print(f"collapsed bound - dual_elbo : {float(collapsed_bound - stepped_bound):.12e}")
print(
    "N * jitter / (2 sigma^2)    : "
    f"{num_data * regression_jitter / (2 * observation_variance):.12e}"
)

# %% [markdown]
# One step from $\boldsymbol{\lambda}=\mathbf{0}$ reproduces the Titsias
# optimum to around $10^{-12}$ in the mean and $10^{-13}$ in the covariance,
# and a second step moves nothing — the numbers above confirm both.
#
# The last two printed lines deserve a sentence, because the residual
# between `dual_elbo` and the analytic collapsed bound is not noise — it is
# $N\varepsilon/(2\sigma^2)$ to eight significant figures, where
# $\varepsilon$ is the model's `Prior.jitter`, which is why both are printed
# to twelve. The
# conditioned sparse posterior adds that jitter to every marginal variance it
# returns, so `elbo` carries the inflation too, and the dual family
# reproduces it deliberately: matching the two objectives to machine
# precision is worth more than matching either to a formula on paper. Lower
# the jitter and the gap falls proportionally.
#
# One thing this demo does *not* show, contrary to a remark in the paper
# that is easy to over-read: the constant $c(\boldsymbol{\theta})$ relating
# the dual ELBO to $\log\mathcal{Z}(\boldsymbol{\theta})$ is **not** zero
# here. Its value depends on which site convention $\mathcal{Z}$ is taken
# against, and the two have to be paired consistently. Against the
# *normalised projected* site
# $t_i(\mathbf{u}) = \mathcal{N}(y_i \mid \mathbf{a}_i^\top\mathbf{u}, \sigma^2)$,
# $c(\boldsymbol{\theta})$ is minus the Titsias trace term — the negated
# `sparsity_gap` computed above — and it vanishes only when
# $\mathbf{Z} = \mathbf{X}$, the non-sparse case the paper's remark actually
# covers. Against the unnormalised, flanked site GPJax stores, it picks up
# the site normaliser as well and is a different and much larger constant.
# Either way it is non-zero and $\boldsymbol{\theta}$-dependent, which is why
# GPJax evaluates the bound as (variational expectation $-$ KL) rather than
# as a log-partition function.

# %% [markdown]
# ## Locating where $\rho=\gamma$ stops holding
#
# The natural-gradients notebook proves the site step and the moment step
# are the *same iteration* whenever the computed $\beta_i$ stay
# non-negative, and shows this on the banana classification problem: a
# single point crosses into $\beta_i<0$ at step five of six matched
# $\rho=\gamma=0.8$ steps, the `beta_floor` clip engages, and the two
# branches part company at $\mathcal{O}(10^{-3})$ in $(\mathbf{m},\mathbf{S})$
# — while agreeing to the float64 noise floor everywhere before that. We
# reproduce that check here rather than take it on faith, because it is the
# load-bearing claim behind everything that follows: the banana benchmark
# and the VEM loop below both interleave many such steps, so knowing exactly
# when and how far the two branches can diverge tells us how much of any
# difference in their trajectories to attribute to the E-step versus the
# M-step.
#
# The next cell is the data-generating function and inducing-point layout
# from the natural-gradients notebook, reproduced character for character —
# same function body, same `jr.key(42)`, same 2000 points, same $10\times5$
# grid — so the problem here is the same problem, point for point, as the
# one there.


# %%
def make_banana(key, num_points):
    """Two-class banana problem with a curved Bayes-optimal boundary."""
    key_latent, key_label = jr.split(key)
    latent = jr.uniform(key_latent, (num_points, 2), minval=-3.0, maxval=3.0)
    decision = latent[:, 1] - (0.7 * latent[:, 0] ** 2 - 1.5)
    probability = jax.nn.sigmoid(3.0 * decision)
    labels = (jr.uniform(key_label, (num_points,)) < probability).astype(jnp.float64)
    return latent, labels[:, None]


banana_key = jr.key(42)
banana_inputs, banana_labels = make_banana(banana_key, 2000)
banana_data = gpx.Dataset(X=banana_inputs, y=banana_labels)

num_train = 1600
train_inputs, test_inputs_2d = banana_inputs[:num_train], banana_inputs[num_train:]
train_labels, test_labels = banana_labels[:num_train], banana_labels[num_train:]
banana_train = gpx.Dataset(X=train_inputs, y=train_labels)

num_banana_inducing = 50
inducing_grid = jnp.meshgrid(jnp.linspace(-2.8, 2.8, 10), jnp.linspace(-2.8, 2.8, 5))
banana_inducing = jnp.stack([axis.ravel() for axis in inducing_grid], axis=1)
banana_jitter = 1e-6

banana_model = (
    gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=jk.RBF(active_dims=[0, 1]),
    )
    * gpx.likelihoods.Bernoulli()
)

banana_gram = paramax.unwrap(banana_model).prior.kernel.gram(
    banana_inducing
).as_matrix() + banana_jitter * jnp.eye(num_banana_inducing)
banana_prior_root = jnp.linalg.cholesky(banana_gram)


def make_banana_moment_family():
    """A fresh SVGP over the banana data, at q = p."""
    return VariationalGaussian(
        model=banana_model,
        inducing_inputs=banana_inducing,
        variational_mean=jnp.zeros((num_banana_inducing, 1)),
        variational_root_covariance=banana_prior_root,
    )


print(f"train / test  : {banana_train.n} / {banana_data.n - banana_train.n}")
print(f"class balance : {float(banana_data.y.mean()):.3f}")
print(f"inducing inputs : {banana_inducing.shape}")
print(f"cond(K_zz)      : {jnp.linalg.cond(banana_gram):.3e}")


# %%
def implied_moments(family):
    """Return $(m, S)$ for either parameterisation."""
    unwrapped = paramax.unwrap(family)
    if isinstance(unwrapped, DualVariationalGaussian):
        return unwrapped.moments()
    root = unwrapped.variational_root_covariance
    return unwrapped.variational_mean, root @ root.T


def price_curvature(family, data):
    """Return the marginal means and $\\beta_i=-2\\,\\partial_{v_i}E_q[\\log p]$."""
    marginal_mean, marginal_variance = family.marginals(data.X)

    def total_expectation(variance):
        return jnp.sum(
            family.model.likelihood.expected_log_likelihood(
                data.y, marginal_mean[:, None], variance[:, None]
            )
        )

    return marginal_mean, -2.0 * jax.grad(total_expectation)(marginal_variance)


def six_matched_steps(beta_floor):
    """Six rho = 0.8 steps in both branches, from the shared q = p start."""
    site_partition, site_hyper = partition_variational(
        DualVariationalGaussian(
            model=banana_model,
            inducing_inputs=banana_inducing,
        )
    )
    moment_partition, moment_hyper = partition_variational(make_banana_moment_family())
    rows = []
    for _ in range(6):
        # Measured before the step, at the q both branches currently share.
        marginal_mean, curvature = price_curvature(
            paramax.unwrap(eqx.combine(site_partition, site_hyper)), banana_train
        )
        site_partition, _ = natural_gradient_step(
            site_partition,
            site_hyper,
            banana_train,
            negative_dual_elbo,
            0.8,
            beta_floor=beta_floor,
        )
        moment_partition, _ = natural_gradient_step(
            moment_partition, moment_hyper, banana_train, negative_elbo, 0.8
        )
        site_mean, site_covariance = implied_moments(
            eqx.combine(site_partition, site_hyper)
        )
        moment_mean, moment_covariance = implied_moments(
            eqx.combine(moment_partition, moment_hyper)
        )
        rows.append(
            (
                max(
                    float(jnp.max(jnp.abs(site_mean - moment_mean))),
                    float(jnp.max(jnp.abs(site_covariance - moment_covariance))),
                ),
                int(jnp.sum(curvature < 0)),
                float(jnp.min(curvature)),
                float(marginal_mean[jnp.argmin(curvature)]),
            )
        )
    return rows


print("step   |(m, S) gap|   beta < 0   min beta   its marginal mean")
for step, (gap, negative_count, smallest, mean_there) in enumerate(
    six_matched_steps(1e-8), start=1
):
    print(
        f"{step:4d}   {gap:12.3e}   {negative_count:4d}/{banana_train.n}"
        f"   {smallest:+8.4f}   {mean_there:+8.3f}"
    )

banana_gap = max(gap for gap, _, _, _ in six_matched_steps(1e-8))
unfloored_gap = max(gap for gap, _, _, _ in six_matched_steps(-jnp.inf))
print(f"\nworst gap, default beta_floor = 1e-8 : {banana_gap:.3e}")
print(f"worst gap, clip disabled (-inf)      : {unfloored_gap:.3e}")

# %% [markdown]
# The table matches the natural-gradients notebook's: for the first four
# steps every $\beta_i$ is positive, the clip is inert, and the branches
# agree to $10^{-13}$. At step five a single training point out of 1600
# crosses into $\beta_i<0$ — a label-$0$ point whose marginal mean of
# $+2.427$ sits almost exactly on the mirrored $+2.44$ threshold the
# cone-safety section derives for GPJax's clipped `inv_probit` —
# `beta_floor` engages, and from that step on the gap jumps
# to $\mathcal{O}(10^{-3})$ and compounds. Disabling the clip brings the same
# six steps back to the noise floor, which is the control that pins the
# cause down to the clip alone.
#
# For the two demos ahead, the practical reading is: whatever difference
# shows up between the dual and moment branches beyond the low-$10^{-3}$
# level located here is not coming from the E-step being a different
# search direction. It is either wall-clock, or the M-step objective.

# %% [markdown]
# ## The banana benchmark: three optimisers
#
# All three models below start at $q=p$, that is $\mathbf{m}=\mathbf{0}$,
# $\mathbf{S}=\mathbf{K}_{zz}$ — where a dual family with zero sites already
# is — over the same inducing grid just built.

# %%
banana_dual_family = DualVariationalGaussian(
    model=banana_model,
    inducing_inputs=banana_inducing,
)
natgrad_family = make_banana_moment_family()
adam_family = make_banana_moment_family()

# The log-linear ramp of the natural-gradients notebook: 1e-4 -> 1e-1 over K = 100.
num_iterations = 1000
batch_size = 256
natgrad_schedule = ox.exponential_decay(
    init_value=1e-4, transition_steps=100, decay_rate=1000.0, end_value=1e-1
)


def timed_fit(run):
    """Run twice: the first call pays JIT compilation, the second is steady state."""
    model, history = run()
    history.block_until_ready()
    start = time.perf_counter()
    model, history = run()
    history.block_until_ready()
    return model, history, time.perf_counter() - start


shared_settings = dict(
    train_data=banana_train,
    optim=ox.adam(1e-2),
    batch_size=batch_size,
    num_iters=num_iterations,
    key=jr.key(1),
    verbose=False,
)

# %%
dual_model, dual_history, dual_seconds = timed_fit(
    lambda: gpx.fit_natgrads(
        model=banana_dual_family,
        objective=negative_dual_elbo,
        natgrad_lr=natgrad_schedule,
        **shared_settings,
    )
)
natgrad_model, natgrad_history, natgrad_seconds = timed_fit(
    lambda: gpx.fit_natgrads(
        model=natgrad_family,
        objective=negative_elbo,
        natgrad_lr=natgrad_schedule,
        **shared_settings,
    )
)
adam_model, adam_history, adam_seconds = timed_fit(
    lambda: gpx.fit(model=adam_family, objective=negative_elbo, **shared_settings)
)

for name, seconds in [
    ("t-SVGP (dual) + Adam", dual_seconds),
    ("natural gradients + Adam", natgrad_seconds),
    ("Adam only", adam_seconds),
]:
    print(
        f"{name:26s}: {seconds:5.2f} s "
        f"({1e3 * seconds / num_iterations:.2f} ms / iteration)"
    )

# %%
smoothing_window = 25


def smooth(history):
    """Trailing mean over `smoothing_window` iterations."""
    return jnp.convolve(
        history, jnp.ones(smoothing_window) / smoothing_window, mode="valid"
    )


smoothed_iterations = jnp.arange(smoothing_window - 1, num_iterations)
curves = [
    ("t-SVGP (dual) + Adam", smooth(dual_history), dual_seconds, cols[2]),
    ("Natural gradients + Adam", smooth(natgrad_history), natgrad_seconds, cols[1]),
    ("Adam only", smooth(adam_history), adam_seconds, cols[0]),
]

elbo_floor = 0.95 * min(float(curve.min()) for _, curve, _, _ in curves)
elbo_ceiling = 1.10 * max(float(curve.max()) for _, curve, _, _ in curves)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0), sharey=True)
for name, curve, seconds, colour in curves:
    axes[0].plot(smoothed_iterations, curve, color=colour, label=name)
    axes[1].plot(
        jnp.linspace(0.0, seconds, num_iterations)[smoothing_window - 1 :],
        curve,
        color=colour,
        label=name,
    )
axes[0].set(xlabel="Iteration", yscale="log", ylim=(elbo_floor, elbo_ceiling))
axes[1].set(xlabel="Wall-clock seconds", yscale="log", ylim=(elbo_floor, elbo_ceiling))
axes[0].set_ylabel("Negative ELBO (mini-batch)")
clean_legend(axes[0])
clean_legend(axes[1])

for name, curve, seconds, _ in curves:
    print(f"{name:26s}: negative ELBO {float(curve[-1]):8.2f} after {seconds:.2f} s")
print(
    "max gap between the two natural-gradient curves: "
    f"{float(jnp.max(jnp.abs(curves[0][1] - curves[1][1]))):.2f} nats"
)

# Sentinel above every attainable iteration index, so "never crossed" is
# distinguishable from "crossed on the last iteration".
adam_target = float(curves[2][1][-1])
never = num_iterations + 1
for name, curve, seconds, _ in curves[:2]:
    crossing = int(jnp.min(jnp.where(curve < adam_target, smoothed_iterations, never)))
    if crossing == never:
        print(f"{name:26s}: never reaches Adam's final value")
    else:
        print(
            f"{name:26s}: reaches Adam's {num_iterations}-iteration value at "
            f"iteration {crossing}, i.e. after "
            f"{crossing * seconds / num_iterations:.2f} s of the {curves[2][2]:.2f} s "
            "Adam spent"
        )

# %% [markdown]
# Both natural-gradient runs leave Adam behind per iteration, and both reach
# Adam's thousand-iteration bound in a fraction of the wall-clock time Adam
# needed for it — the crossings are printed above.
#
# What the timings do *not* show is a cheaper dual iteration. On this
# problem the two natural-gradient runs cost within a few percent of each
# other per iteration, with the dual one marginally the more expensive. The
# round trip the dual parameterisation avoids is $\mathcal{O}(M^3)$ at
# $M=50$, which is nothing next to the $\mathcal{O}(BM^2)$ marginals at
# $B=256$, so there is little to save here in the first place. (GPJax's dual
# step also evaluates the objective once more per iteration than it
# strictly needs to, so that `history[t]` means the same thing in both
# branches — but under `jit`, which is how `fit_natgrads` always runs, XLA
# normally folds that repeat away, and nothing measured here separates the
# two effects.) Adam et al. report their gains at $M=100$ with ten latent
# GPs and $N=70{,}000$, where the constant they save is a much larger share
# of the total. Take the numbers above as a measurement of this
# configuration on this CPU, not as a refutation or a confirmation of
# theirs.
#
# The two natural-gradient curves do *not* lie on top of each other, and the
# gap is far too large to be the $10^{-3}$ located a section ago. Up to that
# clip their E-steps are still the same iteration; what differs is that
# `fit_natgrads` interleaves an Adam step on the kernel hyperparameters and
# the inducing inputs, and the objective it differentiates for that step is
# `dual_elbo` in one run and `elbo` in the other. Those two have the same
# value and different hyperparameter gradients away from a converged
# E-step — and with $\gamma$ ramping up from $10^{-4}$, the E-step spends
# most of the first hundred iterations far from converged. From iteration 1
# onwards the two runs are optimising the same model from different
# hyperparameters, and they never rejoin.
#
# Which way does the divergence go? Here the dual run ends at the *higher*,
# that is worse, negative ELBO of the two; both final values are printed
# above. That is one seed, on a mini-batch bound, with the kernel
# hyperparameters and all fifty inducing inputs moving under a ramping
# $\gamma$ — the two runs sit at different $\boldsymbol{\theta}$ from
# iteration 1, so this is not a controlled comparison of the two M-step
# objectives and should not be read as one, in either direction. The
# controlled version — frozen inducing inputs, one kernel hyperparameter,
# matched E-steps — is the VEM run at the end of the notebook. What this
# figure does establish is that the choice of M-step objective changes the
# trajectory by tens of nats, which is why the rest of the notebook is about
# that choice.

# %% [markdown]
# ## The M-step in practice
#
# The natural-gradients notebook's claims table is the reference for what
# follows; we do not restate its proof here, only exercise it. In brief,
# `elbo` freezes the *whole* natural parameter of $q$ at its E-step optimum
# while $\boldsymbol{\theta}$ moves; `dual_elbo` freezes only the
# data-derived sites and lets the prior half of $q$ track
# $\mathbf{K}_{zz}(\boldsymbol{\theta})$. The two agree in value and
# gradient at a converged E-step, by the envelope theorem, and can disagree
# substantially away from it — which in practice is always, since nobody
# runs an E-step to convergence between Adam steps. Two questions follow.
# Does the dual bound actually dominate the standard one, as the paper's
# figure suggests? And does that translate into a better fitted model at the
# end of a real VEM loop?

# %%
log_offsets = jnp.linspace(-1.2, 0.6, 61)
frozen_sites, frozen_moments = exact_sites(
    regression_lengthscale, regression_inducing, regression_data
)


def bound_slice(inducing_inputs, dataset, sites, moments, offsets):
    """`dual_elbo` and `elbo` along a log-lengthscale slice, at frozen q."""
    dual_values, moment_values = [], []
    for offset in offsets:
        lengthscale = regression_lengthscale * jnp.exp(offset)
        dual_values.append(
            dual_elbo(
                paramax.unwrap(site_family(lengthscale, inducing_inputs, sites)),
                dataset,
            )
        )
        moment_values.append(
            elbo(
                paramax.unwrap(moment_family(lengthscale, inducing_inputs, moments)),
                dataset,
            )
        )
    return jnp.array(dual_values), jnp.array(moment_values)


dual_slice, moment_slice = bound_slice(
    regression_inducing, regression_data, frozen_sites, frozen_moments, log_offsets
)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0))
axes[0].plot(log_offsets, dual_slice, color=cols[2], label=r"$\bar l$ (dual_elbo)")
axes[0].plot(log_offsets, moment_slice, color=cols[1], label=r"$l$ (elbo)")
axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
axes[0].set(
    xlabel=r"$\Delta\log\ell$ from $\theta_t$",
    ylabel="Bound (nats)",
    ylim=(float(dual_slice.min()) - 40.0, float(dual_slice.max()) + 10.0),
    title=f"Sparse, $M = {num_inducing}$",
)
clean_legend(axes[0])

for inducing_count, colour in [(5, cols[0]), (10, cols[3]), (20, cols[2])]:
    sparse_inducing = jnp.linspace(-3.0, 3.0, inducing_count).reshape(-1, 1)
    sparse_sites, sparse_moments = exact_sites(
        regression_lengthscale, sparse_inducing, regression_data
    )
    sparse_dual, sparse_moment = bound_slice(
        sparse_inducing, regression_data, sparse_sites, sparse_moments, log_offsets
    )
    gap = sparse_dual - sparse_moment
    axes[1].plot(log_offsets, gap, color=colour, label=f"$M = {inducing_count}$")
    print(
        f"M = {inducing_count:2d}: smallest gap {float(gap.min()):+.4e} nats at "
        f"delta log-lengthscale {float(log_offsets[jnp.argmin(gap)]):+.3f}"
    )
axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[1].set(
    xlabel=r"$\Delta\log\ell$ from $\theta_t$",
    ylabel=r"$\bar l - l$ (nats)",
    yscale="symlog",
    title="Dominance is not uniform when sparse",
)
clean_legend(axes[1])

# %% [markdown]
# Left panel: both bounds pass through the same point at
# $\boldsymbol{\theta}_t$, which is the value-equality row of the claims
# table. To the *right*, at longer lengthscales, $l$ falls off a cliff — a
# $q$ chosen for one kernel is a bad approximation under another — while
# $\bar l$ has barely moved, since only the data-dependent half of it was
# frozen. To the *left* the two collapse together instead, since the sparse
# approximation itself degrades there regardless of which $q$ is frozen.
# Since the M-step travels rightwards out of a too-short lengthscale, the
# asymmetry is the useful half: in the direction of travel, an M-step on
# $\bar l$ can go much further before the bound it is climbing stops being
# informative.
#
# Right panel: at $M=20$ the dual bound dominates everywhere probed, up to
# float64 noise at the crossing point (the value-equality point itself). At
# $M=5$ and $M=10$ the gap dips below zero for real — printed above — so the
# dominance guarantee, which the claims table restricts to $\mathbf{Z}=\mathbf{X}$,
# genuinely does not extend to every sparse configuration. (At
# $\mathbf{Z}=\mathbf{X}$ itself the flanked sites reduce to
# $\boldsymbol{\lambda}_1=(\mathbf{y}-\boldsymbol{\mu})/\sigma^2$,
# $\boldsymbol{\Lambda}_2=\mathbf{I}/\sigma^2$ — genuinely
# $\boldsymbol{\theta}$-free — and dominance holds everywhere; that is a
# corollary of the claims table, not a new demo, so we do not reproduce it
# here.)

# %% [markdown]
# Bound slices are static. The claim that actually matters is that a real
# VEM loop gets further with `dual_elbo` as its M-step objective, and that
# one is empirical — the paper says as much, and so does the claims table.
# So we run it. Both branches share the same E-step — the same iteration, up
# to the `beta_floor` clip located earlier — and differ only in what the
# M-step differentiates. The inducing inputs are frozen so that only the
# kernel moves, and the lengthscale starts five times too short.

# %%
expectation_steps = 20
maximisation_steps = 5
vem_rounds = 40
vem_rate = 0.5
vem_optimiser = ox.adam(5e-2)
initial_lengthscale = 0.25


def freeze_inducing(model):
    """Hold the inducing inputs still, so the M-step moves only the kernel."""
    return eqx.tree_at(
        lambda tree: tree.inducing_inputs,
        model,
        paramax.non_trainable(model.inducing_inputs),
    )


def vem_joint_model(lengthscale):
    return (
        gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=jk.RBF(active_dims=[0, 1], lengthscale=lengthscale),
            jitter=banana_jitter,
        )
        * gpx.likelihoods.Bernoulli()
    )


vem_gram = paramax.unwrap(vem_joint_model(initial_lengthscale)).prior.kernel.gram(
    banana_inducing
).as_matrix() + banana_jitter * jnp.eye(num_banana_inducing)

vem_dual = freeze_inducing(
    DualVariationalGaussian(
        model=vem_joint_model(initial_lengthscale),
        inducing_inputs=banana_inducing,
    )
)
vem_moments = freeze_inducing(
    VariationalGaussian(
        model=vem_joint_model(initial_lengthscale),
        inducing_inputs=banana_inducing,
        variational_mean=jnp.zeros((num_banana_inducing, 1)),
        variational_root_covariance=jnp.linalg.cholesky(vem_gram),
    )
)


def run_vem(model, objective):
    """Alternate `expectation_steps` E-steps with `maximisation_steps` M-steps."""
    variational, hyper = partition_variational(model)
    opt_state = vem_optimiser.init(eqx.filter(hyper, eqx.is_array))

    @eqx.filter_jit
    def expectation_step(variational, hyper):
        def body(carry, _):
            updated, _ = natural_gradient_step(
                carry, hyper, banana_train, objective, vem_rate
            )
            return updated, None

        return jax.lax.scan(body, variational, None, length=expectation_steps)[0]

    @eqx.filter_jit
    def maximisation_step(variational, hyper, opt_state):
        def hyper_loss(hyper):
            return objective(
                paramax.unwrap(eqx.combine(variational, hyper)), banana_train
            )

        def body(carry, _):
            hyper, opt_state = carry
            loss, gradient = eqx.filter_value_and_grad(hyper_loss)(hyper)
            updates, opt_state = vem_optimiser.update(
                gradient, opt_state, eqx.filter(hyper, eqx.is_array)
            )
            return (eqx.apply_updates(hyper, updates), opt_state), loss

        (hyper, opt_state), losses = jax.lax.scan(
            body, (hyper, opt_state), None, length=maximisation_steps
        )
        return hyper, opt_state, losses[-1]

    lengthscales, bounds = [], []
    for _ in range(vem_rounds):
        variational = expectation_step(variational, hyper)
        hyper, opt_state, loss = maximisation_step(variational, hyper, opt_state)
        combined = paramax.unwrap(eqx.combine(variational, hyper))
        lengthscales.append(float(combined.model.prior.kernel.lengthscale))
        bounds.append(float(loss))
    return eqx.combine(variational, hyper), jnp.array(lengthscales), jnp.array(bounds)


dual_vem_model, dual_lengthscales, dual_bounds = run_vem(vem_dual, negative_dual_elbo)
moment_vem_model, moment_lengthscales, moment_bounds = run_vem(
    vem_moments, negative_elbo
)

# %%
fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0))
rounds = jnp.arange(1, vem_rounds + 1)
for name, lengthscales, colour in [
    ("M-step on dual_elbo", dual_lengthscales, cols[2]),
    ("M-step on elbo", moment_lengthscales, cols[1]),
]:
    axes[0].plot(rounds, lengthscales, color=colour, label=name)
axes[0].set(xlabel="VEM round", ylabel=r"Lengthscale $\ell$")
clean_legend(axes[0])

# The two bound traces are visually identical at this scale, so plot their difference:
# positive means the dual M-step is the further down the negative ELBO of the two.
bound_lead = moment_bounds - dual_bounds
axes[1].plot(rounds, bound_lead, color=cols[2])
axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[1].set(
    xlabel="VEM round",
    ylabel="Bound lead to dual_elbo (nats)",
    title="Lead of the dual M-step over the standard one",
)


def test_metrics(model):
    """Held-out accuracy and negative log predictive density."""
    unwrapped = paramax.unwrap(model)
    probability = unwrapped.model.likelihood(unwrapped(test_inputs_2d)).mean
    labels = test_labels.ravel()
    log_density = jnp.mean(
        labels * jnp.log(probability) + (1.0 - labels) * jnp.log1p(-probability)
    )
    return float(jnp.mean((probability > 0.5) == (labels > 0.5))), float(-log_density)


for name, model, lengthscales, bounds in [
    ("dual_elbo", dual_vem_model, dual_lengthscales, dual_bounds),
    ("elbo     ", moment_vem_model, moment_lengthscales, moment_bounds),
]:
    accuracy, nlpd = test_metrics(model)
    print(
        f"M-step on {name}: lengthscale {float(lengthscales[-1]):.4f}, "
        f"negative ELBO {float(bounds[-1]):8.3f}, "
        f"test accuracy {accuracy:.4f}, test NLPD {nlpd:.4f}"
    )
print(
    f"bound lead to dual_elbo over {vem_rounds} rounds: "
    f"smallest {float(bound_lead.min()):+.3f}, largest {float(bound_lead.max()):+.3f}, "
    f"final {float(bound_lead[-1]):+.3f} nats"
)
# The last round the lead is non-positive, so we can report where it becomes
# permanent rather than where it first (and only briefly) turns positive.
last_nonpositive_round = int(jnp.max(jnp.where(bound_lead <= 0.0, rounds, 0)))
if last_nonpositive_round == 0:
    print("dual_elbo leads for the whole run")
else:
    remaining = vem_rounds - last_nonpositive_round
    print(
        f"last round with a non-positive lead: {last_nonpositive_round}; "
        f"the lead stays positive for all {remaining} rounds after that"
    )

# %% [markdown]
# The two lengthscale traces sit on top of each other for the first several
# rounds and then separate, with the dual branch ending the longer of the
# two; both final values are printed above. The right panel plots the
# difference of the two bounds rather than the two bounds themselves,
# because on a negative ELBO of a few hundred a lead of a nat is invisible
# on the natural scale.
#
# The sign is not settled early. For the first two rounds the dual bound
# leads by a few tenths of a nat, then falls behind for rounds three through
# six — both lengthscales are still moving fast and $q$ is far from
# converged at every round, exactly the regime where the two M-step
# objectives are proven to differ — bottoming out at $-1.15$ nats. The round
# printed above is the *last* one with a non-positive lead; every round
# after it is positive, rising to a peak around $+1.27$ nats and easing back
# to the final value printed above. So the honest reading is not "the
# dual M-step is uniformly ahead"; early on it is not. It is that the two
# branches take different routes to the same place: after forty rounds the
# dual one has the longer lengthscale and the marginally better bound, and
# their held-out metrics agree closely. That is consistent with the theory,
# which promises equality at convergence and says nothing about the rate —
# "exact theoretical reasons behind the speed-ups are currently unknown to
# us."
#
# It is also a soft result on a two-dimensional problem with one kernel
# hyperparameter and fifty fixed inducing points. The regime the paper
# reports gains in — many latent GPs, large $N$, mini-batched,
# hyperparameters far from their optimum — is not this one. Read the demo as
# a mechanism check rather than as a benchmark, and if you want the
# mechanism in one sentence: at an incomplete E-step the two objectives have
# different hyperparameter gradients, and the dual one is the gradient of a
# function that still knows the prior depends on $\boldsymbol{\theta}$.

# %% [markdown]
# ## Caveats
#
# * **One latent process.** Everything above assumes $L=1$. The site
#   structure across multiple latent GPs is block diagonal only when the
#   variational family is itself latent diagonal, and the tied projection
#   has to be re-derived rather than reused for a multi-output model.
# * **$\beta_i \ge 0$ needs a log-concave likelihood — as *computed*, not as
#   written.** Student-$t$ and some heteroscedastic likelihoods are not
#   log-concave at all, and for those the target can push
#   $\boldsymbol{\Lambda}_2$ out of the PSD cone. Less obviously, GPJax's
#   Bernoulli joins them in the far tails: `inv_probit` clips its output
#   into $[10^{-3},\,1-10^{-3}]$, which flattens $\log p$ and makes its
#   second derivative positive for $f \lesssim -2.44$, so a confidently
#   mislabelled point yields $\beta_i < 0$. The `beta_floor` keyword
#   (default $10^{-8}$) clips $\boldsymbol{\beta}$ from below and keeps the
#   step inside the cone. It is *not* a no-op for Bernoulli — it is what
#   broke the $\rho=\gamma$ identity above, by $\sim\!5\times10^{-3}$ in
#   $(\mathbf{m},\mathbf{S})$. Note that it clips $\boldsymbol{\beta}$,
#   never $\boldsymbol{\Lambda}_2$: the update stays affine, so it stays
#   `jit`- and `scan`-safe.
# * **$\rho \in (0,1]$.** The convex-combination guarantee stops at $1$, and
#   beyond it the step extrapolates past a target that is only locally
#   valid. `fit_natgrads` rejects a larger constant rate for this family at
#   call time.
# * **Flanked storage squares $\operatorname{cond}(\mathbf{K}_{zz})$.**
#   Benign in everything measured here at the level of $\mathbf{R}$, the
#   moments and the bound; never write a test against
#   $\boldsymbol{\Lambda}_2$ directly — see the natural-gradients notebook
#   for the numerical demonstration of why.
# * **The E-step is not a free lunch.** Wherever the computed $\beta_i$ stay
#   non-negative it is the *same iteration* as the natural gradient step on
#   $(\mathbf{m},\mathbf{L})$, and where they do not, the difference is the
#   clip above, not a better search direction. Whatever the dual
#   parameterisation buys is either wall-clock per iteration or M-step
#   behaviour; none of it is a better $q$ at the same $\boldsymbol{\theta}$.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
