# -*- coding: utf-8 -*-
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
#     display_name: gpjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Graph Kernels
#
# This notebook demonstrates how regression models can be constructed on the vertices
# of a graph using a Gaussian process with a Matérn kernel presented in
# <strong data-cite="borovitskiy2021matern"></strong>. For a general discussion of the
# kernels supported within GPJax, see the
# [kernels notebook](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels).

# %%
import random

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from examples.utils import use_mpl_style

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.parameters import Parameter


# set the default style for plotting
use_mpl_style()

key = jr.key(42)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Graph construction
#
# Our graph $\mathcal{G}=\lbrace V, E \rbrace$ comprises a set of vertices
# $V = \lbrace v_1, v_2, \ldots, v_n\rbrace$ and edges
# $E=\lbrace (v_i, v_j)\in V \ : \ i \neq j\rbrace$. In particular, we will consider
# a [barbell graph](https://en.wikipedia.org/wiki/Barbell_graph) that is an undirected
# graph containing two clusters of vertices with a single shared edge between the
# two clusters.
#
# Contrary to the typical barbell graph, we'll randomly remove a subset of 30 edges
# within each of the two clusters. Given the 40 vertices within the graph, this results
# in 351 edges as shown below.

# %%
vertex_per_side = 20
n_edges_to_remove = 30
p = 0.8

G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), n_edges_to_remove)]

pos = nx.spring_layout(G, seed=123)  # positions for all nodes

nx.draw(
    G, pos, node_size=100, node_color=cols[1], edge_color="black", with_labels=False
)

# %% [markdown]
#
# ### Computing the graph Laplacian
#
# Graph kernels use the _Laplacian matrix_ $L$ to quantify the smoothness of a signal
# (or function) on a graph
#
# $$L=D-A,$$
#
# where $D$ is the diagonal _degree matrix_ containing each vertices' degree and $A$
# is the _adjacency matrix_ that has an $(i,j)^{\text{th}}$ entry of 1 if $v_i, v_j$
# are connected and 0 otherwise. [Networkx](https://networkx.org) gives us an easy
# way to compute this.

# %%
L = nx.laplacian_matrix(G).toarray()

# %% [markdown]
#
# ## Simulating a signal on the graph
#
# Our task is to construct a Gaussian process $f(\cdot)$ that maps from the graph's
# vertex set $V$ onto the real line.
# To that end, we begin by simulating a signal on the graph's vertices that we will go
# on to try and predict.
# We use a single draw from a Gaussian process prior to draw our response values
# $\boldsymbol{y}$ where we hardcode parameter values.
# The corresponding input value set for this model, denoted $\boldsymbol{x}$, is the
# index set of the graph's vertices.

# %%
x = jnp.arange(G.number_of_nodes()).reshape(-1, 1)

true_kernel = gpx.kernels.GraphKernel(
    laplacian=L,
    lengthscale=2.3,
    variance=3.2,
    smoothness=6.1,
)
prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=true_kernel)

fx = prior(x)
y = fx.sample(key=key, sample_shape=(1,)).reshape(-1, 1)

D = gpx.Dataset(X=x, y=y)

# %% [markdown]
#
# We can visualise this signal in the following cell.

# %%
nx.draw(G, pos, node_color=y, with_labels=False, alpha=0.5)

vmin, vmax = y.min(), y.max()
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.inferno, norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm.set_array([])
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)

# %% [markdown]
#
# ## Constructing a graph Gaussian process
#
# With our dataset created, we proceed to define our posterior Gaussian process and
# optimise the model's hyperparameters.
# Whilst our underlying space is the graph's vertex set and is therefore
# non-Euclidean, our likelihood is still Gaussian and the model is still conjugate.
# For this reason, we simply perform gradient descent on the GP's marginal
# log-likelihood term as in the
# [regression notebook](https://docs.jaxgaussianprocesses.com/_examples/regression/).
# We do this using the BFGS optimiser.

# %%
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
kernel = gpx.kernels.GraphKernel(laplacian=L)
prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
posterior = prior * likelihood

# %% [markdown]
#
# For researchers and the curious reader, GPJax provides the ability to print the
# bibtex citation for objects such as the graph kernel through the `cite()` function.

# %%
print(gpx.cite(kernel))

# %% [markdown]
#
# With a posterior defined, we can now optimise the model's hyperparameters.

# %%
opt_posterior, training_history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    trainable=Parameter,
)

# %% [markdown]
#
# ## Making predictions
#
# Having optimised our hyperparameters, we can now make predictions on the graph.
# Though we haven't defined a training and testing dataset here, we'll simply query
# the predictive posterior for the full graph to compare the root-mean-squared error
# (RMSE) of the model for the initialised parameters vs the optimised set.

# %%
initial_dist = likelihood(posterior(x, D))
predictive_dist = opt_posterior.likelihood(opt_posterior(x, D))

initial_mean = initial_dist.mean
learned_mean = predictive_dist.mean

rmse = lambda ytrue, ypred: jnp.sum(jnp.sqrt(jnp.square(ytrue - ypred)))

initial_rmse = jnp.sum(jnp.sqrt(jnp.square(y.squeeze() - initial_mean)))
learned_rmse = jnp.sum(jnp.sqrt(jnp.square(y.squeeze() - learned_mean)))
print(
    f"RMSE with initial parameters: {initial_rmse: .2f}\nRMSE with learned parameters:"
    f" {learned_rmse: .2f}"
)

# %% [markdown]
#
# We can also plot the source of error in our model's predictions on the graph by the
# following.

# %%
error = jnp.abs(learned_mean - y.squeeze())

nx.draw(G, pos, node_color=error, with_labels=False, alpha=0.5)

vmin, vmax = error.min(), error.max()
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.inferno, norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)

# %% [markdown]
#
# Reassuringly, our model seems to provide equally good predictions in each cluster.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
