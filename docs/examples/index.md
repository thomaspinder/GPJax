# Examples

Every example on this page is an executable `py:percent` notebook under
`docs/examples/`, run at build time by MyST-NB. API names in the code cells
link straight into the [API reference](../reference/index.md).

Start with **New to Gaussian Processes?** if you are new to GPs, then
**Regression** for the canonical end-to-end workflow.

## Background

| Notebook | What you'll learn |
|----------|-------------------|
| [New to Gaussian Processes?](intro_to_gps.py) | Priors, posteriors and the marginal likelihood from first principles |
| [Introduction to Kernels](intro_to_kernels.py) | What a kernel encodes, and how composition changes the prior |

## Tutorials

| Notebook | What you'll learn |
|----------|-------------------|
| [Regression](regression.py) | Conjugate GP regression: prior, likelihood, posterior, hyperparameter fitting |
| [Classification](classification.py) | Non-conjugate inference with a Bernoulli likelihood, Laplace and MCMC |
| [Count data regression](poisson.py) | Poisson likelihood and latent-function inference |
| [Gaussian Processes Barycentres](barycentres.py) | Wasserstein barycentres of GP posteriors |
| [Deep Kernel Learning](deep_kernels.py) | Composing a neural network feature map with a GP kernel |
| [Graph Kernels](graph_kernels.py) | GPs on non-Euclidean domains via the graph Laplacian |
| [Sparse Gaussian Process Regression](collapsed_vi.py) | Collapsed variational inference with inducing points |
| [Sparse Stochastic Variational Inference](uncollapsed_vi.py) | Uncollapsed SVGP and minibatch training |
| [State-Space (Markovian) Gaussian Processes](state_space_gps.py) | Linear-time inference via the SDE representation |
| [Gaussian Processes for Vector Fields and Ocean Current Modelling](oceanmodelling.py) | Vector-valued GPs on real ocean-current data |
| [Heteroscedastic Inference](heteroscedastic_inference.py) | Input-dependent noise with a chained GP |
| [Multi-Output Gaussian Processes](multioutput.py) | ICM and LCM kernels for correlated outputs |
| [Scalable Multi-Output GPs with OILMM](oilmm.py) | Orthogonal instantaneous linear mixing for many outputs |
| [Orthogonal Additive Kernels](oak.py) | Decomposing a GP into interpretable additive components |
| [Joint Inference with Numpyro](numpyro_integration.py) | MCMC over GP hyperparameters through NumPyro |
| [Spatial Modelling with Composable Gaussian Processes](spatial_linear_gp.py) | Combining a linear trend with a spatial GP |

## Guides for customisation

| Notebook | What you'll learn |
|----------|-------------------|
| [Kernel Guide](constructing_new_kernels.py) | Writing your own kernel against `AbstractKernel` |
| [Likelihood guide](likelihoods_guide.py) | Writing your own likelihood and integrator |
| [Backend Module Design](backend.py) | How GPJax modules, parameters and pytrees fit together |
| [UCI Data Benchmarking](yacht.py) | A full benchmarking workflow on a UCI dataset |

```{toctree}
:hidden:
:maxdepth: 1

intro_to_gps
intro_to_kernels
regression
classification
poisson
barycentres
deep_kernels
graph_kernels
collapsed_vi
uncollapsed_vi
state_space_gps
oceanmodelling
heteroscedastic_inference
multioutput
oilmm
oak
numpyro_integration
spatial_linear_gp
constructing_new_kernels
likelihoods_guide
backend
yacht
```
