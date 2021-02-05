# GPJax

[![codecov](https://codecov.io/gh/thomaspinder/gpjax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/thomaspinder/gpjax)
[![CodeFactor](https://www.codefactor.io/repository/github/thomaspinder/gpjax/badge)](https://www.codefactor.io/repository/github/thomaspinder/gpjax)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)

GPJax aims to provide a low-level interface to Gaussian process models. Code is written entirely in Jax and Objax to enhance readability, and structured so as to allow researchers to easily extend the code to suit their own needs. When defining GP prior in GPJax, the user need only specify a mean and kernel function. A GP posterior can then be realised by computing the product of our prior with a likelihood function. The idea behind this is that the code should be as close as possible to the maths that we would write on paper when working with GP models.

## Simple example

After importing the necessary dependencies, we'll first simulate some data. 
```python
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(123)

X = jnp.linspace(-2., 2., 100)
y = jnp.sin(x) + jr.normal(key, shape=X.shape)*0.05
```

As can be seen, the latent function of interest here is a sinusoidal function. However, it has been perturbed by some zero-mean Gaussian noise with variance of 0.05. We can use a Gaussian process model to try and recover this latent function.

```python
from gpjax import Prior, RBF

kernel = RBF(lengthscale=1.0, variance=1.0)
f = Prior(kernel)
```
 
In the presence of a likelihood function which we'll here assume to be Gaussian, we can optimise the marginal log-likelihood of the Gaussian process prior multiplied by the likelihood to obtain a posterior distribution over the latent function.

```python
from gpjax import Gaussian

likelihood = Gaussian()
posterior = f * likelihood
```

Equipped with the Gaussian process posterior, we can now optimise the model's hyperparameters (note, we need not optimise the latent function here due to the Gaussian conjugacy.). Through Objax, this procedure is straightforward, as we can access all of the GP's hyperparameters using the `.vars()` method. Using the Objax provided optimisers, we can then perform gradient-based optimisation on the marginal log-likelihood with respect to the hyperparameters.

```python
import objax
hyperparameters = posterior.vars()
opt = objax.optimizer.SGD(hyperparameters)
gv = objax.GradValues(posterior.neg_mll, hyperparameters) 

def train_op(x, label):
    g, v = gv(x, label)
    opt(0.01, g)
    return v
```

We can also Jit compile our objective function to accelerate training. You'll notice that it is only now that we have incorporated any data into our GP. This is desirable, as this is exactly how model building works in principle too, where we first build our prior model, then observe some data and use this data to build a posterior. 

```python
nits = 100
train_op = objax.Jit(train_op, gv.vars() + opt.vars())
loss = [train_op(X, y.squeeze())[0].item() for _ in range(nits)]
```

With an optimised set of hyperparameters, we can now sample from the posterior predictive distribution to make predictions at a set of unseen test points.

```python
Xtest = jnp.linspace(X.min()-0.1, X.max()+0.1, 500)
mu, cov = posterior.predict(Xtest, X, y.squeeze())
```

## Installation

### Stable version

To install the latest stable version of gpjax run
```bash
pip install gpjax
```
Note, this will require a Python version >= 3.7.

### Development version

To install the lastest, possibly unstable, version, the following steps should be followed. It is by no means compulsory, but we do advise that you do all of the below inside a virtual environment.

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax 
python setup.py develop
```

## To do

* Spectral kernels ([in progress](https://github.com/thomaspinder/GPJax/tree/spectral))
* Inducing point schemes ([in progress](https://github.com/thomaspinder/GPJax/tree/inducing_points))
