from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import gpjax as gpx
from gpjax.citation import (
    AbstractCitation,
    NullCitation,
    PaperCitation,
    PhDThesisCitation,
    cite,
)
from gpjax.kernels import (
    RBF,
    RFF,
    ArcCosine,
    GraphKernel,
    Linear,
    Matern12,
    Matern32,
    Matern52,
)
from gpjax.likelihoods import HeteroscedasticGaussian
from gpjax.mean_functions import Zero


def _check_no_fallback(citation: AbstractCitation):
    # Check the fallback has not been used
    assert repr(citation) != repr(
        "No citation available. If you think this is an error, please open a pull"
        " request."
    )
    assert (
        str(citation)
        != "No citation available. If you think this is an error, please open a pull"
        " request."
    )


@pytest.mark.parametrize("kernel", [Matern12(), Matern32(), Matern52()])
def test_matern_kernels(kernel):
    citation = cite(kernel)
    # Check type
    assert isinstance(citation, PhDThesisCitation)
    # Check some fields
    assert citation.authors == "Bertil Matérn"
    assert citation.year == "1960"
    assert citation.institution == "Stockholm University"

    _check_no_fallback(citation)


def test_arc_cosine():
    kernel = ArcCosine()
    citation = cite(kernel)

    assert isinstance(citation, PaperCitation)

    assert citation.citation_key == "cho2009kernel"
    assert citation.title == "Kernel Methods for Deep Learning"
    assert citation.authors == "Cho, Youngmin and Saul, Lawrence"
    _check_no_fallback(citation)


def test_graph_kernel():
    L = jnp.eye(3)
    kernel = GraphKernel(
        laplacian=L,
    )
    citation = cite(kernel)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "borovitskiy2021matern"
    assert citation.title == "Matérn Gaussian Processes on Graphs"
    assert citation.year == "2021"
    _check_no_fallback(citation)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_rff(kernel):
    base_kernel = kernel(n_dims=1)
    rff = RFF(base_kernel=base_kernel)
    citation = cite(rff)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "rahimi2007random"
    assert citation.title == "Random features for large-scale kernel machines"
    assert citation.authors == "Rahimi, Ali and Recht, Benjamin"
    assert citation.year == "2007"
    _check_no_fallback(citation)


@pytest.mark.parametrize("kernel", [RBF(), Linear()])
def test_missing_citation(kernel):
    citation = cite(kernel)
    assert isinstance(citation, NullCitation)


def test_heteroscedastic_citation():
    noise_prior = gpx.gps.Prior(mean_function=Zero(), kernel=RBF())
    likelihood = HeteroscedasticGaussian(num_datapoints=10, noise_prior=noise_prior)
    citation = cite(likelihood)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "lazaro2011variational"
    assert citation.title == "Variational heteroscedastic Gaussian process regression"
    assert citation.authors == "Lázaro-Gredilla, Miguel and Titsias, Michalis"
    assert citation.year == "2011"
    _check_no_fallback(citation)
