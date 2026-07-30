# Installation

## Stable version

The latest stable release of `GPJax` can be installed from [PyPI](https://pypi.org/project/gpjax/):

```bash
pip install gpjax
```

or from [conda-forge](https://github.com/conda-forge/gpjax-feedstock) with [Pixi](https://pixi.sh/):

```bash
pixi add gpjax
```

or with [conda](https://docs.conda.io/projects/conda/):

```bash
conda install --channel conda-forge gpjax
```

:::{note} Check your installation
We recommend you check your installation version:

```bash
python -c 'import gpjax; print(gpjax.__version__)'
```
:::

## GPU/TPU support

Fancy using GPJax on GPU/TPU? Then you'll need to install JAX with the relevant
hardware acceleration support as detailed in the
[JAX installation guide](https://github.com/google/jax#installation).


## Development version

:::{warning}
This version is possibly unstable and may contain bugs.

The latest development version of `GPJax` can be installed via running following:

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
uv sync
```
:::

:::{tip}
We advise you create virtual environment before installing:

```bash
conda create -n gpjax_experimental python=3.11.0
conda activate gpjax_experimental
```

and recommend you check your installation passes the supplied unit tests:

```bash
uv run poe all-tests
```
:::
