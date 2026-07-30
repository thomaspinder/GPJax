"""Sphinx configuration for the GPJax documentation.

Single-engine build: MyST-NB parses and executes the example notebooks, Sphinx
renders everything. No MkDocs, no mkdocstrings, no markdown post-processing.

Ported from ../impulso/docs/conf.py. Deviations from that file are commented
inline and are limited to (a) names/URLs, (b) `bibtex_default_style`,
(c) `default_role`, (d) `exclude_patterns`.
"""

from __future__ import annotations

import os

# -- Project information -----------------------------------------------------
project = "GPJax"
author = "Thomas Pinder"
copyright = "2022-2026, The GPJax Contributors"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",  # MyST markdown + executable notebooks (enables myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google-style docstrings
    # Renders $...$ / $$...$$ inside docstrings. MyST's dollarmath covers .md and
    # notebooks but never reaches docstrings, which napoleon emits as RST. Must
    # precede sphinx.ext.mathjax.
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_codeautolink",  # link API names in example code to the reference
    "sphinx_sitemap",  # emit sitemap.xml
    "sphinx_reredirects",  # meta-refresh stubs for the retired MkDocs URLs
]

# -- Bibliography (sphinxcontrib-bibtex) -------------------------------------
bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"  # {cite:t} -> Rasmussen and Williams (2006)
# DEVIATION from impulso, which pairs `author_year` with `unsrt`. `unsrt` emits a
# numbered [1][2][3] reference list, so its prose citations read "Titsias (2009)"
# against numbers that match nothing. `plain` renders an author-sorted, non-numbered
# list, which is what an author-year reference style is meant to point into.
bibtex_default_style = "plain"

templates_path = ["_templates"]

# Examples are jupytext py:percent notebooks; MyST-NB reads them via jupytext.
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".py": "myst-nb",
    ".rst": "restructuredtext",  # autosummary-generated API stubs
}
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

# DEVIATION from impulso, which leaves `default_role` unset. Unset, RST resolves a
# single-backtick span to `title-reference`, so `Prior` renders as italic <cite>
# rather than code — impulso's own built reference has 350 such <cite> elements.
# GPJax has 108 single-backtick docstring spans; `literal` renders them as code.
default_role = "literal"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "conf.py",  # this config module is not a document
    "**.ipynb_checkpoints",
    # Underscore-prefixed .py under examples/ are helper and data-pull modules,
    # not notebooks — keep MyST-NB from collecting them as standalone documents
    # (they'd warn as orphan pages). This is impulso's `tutorials/_*.py` rule.
    "examples/_*.py",
    "examples/**/_*.py",
    # GPJax's notebook helper predates that naming convention and is imported by
    # name (`from utils import use_mpl_style`), so it cannot simply be renamed.
    "examples/utils.py",
    # Image/style assets, plus one stray legacy module (static/jaxkern/main.py)
    # that source_suffix would otherwise read as a notebook.
    "static/**",
]

# -- MyST / MyST-NB ----------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$
    "amsmath",  # \begin{align} etc.
    "colon_fence",  # ::: {note} admonitions
    "deflist",
    "tasklist",
    "html_image",
    "attrs_inline",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3  # auto-slug headings so in-page [](#anchor) links resolve

# Execute notebooks and cache results in a gitignored cache. Source .py carry
# no outputs; a cell re-runs only when its code changes.
nb_execution_mode = "cache"
# Smoke (GPJAX_DOCS_CI=1) and full renders execute the SAME notebook source but
# with very different optimiser/MCMC budgets — yet jupyter-cache keys on source
# alone, so a shared cache lets a tiny smoke run poison the full-fidelity build.
# Separate the cache directories so the two modes can never overwrite one
# another (locally or in CI).
_smoke_render = os.environ.get("GPJAX_DOCS_CI") == "1"
nb_execution_cache_path = os.path.join(
    os.path.dirname(__file__),
    "_build",
    ".jupyter_cache_ci" if _smoke_render else ".jupyter_cache",
)
nb_execution_timeout = 1800  # heavy MCMC / sparse-GP notebooks
# Strict by default (PR gate + local): a failed cell fails the build. On the
# deploy path (GPJAX_DOCS_RESILIENT=1) we do NOT raise, so one slow/broken
# notebook cannot block the whole site — it renders an error cell while every
# other page (and the last-good cached output) still deploys.
nb_execution_raise_on_error = os.environ.get("GPJAX_DOCS_RESILIENT") != "1"
nb_merge_streams = True

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Numbered figures, tables and equations ----------------------------------
numfig = True
math_eqref_format = "Eq. {number}"

# -- sphinx-codeautolink -----------------------------------------------------
# Adds a "Examples using …" backreference block to each documented object.
codeautolink_autodoc_inject = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "shibuya"
html_title = "GPJax"
html_baseurl = "https://docs.jaxgaussianprocesses.com/"  # for sitemap + canonical
sitemap_url_scheme = "{link}"
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
html_favicon = "static/favicon.ico"
# GPJax serves the docs from a custom domain, so GitHub Pages needs a CNAME file
# at the site root. impulso has no counterpart: it deploys to a github.io path.
html_extra_path = ["CNAME"]
html_theme_options = {
    "accent_color": "crimson",  # radix name closest to the brand #b71c1c
    "color_mode": "auto",  # follow the reader's light/dark preference
    "github_url": "https://github.com/thomaspinder/GPJax",
    "nav_links": [
        {"title": "PyPI", "url": "https://pypi.org/project/gpjax"},
    ],
}
html_context = {
    "github_user": "thomaspinder",
    "github_repo": "GPJax",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Redirects from the retired MkDocs site (sphinx-reredirects) -------------
# docs.jaxgaussianprocesses.com served the MkDocs site with `use_directory_urls`
# (the default), so every old page lived at `<path>/`, i.e. the file
# `<path>/index.html`. The keys below therefore end in `/index`, which is what
# puts the emitted meta-refresh stub exactly where the old URL pointed.
#
# Targets are written site-absolute; sphinx-reredirects rewrites each one into
# the right number of `../` hops for the depth of its own key.
#
# Two families of URL moved:
#
# 1. API reference. `docs/scripts/gen_pages.py` walked
#    `sorted(Path("gpjax").rglob("*.py"))` and emitted one mkdocstrings page per
#    module at `/api/<module path>/`. There is no per-module page any more: the
#    reference is now hand-written per top-level package with autosummary stubs
#    underneath. Each old module page therefore points at the reference page for
#    its top-level package -- the most specific page that still exists -- rather
#    than at the reference index.
#
# 2. Tutorials. Notebooks moved from `/_examples/<name>/` to
#    `/examples/<name>.html`. Heading anchors survive the move
#    (`myst_heading_anchors = 3` reproduces the MkDocs slugs), and the stub
#    forwards `window.location.hash`, so deep links such as
#    `/_examples/classification/#laplace-approximation` still land correctly.
redirects = {
    # -- 1. API reference: /api/<module>/ -> /reference/<package>.html --------
    "api/citation/index":                               "/reference/citation.html",
    "api/dataset/index":                                "/reference/dataset.html",
    "api/distributions/index":                          "/reference/distributions.html",
    "api/fit/index":                                    "/reference/fit.html",
    "api/gps/index":                                    "/reference/gps.html",
    "api/integrators/index":                            "/reference/integrators.html",
    "api/kernels/additive/decompose/index":             "/reference/kernels.html",
    "api/kernels/additive/oak/index":                   "/reference/kernels.html",
    "api/kernels/additive/sobol/index":                 "/reference/kernels.html",
    "api/kernels/additive/transforms/index":            "/reference/kernels.html",
    "api/kernels/approximations/rff/index":             "/reference/kernels.html",
    "api/kernels/base/index":                           "/reference/kernels.html",
    "api/kernels/computations/base/index":              "/reference/kernels.html",
    "api/kernels/computations/basis_functions/index":   "/reference/kernels.html",
    "api/kernels/computations/constant_diagonal/index": "/reference/kernels.html",
    "api/kernels/computations/dense/index":             "/reference/kernels.html",
    "api/kernels/computations/diagonal/index":          "/reference/kernels.html",
    "api/kernels/computations/eigen/index":             "/reference/kernels.html",
    "api/kernels/multioutput/base/index":               "/reference/kernels.html",
    "api/kernels/multioutput/computation/index":        "/reference/kernels.html",
    "api/kernels/multioutput/icm/index":                "/reference/kernels.html",
    "api/kernels/multioutput/lcm/index":                "/reference/kernels.html",
    "api/kernels/non_euclidean/graph/index":            "/reference/kernels.html",
    "api/kernels/non_euclidean/utils/index":            "/reference/kernels.html",
    "api/kernels/nonstationary/arccosine/index":        "/reference/kernels.html",
    "api/kernels/nonstationary/linear/index":           "/reference/kernels.html",
    "api/kernels/nonstationary/polynomial/index":       "/reference/kernels.html",
    "api/kernels/stationary/base/index":                "/reference/kernels.html",
    "api/kernels/stationary/matern12/index":            "/reference/kernels.html",
    "api/kernels/stationary/matern32/index":            "/reference/kernels.html",
    "api/kernels/stationary/matern52/index":            "/reference/kernels.html",
    "api/kernels/stationary/periodic/index":            "/reference/kernels.html",
    "api/kernels/stationary/powered_exponential/index": "/reference/kernels.html",
    "api/kernels/stationary/rational_quadratic/index":  "/reference/kernels.html",
    "api/kernels/stationary/rbf/index":                 "/reference/kernels.html",
    "api/kernels/stationary/utils/index":               "/reference/kernels.html",
    "api/kernels/stationary/white/index":               "/reference/kernels.html",
    "api/likelihoods/index":                            "/reference/likelihoods.html",
    "api/linalg/custom_operators/index":                "/reference/linalg.html",
    "api/linalg/utils/index":                           "/reference/linalg.html",
    "api/mean_functions/index":                         "/reference/mean_functions.html",
    "api/models/oilmm/index":                           "/reference/models.html",
    "api/objectives/index":                             "/reference/objectives.html",
    "api/parameters/index":                             "/reference/parameters.html",
    "api/scan/index":                                   "/reference/scan.html",
    "api/state_space/_bessel/index":                    "/reference/state_space.html",
    "api/state_space/_validation/index":                "/reference/state_space.html",
    "api/state_space/fit/index":                        "/reference/state_space.html",
    "api/state_space/gps/index":                        "/reference/state_space.html",
    "api/state_space/inference/index":                  "/reference/state_space.html",
    "api/state_space/kernels/index":                    "/reference/state_space.html",
    "api/state_space/objectives/index":                 "/reference/state_space.html",
    "api/state_space/prediction/index":                 "/reference/state_space.html",
    "api/state_space/sde/index":                        "/reference/state_space.html",
    "api/summary/index":                                "/reference/summary.html",
    "api/typing/index":                                 "/reference/typing.html",
    "api/variational_families/index":                   "/reference/variational_families.html",
    # -- 2. Tutorials: /_examples/<name>/ -> /examples/<name>.html ------------
    "_examples/intro_to_gps/index":                     "/examples/intro_to_gps.html",
    "_examples/intro_to_kernels/index":                 "/examples/intro_to_kernels.html",
    "_examples/regression/index":                       "/examples/regression.html",
    "_examples/classification/index":                   "/examples/classification.html",
    "_examples/poisson/index":                          "/examples/poisson.html",
    "_examples/barycentres/index":                      "/examples/barycentres.html",
    "_examples/deep_kernels/index":                     "/examples/deep_kernels.html",
    "_examples/graph_kernels/index":                    "/examples/graph_kernels.html",
    "_examples/collapsed_vi/index":                     "/examples/collapsed_vi.html",
    "_examples/uncollapsed_vi/index":                   "/examples/uncollapsed_vi.html",
    "_examples/state_space_gps/index":                  "/examples/state_space_gps.html",
    "_examples/oceanmodelling/index":                   "/examples/oceanmodelling.html",
    "_examples/heteroscedastic_inference/index":        "/examples/heteroscedastic_inference.html",
    "_examples/multioutput/index":                      "/examples/multioutput.html",
    "_examples/oilmm/index":                            "/examples/oilmm.html",
    "_examples/oak/index":                              "/examples/oak.html",
    "_examples/numpyro_integration/index":              "/examples/numpyro_integration.html",
    "_examples/spatial_linear_gp/index":                "/examples/spatial_linear_gp.html",
    "_examples/constructing_new_kernels/index":         "/examples/constructing_new_kernels.html",
    "_examples/likelihoods_guide/index":                "/examples/likelihoods_guide.html",
    "_examples/backend/index":                          "/examples/backend.html",
    "_examples/yacht/index":                            "/examples/yacht.html",
}

# Signal to notebooks that they are running inside a docs build.
os.environ["GPJAX_DOCS_BUILD"] = "1"

# Smoke-render flag for CI. Notebooks read this to shrink optimiser steps and
# MCMC draws when set.
os.environ.setdefault("GPJAX_DOCS_CI", "0")

# -- Render-mode stamp --------------------------------------------------------
# Belt-and-braces against smoke renders masquerading as the production docs:
# smoke builds carry a visible banner, and every HTML build writes
# `render-mode.txt` at the site root so what is actually deployed can be
# checked with `curl <site>/render-mode.txt`.
if _smoke_render:
    html_theme_options["announcement"] = (
        "Smoke render: optimisation and MCMC are shrunk for CI speed, so figures "
        "and diagnostics are not publication-fidelity. The production docs are "
        "built with full-length inference."
    )


def setup(app):
    """Register the render-mode marker hook."""

    def _write_render_mode(app, exception):
        if exception is None and app.builder.name == "html":
            mode = "smoke" if _smoke_render else "full"
            with open(os.path.join(app.outdir, "render-mode.txt"), "w") as fh:
                fh.write(mode + "\n")

    app.connect("build-finished", _write_render_mode)
