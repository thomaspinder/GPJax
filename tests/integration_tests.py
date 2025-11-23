# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: docs
#     language: python
#     name: python3
# ---

# %%
from dataclasses import (
    dataclass,
    field,
)

# %%
from beartype.typing import (
    Any,
    Callable,
    Dict,
)
import jax.numpy as jnp  # noqa: F401
import jupytext

import gpjax

# %%
get_last = lambda x: x[-1]


# %%
@dataclass
class Result:
    path: str
    comparisons: field(default_factory=dict)  # type: ignore
    precision: int = 1
    compare_history: bool = True

    def __post_init__(self):
        self.name: str = self.path.split("/")[-1].split(".")[0].replace("_", "-")

    def _compare(
        self,
        observed_variables: Dict[str, Any],
        variable_name: str,
        true_value: float,
        operation: Callable[[Any], Any],
    ):
        if variable_name == "history" and not self.compare_history:
            return
        try:
            value = operation(observed_variables[variable_name])
            assert abs(true_value - value) < self.precision
        except AssertionError as e:
            print(e)

    def test(self):
        notebook = jupytext.read(self.path)
        contents = ""
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                if c["source"].startswith("%"):
                    pass
                else:
                    contents += c["source"]
            contents += "\n"

        contents = contents.replace('plt.style.use("./gpjax.mplstyle")', "").replace(
            "plt.show()", ""
        )
        lines = contents.split("\n")
        contents = "\n".join([line for line in lines if not line.startswith("%")])

        loc = {}
        # weird bug in interactive interpreter: lambda functions
        # don't have access to the global scope of the executed file
        # so we need to pass gpjax in the globals explicitly
        # since it's used in a lambda function inside the examples
        _globals = globals()
        _globals["gpx"] = gpjax
        exec(contents, _globals, loc)
        for k, v in self.comparisons.items():
            truth, op = v
            self._compare(
                observed_variables=loc, variable_name=k, true_value=truth, operation=op
            )


# %%
regression = Result(
    path="examples/regression.py",
    comparisons={
        "history": (55.07405622, get_last),
        "predictive_mean": (36.24383416, jnp.sum),
        "predictive_std": (197.04727051, jnp.sum),
    },
)
regression.test()

# %%
sparse = Result(
    path="examples/collapsed_vi.py",
    comparisons={
        "history": (1924.7634809, get_last),
        "predictive_mean": (-8.39869652, jnp.sum),
        "predictive_std": (255.74838027, jnp.sum),
    },
)
sparse.test()

# %%
stochastic = Result(
    path="examples/uncollapsed_vi.py",
    comparisons={
        "history": (-2678.41302494, get_last),
        "meanf": (-54.14787028, jnp.sum),
        "sigma": (121.4298333, jnp.sum),
    },
)
stochastic.test()

# %%
heteroscedastic = Result(
    path="examples/heteroscedastic_inference.py",
    comparisons={
        "history": (251.918, get_last),
    },
)
heteroscedastic.test()
