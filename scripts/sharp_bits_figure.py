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
#     display_name: gpjax_baselines
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches

plt.style.use(
    "https://raw.githubusercontent.com/thomaspinder/GPJax/refs/heads/main/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %%
fig, ax = plt.subplots()
ax.axhline(y=0.25, color=cols[0], linewidth=1.5)

xs = [0.02, 0.06, 0.1, 0.17]
ys = np.ones_like(xs) * 0.25

ax.scatter(xs, ys, color=cols[1], marker="o", s=100, zorder=2)

for idx, x in enumerate(xs):
    ax.annotate(
        text=f"$\\ell_{{t-{idx+1}}}$",
        xy=(x, 0.25),
        xytext=(x + 0.01, 0.275),
        ha="center",
        va="bottom",
    )


style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")

for i in range(len(xs) - 1):
    a = patches.FancyArrowPatch(
        (xs[i + 1], 0.25), (xs[i], 0.25), connectionstyle="arc3,rad=-.5", **kw
    )
    ax.add_patch(a)


ax.scatter(-0.03, 0.25, color=cols[1], marker="x", s=100, linewidth=5, zorder=2)

a = patches.FancyArrowPatch(
    (xs[0], 0.25), (-0.03, 0.25), connectionstyle="arc3,rad=-.5", **kw
)
ax.add_patch(a)

ax.axvline(x=0, color="black", linewidth=0.5, linestyle="-.")
ax.get_yaxis().set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_ylim(0.0, 0.5)
ax.set_xlim(-0.07, 0.25)
plt.savefig("../_static/step_size_figure.png", bbox_inches="tight")

# %%
import numpyro.distributions.transforms as npt

bij = npt.ExpTransform()

x = np.linspace(0.05, 3.0, 6)
y = np.asarray(bij.inv(x))
lval = 0.5
rval = 0.52

fig, ax = plt.subplots()
ax.scatter(x, np.ones_like(x) * lval, s=100, label="Constrained value")
ax.scatter(y, np.ones_like(y) * rval, marker="o", s=100, label="Unconstrained value")

style = "Simple, tail_width=0.25, head_width=2, head_length=8"
for i in range(len(x)):
    if i % 2 != 0:
        a = patches.FancyArrowPatch(
            (x[i], lval), (y[i], rval), connectionstyle="arc3,rad=-.15", **kw
        )
    # a = patches.Arrow(lval, x[i], rval-lval, y[i]-x[i], width=0.05, color='k')
    else:
        a = patches.FancyArrowPatch(
            (x[i], lval), (y[i], rval), connectionstyle="arc3,rad=.005", **kw
        )
    ax.add_patch(a)

ax.get_yaxis().set_visible(False)
ax.spines["left"].set_visible(False)
ax.legend(loc="best")
# ax.set_ylim(0.1, 0.32)
plt.savefig("../_static/bijector_figure.svg", bbox_inches="tight")

# %%
np.log(0.05)

# %%
print(x)
