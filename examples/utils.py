from pathlib import Path

from matplotlib import transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def clean_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys())
    return ax


def use_mpl_style():
    style_file = Path(__file__).parent / "gpjax.mplstyle"
    plt.style.use(style_file)


def plot_output_panel(
    ax,
    p,
    X_train,
    y_train,
    y_clean,
    cols,
    X_test=None,
    mean=None,
    func_std=None,
    obs_std=None,
):
    col = cols[p % len(cols)]
    ax.scatter(
        X_train, y_train[:, p], alpha=0.4, s=12, color=col, label="Training data"
    )
    ax.plot(
        X_train, y_clean[:, p], "--", color="grey", alpha=0.6, label="Noiseless signal"
    )
    if mean is not None:
        ax.plot(X_test, mean[:, p], color=col, linewidth=2, label="Predictive mean")
        if obs_std is not None:
            ax.fill_between(
                X_test.squeeze(),
                mean[:, p] - 2 * obs_std[:, p],
                mean[:, p] + 2 * obs_std[:, p],
                alpha=0.14,
                color=col,
                label="Two sigma - Observed Output",
            )
        if func_std is not None:
            alpha = 0.26 if obs_std is not None else 0.2
            label = (
                "Two sigma - Latent Function" if obs_std is not None else "Two sigma"
            )
            ax.fill_between(
                X_test.squeeze(),
                mean[:, p] - 2 * func_std[:, p],
                mean[:, p] + 2 * func_std[:, p],
                alpha=alpha,
                color=col,
                label=label,
            )
    ax.set_ylabel(f"Output {p + 1}")
