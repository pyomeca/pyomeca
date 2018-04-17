""""
Figures in pyomeca
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_vector3d(y, x=None, idx=None, ax=None, fmt='k', lw=1, label=None, alpha=1):
    """
    Plot a pyomeca vector3d (Markers3d, Analogs3d, etc.)

    Parameters
    ----------
    y : np.ndarray
        data to plot on y axis
    x : np.ndarray, optional
        data to plot on x axis
    idx : np.ndarray
        index of the data to plot (hint: `...` to index an entire dimension. For example, idx=[0, [1, 2], ...])
    ax : matplotlib axe, optional
        axis on which the data will be ploted
    fmt : str
        color of the line
    lw : int
        line width of the line
    label : str
        label associated with the data (useful to plot legend)
    """
    if not ax:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    if idx:
        current = y[idx].squeeze()
        if current.ndim > 1 and current.shape[0] < current.shape[1]:
            current = current.T
        if np.any(x):
            ax.plot(x, current, fmt, lw=lw, label=label, alpha=alpha)
        else:
            ax.plot(current, fmt, lw=lw, label=label, alpha=alpha)
    else:
        if y.shape[0] == 1 and y.shape[1] == 1:
            current = y.squeeze()
        else:
            current = y
        if np.any(x):
            ax.plot(x, current, fmt, lw=lw, label=label, alpha=alpha)
        else:
            ax.plot(current, fmt, lw=lw, label=label, alpha=alpha)
    return ax
