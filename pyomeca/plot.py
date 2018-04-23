""""
Figures in pyomeca
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_vector3d(y, x=None, ax=None, fmt='k', lw=1, label=None, alpha=1):
    """
    Plot a pyomeca vector3d (Markers3d, Analogs3d, etc.)

    Parameters
    ----------
    y : np.ndarray
        data to plot on y axis
    x : np.ndarray, optional
        data to plot on x axis
    ax : matplotlib axe, optional
        axis on which the data will be ploted
    fmt : str
        color of the line
    lw : int
        line width of the line
    label : str
        label associated with the data (useful to plot legend)
    alpha : int, float
        alpha
    """
    if not ax:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    if y.shape[0] == 1 and y.shape[1] == 1:
        current = y.squeeze()
    else:
        current = y
    if np.any(x):
        ax.plot(x, current, fmt, lw=lw, label=label, alpha=alpha)
    else:
        ax.plot(current, fmt, lw=lw, label=label, alpha=alpha)
    return ax
