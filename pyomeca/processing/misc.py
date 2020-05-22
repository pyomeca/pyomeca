from typing import Union

import numpy as np
import xarray as xr
from scipy import fftpack


def has_correct_name(array: xr.DataArray, name: str):
    if array.name != name:
        raise ValueError(f"The provided array is not a {name}; It is a {array.name}.")


def fft(
    array: xr.DataArray, freq: Union[int, float], only_positive=True
) -> xr.DataArray:
    n = array.time.shape[0]
    yfft = fftpack.fft(array.values, n)
    freqs = fftpack.fftfreq(n, 1 / freq)
    if only_positive:
        amp = 2 * np.abs(yfft) / n
        half = int(np.floor(n / 2))
        amp = amp[..., :half]
        freqs = freqs[:half]
    else:
        amp = np.abs(yfft) / n

    coords = {}
    if "axis" in array.dims:
        coords["axis"] = array.axis
    coords["channel"] = array.channel
    coords["freq"] = freqs

    return xr.DataArray(data=amp, dims=coords.keys(), coords=coords)


def detect_onset(
    x,
    threshold: Union[float, int],
    n_above: int = 1,
    n_below: int = 0,
    threshold2: int = None,
    n_above2: int = 1,
) -> np.array:
    if x.ndim != 1:
        raise ValueError(
            f"detect_onset works only for one-dimensional vector. You have {x.ndim} dimensions."
        )
    if isinstance(threshold, xr.DataArray):
        threshold = threshold.item()
    if isinstance(threshold2, xr.DataArray):
        threshold2 = threshold2.item()

    x = np.atleast_1d(x.copy())
    x[np.isnan(x)] = -np.inf
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack(
            (
                inds[np.diff(np.hstack((-np.inf, inds))) > n_below + 1],
                inds[np.diff(np.hstack((inds, np.inf))) > n_below + 1],
            )
        ).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1] - inds[:, 0] >= n_above - 1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if (
                    np.count_nonzero(x[inds[i, 0] : inds[i, 1] + 1] >= threshold2)
                    < n_above2
                ):
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])
    return inds


def detect_outliers(array: xr.DataArray, threshold: int = 3) -> xr.DataArray:
    mu = array.mean(dim="time")
    sigma = array.std(dim="time")
    return xr.DataArray(
        (array < mu - threshold * sigma) | (array > mu + threshold * sigma)
    )
