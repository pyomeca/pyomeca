from typing import Union

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt


def _base_filter(
    array: xr.DataArray,
    freq: Union[int, float],
    order: int,
    cutoff: Union[list, tuple, np.array],
    btype: str,
) -> xr.DataArray:
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype=btype)
    return xr.apply_ufunc(filtfilt, b, a, array)


def low_pass(
    array: xr.DataArray,
    freq: Union[int, float],
    order: int,
    cutoff: Union[int, float, np.array],
) -> xr.DataArray:
    return _base_filter(array, freq, order, cutoff, btype="low")


def high_pass(
    array: xr.DataArray,
    freq: Union[int, float],
    order: int,
    cutoff: Union[int, float, np.array],
) -> xr.DataArray:
    return _base_filter(array, freq, order, cutoff, btype="high")


def band_pass(
    array: xr.DataArray,
    freq: Union[int, float],
    order: int,
    cutoff: Union[list, tuple, np.array],
) -> xr.DataArray:
    return _base_filter(array, freq, order, cutoff, btype="bandpass")


def band_stop(
    array: xr.DataArray,
    freq: Union[int, float],
    order: int,
    cutoff: Union[list, tuple, np.array],
) -> xr.DataArray:
    return _base_filter(array, freq, order, cutoff, btype="bandstop")
