from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt


def _base_filter(
    array: xr.DataArray,
    order: int,
    cutoff: Union[list, tuple, np.array],
    freq: Optional[Union[int, float]],
    btype: str,
) -> xr.DataArray:
    if freq is None:
        if array.attrs.get("rate"):
            freq = array.rate
        else:
            raise ValueError(
                "the `freq` param is optional only if `rate` is available in the attrs dictionnary (array.attrs`)"
            )
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype=btype)
    return xr.apply_ufunc(filtfilt, b, a, array)


def low_pass(
    array: xr.DataArray,
    order: int,
    cutoff: Union[int, float, np.array],
    freq: Optional[Union[int, float]] = None,
) -> xr.DataArray:
    return _base_filter(array, order, cutoff, freq, btype="low")


def high_pass(
    array: xr.DataArray,
    order: int,
    cutoff: Union[int, float, np.array],
    freq: Optional[Union[int, float]] = None,
) -> xr.DataArray:
    return _base_filter(array, order, cutoff, freq, btype="high")


def band_pass(
    array: xr.DataArray,
    order: int,
    cutoff: Union[list, tuple, np.array],
    freq: Optional[Union[int, float]] = None,
) -> xr.DataArray:
    return _base_filter(array, order, cutoff, freq, btype="bandpass")


def band_stop(
    array: xr.DataArray,
    freq: Optional[Union[int, float]],
    order: int,
    cutoff: Union[list, tuple, np.array],
) -> xr.DataArray:
    return _base_filter(array, freq, order, cutoff, btype="bandstop")
