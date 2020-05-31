from typing import Union

import numpy as np
import xarray as xr


def abs_(array: xr.DataArray) -> xr.DataArray:
    return np.abs(array)


def matmul(array: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    return array @ other


def square(array: xr.DataArray, **kwargs) -> xr.DataArray:
    return np.square(array, **kwargs)


def norm(array: xr.DataArray, dim: Union[str, list], ord: int = None) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.norm,
        array.drop_sel(axis="ones")
        if hasattr(array, "axis") and "ones" in array.axis
        else array,
        input_core_dims=[[dim]] if isinstance(dim, str) else dim,
        kwargs={"ord": ord, "axis": -1},
    )


def sqrt(array: xr.DataArray, **kwargs) -> xr.DataArray:
    return np.sqrt(array, **kwargs)


def rms(array: xr.DataArray) -> xr.DataArray:
    return array.meca.square().mean().meca.sqrt()


def center(
    array: xr.DataArray, mu: Union[xr.DataArray, np.array, float, int] = None
) -> xr.DataArray:
    if mu is None:
        return array - array.mean(dim="time")
    return array - mu


def normalize(
    array: xr.DataArray,
    ref: Union[xr.DataArray, np.array, float, int] = None,
    scale: Union[int, float] = 100,
) -> xr.DataArray:
    if ref is None:
        ref = array.max(dim="time")
    return array / (ref / scale)
