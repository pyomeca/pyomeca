import numpy as np
import xarray as xr


def markers_from_rototrans(markers: xr.DataArray, rt: xr.DataArray) -> xr.DataArray:
    rotated_markers = markers.copy()

    if rt.ndim == 3 and markers.ndim == 3:
        rotated_markers.data = np.einsum("ijk,jlk->ilk", rt, markers)
    elif rt.ndim == 2 and markers.ndim == 2:
        rotated_markers.data = np.dot(rt, markers)
    elif rt.ndim == 2 and markers.ndim == 3:
        rotated_markers.data = np.einsum("ij,jkl->ikl", rt, markers)
    else:
        raise ValueError("`rt` and `markers` dimensions do not match.")

    return rotated_markers
