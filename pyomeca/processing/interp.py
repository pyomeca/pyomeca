from typing import Union

import numpy as np
import xarray as xr


def time_normalize(
    array: xr.DataArray,
    time_vector: Union[xr.DataArray, np.array] = None,
    n_frames: int = 100,
    norm_time: bool = False,
) -> xr.DataArray:
    if time_vector is None:
        if norm_time:
            first_last_time = (0, 99)
            array["time"] = np.linspace(
                first_last_time[0],
                first_last_time[1],
                array["time"].shape[0],
            )
        else:
            first_last_time = (array.time[0], array.time[-1])
        time_vector = np.linspace(first_last_time[0], first_last_time[1], n_frames)
    return array.interp(time=time_vector)


def interpolate_missing_data(array: xr.DataArray) -> xr.DataArray:

    interpolated_array = np.zeros_like(array)
    for i in range(array.shape[0]):
        bad_indexes = np.isnan(array[i, :])
        good_indexes = np.logical_not(bad_indexes)
        good_array = array[i, good_indexes]
        interpolated_array[i, good_indexes] = array[i, good_indexes]
        interpolated = np.interp(np.nonzero(np.array(bad_indexes))[0], np.nonzero(np.array(good_indexes))[0], np.array(good_array))
        interpolated_array[i, bad_indexes] = interpolated

    new_array = array.copy()
    new_array.values = interpolated_array[:, :]
    return new_array