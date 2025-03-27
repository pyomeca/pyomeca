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

    def interpolate(vector: np.ndarray) -> np.ndarray:
        interpolated_vector = np.zeros_like(vector)
        bad_indexes = np.isnan(vector)
        good_indexes = np.logical_not(bad_indexes)
        good_vector = vector[good_indexes]
        interpolated_vector[good_indexes] = vector[good_indexes]
        interpolated = np.interp(np.nonzero(np.array(bad_indexes))[0], np.nonzero(np.array(good_indexes))[0], np.array(good_vector))
        interpolated_vector[bad_indexes] = interpolated
        return interpolated_vector

    interpolated_array = np.zeros_like(array)
    if len(array.shape) == 2:
        for i_shape0 in range(array.shape[0]):
            interpolated_array[i_shape0, :] = interpolate(array[i_shape0, :])
    elif len(array.shape) == 3:
        for i_shape0 in range(array.shape[0]):
            for i_shape1 in range(array.shape[1]):
                interpolated_array[i_shape0, i_shape1, :] = interpolate(array[i_shape0, i_shape1, :])
    else:
        raise NotImplementedError("Only 2D and 3D arrays are supported yet for interpolate_missing_data.")

    new_array = array.copy()
    new_array.values = interpolated_array[:, :]
    return new_array