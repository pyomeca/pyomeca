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
