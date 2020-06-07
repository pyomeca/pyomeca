from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .processing import angles


class Angles:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray]] = None,
        time: Optional[Union[np.array, list, pd.Series]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Angles DataArray with `axis`, `channel` and `time` dimensions used for joint angles.
         ![angles](/images/objects/angles.svg)

        Arguments:
            data: Array to be passed to xarray.DataArray
            time: Time vector in seconds associated with the `data` parameter
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Angles `xarray.DataArray` with the specified data and coordinates

        !!! example
            To instantiate an `Angles` 3 by 3 and 100 frames filled with some random data:

            ```python
            import numpy as np
            from pyomeca import Angles

            n_axis = 3
            n_channel = 4
            n_frames = 100
            data = np.random.random(size=(n_axis, n_channel, n_frames))
            angles = Angles(data)
            ```

            You can an associate time vector:

            ```python
            rate = 100  # Hz
            time = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
            angles = Angles(data, time=time)
            ```

        !!! note
            Calling `Angles()` generate an empty array.
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0, 0))
        if time is not None:
            coords["time"] = time
        return xr.DataArray(
            data=data,
            dims=("axis", "channel", "time"),
            coords=coords,
            name="angles",
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (3, 10, 100), **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk.

        Arguments:
            distribution: Distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random angles `xarray.DataArray` sampled from a given distribution

        !!! example
            To instantiate an `Angles` with some random data sampled from a normal distribution:

            ```python
            from pyomeca import Angles

            n_frames = 100
            size = 3, 10, n_frames
            angles = Angles.from_random_data(size=size)
            ```

            You can choose any distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions):

            ```python
            angles = Angles.from_random_data(distribution="uniform", size=size, low=1, high=10)
            ```
        """
        return Angles(getattr(np.random, distribution)(size=size, **kwargs).cumsum(-1))

    @classmethod
    def from_rototrans(cls, rt: xr.DataArray, angle_sequence: str) -> xr.DataArray:
        """
        Angles DataArray from a rototranslation matrix and specified angle sequence.

        Arguments:
            rt: Rototranslation matrix created with pyomeca.Rototrans()
            angle_sequence: Euler sequence of angles. Valid values are all permutations of "xyz"

        Returns:
            Angles `xarray.DataArray` from the specified rototrans and angles sequence

        !!! example
            To get the euler angles from a random rototranslation matrix with a given angle sequence type:

            ```python
            from pyomeca import Angles, Rototrans

            size = (4, 4, 100)
            rt = Rototrans.from_random_data(size=size)
            angles_sequence = "xyz"

            angles = Angles.from_rototrans(rt, angles_sequence)
            ```
        """
        return angles.angles_from_rototrans(cls, rt, angle_sequence)
