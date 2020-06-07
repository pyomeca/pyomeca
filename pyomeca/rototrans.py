from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from pyomeca import Angles

from .processing import rototrans


class Rototrans:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray]] = None,
        time: Optional[Union[np.array, list, pd.Series]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Rototrans DataArray with `row`, `col` and `time` dimensions used for rototranslation matrix.
         ![rototrans](/images/objects/rototrans.svg)

        Arguments:
            data: Array to be passed to xarray.DataArray
            time: Time vector in seconds associated with the `data` parameter
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Rototrans `xarray.DataArray` with the specified data and coordinates

        !!! example
            To instantiate a `Rototrans` 4 by 4 and 100 frames filled with some random data:

            ```python
            from pyomeca import Rototrans
            import numpy as np

            # create random yet homogeneous data
            n_frames = 100
            data = Rototrans.from_random_data(size=(4, 4, 100)).data

            rt = Rototrans(data)
            ```

            You can an associate time vector:

            ```python
            rate = 100  # Hz
            time = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
            rt = Rototrans(data, time=time)
            ```

        !!! notes
            Calling `Rototrans()` generate an empty array.
        """
        coords = {}
        if data is None:
            data = np.eye(4)
        else:
            # if we provide data, we copy them to avoid making inplace changes
            data = data.copy()

            if data.shape[0] not in (3, 4) or data.shape[0] != data.shape[1]:
                raise IndexError(
                    f"data must have first and second dimensions of length 4, you have: {data.shape}"
                )

        if data.ndim == 2:
            data = data[..., np.newaxis]

        if time is not None:
            coords["time"] = time

        # Make sure last line reads [0, 0, 0, 1]
        zeros = data[3, :3, :]
        ones = data[3, 3, :]
        if not np.alltrue(zeros == 0) or not np.alltrue(ones == 1):
            some_zeros = np.random.choice(zeros.ravel(), 5)
            some_ones = np.random.choice(ones.ravel(), 5)
            raise ValueError(
                "Last line does not read [0, 0, 0, 1].\n"
                f"Here are some values that should be 0: {some_zeros}\n"
                f"And others that should 1: {some_ones}"
            )

        return xr.DataArray(
            data=data,
            dims=("row", "col", "time"),
            coords=coords,
            name="rototrans",
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (3, 1, 100), **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk.

        Arguments:
            distribution: Distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random rototrans `xarray.DataArray` sampled from a given distribution

        !!! example
            To instantiate a `Rototrans` with some random data sampled from a normal distribution:

            ```python
            from pyomeca import Rototrans

            n_frames = 100
            size = 4, 4, n_frames
            rt = Rototrans.from_random_data(size=size)
            ```

            You can choose any distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions):

            ```python
            rt = Rototrans.from_random_data(distribution="uniform", size=size, low=1, high=10)
            ```
        """
        return Rototrans.from_euler_angles(
            Angles.from_random_data(distribution, size=(3, 1, size[-1]), **kwargs),
            "xyz",
        )

    @classmethod
    def from_euler_angles(
        cls,
        angles: Optional[xr.DataArray] = None,
        angle_sequence: Optional[str] = None,
        translations: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """
        Rototrans DataArray from euler angles and specified angle sequence.

        Arguments:
            angles: Euler angles of the rototranslation matrix
            angle_sequence: Euler sequence of angles. Valid values are all permutations of "xyz"
            translations: Translation part of the Rototrans matrix

        Returns:
            Rototrans `xarray.DataArray` from the specified angles and angles sequence

        !!! example
            To get the rototranslation matrix from random euler angles with a given angle sequence type:

            ```python
            from pyomeca import Angles, Rototrans

            size = (3, 1, 100)
            angles = Angles.from_random_data(size=size)
            angles_sequence = "xyz"

            rt = Rototrans.from_euler_angles(angles=angles, angle_sequence=angles_sequence)
            ```

            A translation vector can also be specified:

            ```python
            translation = Angles.from_random_data(size=size)
            rt = Rototrans.from_euler_angles(
                angles=angles, angle_sequence=angles_sequence, translations=translation
            )
            ```
        """
        return rototrans.rototrans_from_euler_angles(
            cls, angles, angle_sequence, translations
        )

    @classmethod
    def from_markers(
        cls,
        origin: xr.DataArray,
        axis_1: xr.DataArray,
        axis_2: xr.DataArray,
        axes_name: str,
        axis_to_recalculate: str,
    ) -> xr.DataArray:
        """
        Rototrans DataArray from a specified set of markers.

        Arguments:
            origin: A marker constructed with `pyomeca.Markers()` corresponding
                to the origin in the global reference frame
            axis_1: Two markers that describe the first axis.
                The first markers being the beginning of the vector and the second being the end.
            axis_2: Two markers that describe the second axis.
                The first markers being the beginning of the vector and the second being the end.
            axes_name: Any combination of `x`, `y` and `z` describing the first and second axes.
            axis_to_recalculate: Which of the two axes to recalculate

        Returns:
            Rototrans `xarray.DataArray` from the specified angles and angles sequence

        !!! example
            To create a system of axes from random markers:

            ```python
            from pyomeca import Markers, Rototrans

            markers = Markers.from_random_data()

            rt = Rototrans.from_markers(
                origin=markers.isel(channel=[0]),  # first marker
                axis_1=markers.isel(channel=[0, 1]),  # vector from the first and second markers
                axis_2=markers.isel(channel=[0, 2]),  # vector from the first and third markers
                axes_name="xy",  # axis_1 is x and axis_2 is y
                axis_to_recalculate="y",  # we want to recalculate y
            )
            ```
        """
        return rototrans.rototrans_from_markers(
            cls, origin, axis_1, axis_2, axes_name, axis_to_recalculate
        )

    @classmethod
    def from_transposed_rototrans(cls, rt: xr.DataArray) -> xr.DataArray:
        """
        Rototrans DataArray from a tranposed Rototrans.

        Arguments:
            rt: Rototrans to transpose

        Returns:
            Transposed Rototrans `xarray.DataArray`

        !!! example
            ```python
            from pyomeca import Rototrans

            rt = Rototrans.from_random_data()

            rt_t = Rototrans.from_transposed_rototrans(rt)
            ```

        !!! notes
            The inverse Rototrans is, by definition, equivalent to the tranposed Rototrans.
        """
        return rototrans.rototrans_from_transposed_rototrans(cls, rt)

    @classmethod
    def from_averaged_rototrans(cls, rt: xr.DataArray) -> xr.DataArray:
        """
        Rototrans DataArray from an averaged Rototrans.

        Arguments:
            rt: Rototrans to average

        Returns:
            Averaged Rototrans `xarray.DataArray`

        !!! example
            To average a `Rototrans` computed from random angles:

            ```python
            import numpy as np
            from pyomeca import Angles, Rototrans

            angles = Angles(np.random.rand(3, 1, 100))
            seq = "xyz"

            rt = Rototrans.from_euler_angles(angles, seq)
            rt_mean = Rototrans.from_averaged_rototrans(rt)
            ```

            Let's make sure the resulting angles are roughly equivalent
            to the averaged angles:

            ```python
            angles_mean = Angles.from_rototrans(rt_mean, seq).isel(time=0)
            angles_mean_ref = Angles.from_rototrans(rt, seq).mean(dim="time")

            error = (angles_mean - angles_mean_ref).meca.abs().sum()
            print(error)
            ```
        """
        return rototrans.rototrans_from_averaged_rototrans(cls, rt)
