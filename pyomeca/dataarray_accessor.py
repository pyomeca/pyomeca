from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .io import write
from .processing import filter, interp, matrix, misc, rototrans


@xr.register_dataarray_accessor("meca")
class DataArrayAccessor(object):
    """Meca DataArray accessor used for processing or file writing."""

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    # io ----------------------------------------
    def to_matlab(self, filename: Union[str, Path]):
        """
        Write a matlab file from a xarray.DataArray.

        Arguments:
            filename: File path

        !!! example
            To write a matlab file from any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.to_matlab(filename="temp.mat")
            ```
        """
        write.write_matlab(self._obj, filename)

    def to_csv(self, filename: Union[str, Path], wide: Optional[bool] = True):
        """
        Write a csv file from a xarray.DataArray.

        Arguments:
            filename: File path
            wide: True if you want a wide dataframe (one column for each channel). False if you want a tidy dataframe.

        !!! example
            To write a csv file from any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.to_csv(filename="temp.csv")
            ```

            By default, `to_csv` will export the data in a "wide" format (1 column by channel).
            You can also export the data in a "tidy" format with `wide=False`:

            ```python
            analogs.meca.to_csv(filename="temp.csv", wide=False)
            ```
        """
        write.write_csv(self._obj, filename, wide)

    def to_wide_dataframe(self) -> pd.DataFrame:
        """
        Return a wide pandas.DataFrame (one column by channel).

        Returns:
            A wide pandas DataFrame (one column by channel).
                Works only for 2 and 3-dimensional arrays.
                If you want a tidy dataframe type: `array.to_series()`, or `array.to_dataframe()`.

        !!! example
            To return a dataframe from any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.to_wide_dataframe()
            ```
        """
        return write.to_wide_dataframe(self._obj)

        # matrix -----------------------------------

    def abs(self) -> xr.DataArray:
        """
        Calculate the absolute value element-wise.

        Returns:
            A `xarray.DataArray` containing the absolute of each element

        !!! example
            To compute the absolute value of any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.abs()
            ```
        """
        return matrix.abs_(self._obj)

    def matmul(self, other: xr.DataArray) -> xr.DataArray:
        """
        Matrix product of two arrays.

        Arguments:
            other: second array to multiply

        Returns:
            A `xarray.DataArray` containing the matrix product of the two arrays

        !!! example
            To compute the matrix product of two `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            first_analogs = Analogs.from_random_data()
            second_analogs = Analogs.from_random_data()

            first_analogs.meca.matmul(second_analogs)
            ```

            You can also use the shorthand `@`:

            ```python
            first_analogs @ second_analogs
            ```
        """
        return matrix.matmul(self._obj, other)

    def square(self, **kwargs) -> xr.DataArray:
        """
        Return the element-wise square of the input.

        Arguments:
            kwargs: For other keyword-only arguments,
                see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html)

        Returns:
            A `xarray.DataArray` containing the matrix squared.

        !!! example
            To compute the element-wise square of any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.square()
            ```
        """
        return matrix.square(self._obj, **kwargs)

    def sqrt(self, **kwargs) -> xr.DataArray:
        """
        Return the non-negative square-root of an array, element-wise.

        Arguments:
            kwargs: For other keyword-only arguments,
                see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)

        Returns:
            A `xarray.DataArray` containing the square root of the matrix.

        !!! example
            To compute the non-negative square-root of any `xarray.DataArray`
            (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.sqrt()
            ```
        """
        return matrix.sqrt(self._obj, **kwargs)

    def norm(self, dim: Union[str, list], ord: int = None) -> xr.DataArray:
        """
        Return the norm of an array.

        Arguments:
            dim: Name(s) of the data dimension(s)
            ord: Order of the norm

        Returns:
            A `xarray.DataArray` containing the norm of the matrix.

        !!! example
            To compute the norm of any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans)
            along a given dimension:

            ```python
            from pyomeca import Markers

            markers = Markers.from_random_data()
            markers.meca.norm(dim="axis")
            ```

        Note:
            If the array contains an `"axis"` dimension with a `"ones"` coordinate
            (e.g., the object was created using `pyomeca.Markers`), this coordinate is ignored.

        """
        return matrix.norm(self._obj, dim, ord)

    def rms(self) -> xr.DataArray:
        """
        Return the root-mean-square of an array.

        Returns:
            A `xarray.DataArray` containing the root-mean-square of the matrix.

        !!! example
            To compute the root-mean-square of any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.rms()
            ```
        """
        return matrix.rms(self._obj)

    def center(
        self, mu: Union[xr.DataArray, np.array, float, int] = None
    ) -> xr.DataArray:
        """
        Center an array (i.e., subtract the mean).

        Arguments:
            mu: the value to be subtracted. If unspecified, take the mean along the time axis.

        Returns:
            a `xarray.DataArray` containing the root-mean-square of the matrix

        !!! example
            To center any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            import numpy as np

            from pyomeca import Analogs

            random_data = np.random.uniform(low=2, high=4, size=(1, 100))
            analogs = Analogs(random_data)
            centered = analogs.meca.center()
            ```

            This will substract the mean of the signal by default.
            The previous random signal was sampled from a uniform distribution from 2 and 4 (mean around 3).
            When centered, the signal is now center around 0 (mean around 0).

            ```python
            import matplotlib.pyplot as plt

            analogs.plot(label="raw")
            centered.plot(label="centered")
            plt.legend()
            plt.show()
            ```

            ![center](/images/api/center.svg)
        """
        return matrix.center(self._obj, mu)

    def normalize(
        self,
        ref: Union[xr.DataArray, np.array, float, int] = None,
        scale: Union[int, float] = 100,
    ) -> xr.DataArray:
        """
        Normalize a signal against `ref` on a scale of `scale`.

        Arguments:
            ref: Reference value. Could have multiple dimensions.
                If not provided, takes the mean along the time axis
            scale: Scale on which to express array (e.g. if 100, the signal is normalized from 0 to 100)
        Returns:
            A `xarray.DataArray` containing the normalized signal

        !!! example
            To normalize any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            import matplotlib.pyplot as plt

            from pyomeca import Analogs

            analogs = Analogs.from_random_data(size=(1, 100)).meca.abs()
            normalized = analogs.meca.normalize()

            normalized.plot()
            plt.show()
            ```

            ![normalize](/images/api/normalize.svg)

            By default, this function normalize against the signal's max.
            To specify any other value, use the `ref` parameter:

            ```python
            normalized = analogs.meca.normalize(ref=1)
            ```
        """
        return matrix.normalize(self._obj, ref, scale)

    # interp ------------------------------------
    def time_normalize(
        self,
        time_vector: Union[xr.DataArray, np.array] = None,
        n_frames: int = 100,
        norm_time: bool = False,
    ) -> xr.DataArray:
        """
        Time normalization used for temporal alignment of data.

        Arguments:
            time_vector: desired time vector (first to last time with n_frames points by default)
            n_frames: if time_vector is not specified, the length of the desired time vector
            norm_time: Normalize the time dimension from 0 to 100 if True

        Returns:
            A time-normalized `xarray.DataArray`

        !!! example
            To time-normalize any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans):

            ```python
            import matplotlib.pyplot as plt

            from pyomeca import Analogs

            analogs = Analogs.from_random_data(size=(1, 847))
            time_normalized = analogs.meca.time_normalize()
            print(time_normalized.time.size)  # 100
            ```

            To normalize the corresponding time dimension from 0 to 100%, specify `norm_time=True`:

            ```python
            time_normalized = analogs.meca.time_normalize(norm_time=True)
            time_normalized.plot()
            plt.show()
            ```

            ![time_normalize](/images/api/time_normalize.svg)

            By default, `time_normalize` use a time vector with 100 frames from 0 to 100.
            However, you can specify the desired number of frames:

            ```python
            time_normalized = analogs.meca.time_normalize(n_frames=500)
            print(time_normalized.time.size)  # 500
            ```

            You can also specify the desired time_vector directly in the `time_vector` parameter:

            ```python
            import numpy as np

            time_normalized = analogs.meca.time_normalize(time_vector=np.linspace(0, 200, 300))
            ```
        """
        return interp.time_normalize(
            self._obj, time_vector, n_frames, norm_time=norm_time
        )

    # filter ------------------------------------
    def low_pass(
        self,
        order: int,
        cutoff: Union[int, float, np.array],
        freq: Optional[Union[int, float]] = None,
    ) -> xr.DataArray:
        """
        Low-pass Butterworth filter.

        Arguments:
            order: Order of the filter
            cutoff: Cut-off frequency
            freq: Sampling frequency.
                Optional if attrs["rate"] is specified.

        Returns:
            A low-pass filtered `xarray.DataArray`

        !!! example
            To low-pass any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans) signal at 5Hz:

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            analogs.meca.low_pass(order=2, cutoff=5, freq=100)
            ```

            Let's see how the low-pass smooth a fake sinusoidal signal:

            ```python
            import matplotlib.pyplot as plt
            import numpy as np

            # generate fake data
            freq = 100  # Hz
            time_vector = np.linspace(start=0, stop=100, num=100)
            w = 2 * np.pi * 1
            y = np.sin(w * time_vector) + 0.1 * np.sin(10 * w * time_vector)

            analogs = Analogs(y.reshape(1, -1))
            low_pass = analogs.meca.low_pass(order=2, cutoff=5, freq=freq)

            analogs.plot(label="raw")
            low_pass.plot(label="low-pass @ 5Hz")
            plt.legend()
            plt.show()
            ```

            ![low_pass](/images/api/low_pass.svg)
        """
        return filter.low_pass(self._obj, order, cutoff, freq)

    def high_pass(
        self,
        order: int,
        cutoff: Union[int, float, np.array],
        freq: Optional[Union[int, float]] = None,
    ) -> xr.DataArray:
        """
        High-pass Butterworth filter.

        Arguments:
            order: Order of the filter
            cutoff: Cut-off frequency
            freq: Sampling frequency.
                Optional if attrs["rate"] is specified.

        Returns:
            A high-pass filtered `xarray.DataArray`

        !!! example
            To high-pass any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans) signal at 100Hz:

            ```python
            import matplotlib.pyplot as plt
            import numpy as np

            from pyomeca import Analogs

            fake_emg = np.random.uniform(low=-1, high=1, size=(1, 1000))
            analogs = Analogs(fake_emg)
            freq = 1000  # Hz
            high_pass = analogs.meca.high_pass(order=2, cutoff=100, freq=freq)

            analogs.plot(label="raw")
            high_pass.plot(label="high-pass @ 100Hz")
            plt.legend()
            plt.show()
            ```

            ![high_pass](/images/api/high_pass.svg)
        """
        return filter.high_pass(self._obj, order, cutoff, freq)

    def band_stop(
        self,
        order: int,
        cutoff: Union[list, tuple, np.array],
        freq: Optional[Union[int, float]] = None,
    ) -> xr.DataArray:
        """
        Band-stop Butterworth filter.

        Arguments:
            order: Order of the filter
            cutoff: Cut-off frequency such as (lower, upper)
            freq: Sampling frequency.
                Optional if attrs["rate"] is specified.

        Returns:
            A band-stop filtered `xarray.DataArray`

        !!! example
            To band-stop any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans) signal at 40-60Hz:

            ```python
            import matplotlib.pyplot as plt
            import numpy as np

            from pyomeca import Analogs

            fake_emg = np.random.uniform(low=-1, high=1, size=(1, 1000))
            analogs = Analogs(fake_emg)
            freq = 1000  # Hz
            band_stop = analogs.meca.band_stop(order=2, cutoff=[40, 60], freq=freq)

            analogs.plot(label="raw")
            band_stop.plot(label="band-stop @ 40-60Hz")
            plt.legend()
            plt.show()
            ```

            ![band_stop](/images/api/band_stop.svg)

        !!! note
            You can also perform a notch filter with this method.
            A notch filter is a band-stop filter with a narrow bandwidth.
            It rejects a narrow frequency band and leaves the rest of the spectrum little changed.
        """
        return filter.band_stop(self._obj, order, cutoff, freq)

    def band_pass(
        self,
        order: int,
        cutoff: Union[list, tuple, np.array],
        freq: Optional[Union[int, float]] = None,
    ) -> xr.DataArray:
        """
        Band-pass Butterworth filter.

        Arguments:
            order: Order of the filter
            cutoff: Cut-off frequency such as (lower, upper)
            freq: Sampling frequency.
                Optional if attrs["rate"] is specified.

        Returns:
            A band-pass filtered `xarray.DataArray`

        !!! example
            To band-pass any `xarray.DataArray` (including Analogs, Angles, Markers or Rototrans) signal at 10-200Hz:

            ```python
            import matplotlib.pyplot as plt
            import numpy as np

            from pyomeca import Analogs

            fake_emg = np.random.uniform(low=-1, high=1, size=(1, 1000))
            analogs = Analogs(fake_emg)
            freq = 1000  # Hz
            band_pass = analogs.meca.band_pass(order=2, cutoff=[10, 200], freq=freq)

            analogs.plot(label="raw")
            band_pass.plot(label="band-pass @ 10-200Hz")
            plt.legend()
            plt.show()
            ```

            ![band_pass](/images/api/band_pass.svg)
        """
        return filter.band_pass(self._obj, order, cutoff, freq)

    # signal processing misc --------------------
    def fft(self, freq: Union[int, float], only_positive: bool = True) -> xr.DataArray:
        """
        Performs a discrete Fourier Transform and return a DataArray with the corresponding amplitudes and frequencies.

        Arguments:
            freq: Sampling frequency (usually stored in `array.attrs["rate"]`)
            only_positive: If `True`, returns only the positive frequencies

        Returns:
            A `xarray.DataArray` with the corresponding amplitudes and frequencies

        !!! example
            Let's compare the resulting fft on a raw and low-passed signal:

            ```python
            import matplotlib.pyplot as plt
            import numpy as np

            from pyomeca import Analogs

            # generate fake data
            freq = 100
            time = np.arange(0, 1, 0.01)
            w = 2 * np.pi
            y = np.sin(w * time) + 0.1 * np.sin(10 * w * time)

            analogs = Analogs(y.reshape(1, -1))
            analogs_low_passed = analogs.meca.low_pass(order=2, cutoff=5, freq=freq)

            # compute fft on raw and low-passed signal
            fft_raw = analogs.meca.fft(freq=freq)
            fft_low_passed = analogs_low_passed.meca.fft(freq=freq)

            fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

            # plot signal vs. time
            analogs.plot.line(x="time", ax=ax[0], color="black", add_legend=False)
            analogs_low_passed.plot.line(x="time", ax=ax[0], color="red", add_legend=False)
            ax[0].set_title("Signal vs. Time")

            # plot amplitudes vs. frequencies
            fft_raw.plot.line(x="freq", ax=ax[1], color="black", label="raw")
            fft_low_passed.plot.line(x="freq", ax=ax[1], color="red", label="low-pass @ 5Hz")
            ax[1].set_title("Amplitudes vs. Freq")

            plt.legend()
            plt.show()
            ```

            ![fft](/images/api/fft.svg)
        """
        return misc.fft(self._obj, freq, only_positive)

    def detect_onset(
        self,
        threshold: Union[float, int],
        n_above: int = 1,
        n_below: int = 0,
        threshold2: int = None,
        n_above2: int = 1,
    ) -> np.array:
        """
        Detects onset based on amplitude threshold.

        Arguments:
            threshold: minimum amplitude to detect
            n_above: minimum number of continuous samples >= `threshold` to detect
            n_below: minimum number of continuous samples below `threshold`
                that will be ignored in the detection of `x` >= `threshold`
            threshold2: minimum amplitude of `n_above2` values in `x` to detect
            n_above2: minimum number of samples >= `threshold2` to detect

        Note:
            You might have to tune the parameters according to the signal-to-noise
              characteristic of the data.

        Returns:
            inds: 1D array_like [indi, indf] containing initial and final indexes of the onset events

        !!! example
            To detect the onsets of any __one-dimensional__ `xarray.DataArray`
            (including Analogs, Angles, Markers or Rototrans):

            ```python
            import matplotlib.pyplot as plt
            import numpy as np
            import scipy.signal as sig

            from pyomeca import Analogs

            # simulate fake ecg data
            rr = 2.5  # rr time in seconds
            freq = 100  # sampling rate
            pqrst = sig.resample(sig.wavelets.daub(10), int(rr * freq))
            ecg = np.concatenate([pqrst, pqrst, pqrst]).reshape(1, -1)

            analogs = Analogs(ecg)
            analogs.plot()

            onsets = analogs.sel(channel=0).meca.detect_onset(
                threshold=analogs.mean(),  # mean of the signal
                n_above=freq / 2,  # we want at least 1/2 second above the threshold
                n_below=freq / 2,  # we accept point below threshold for 1/2 second
            )
            for (start, end) in onsets:
                plt.axvline(x=start, color="g")
                plt.axvline(x=end, color="r")

            plt.show()
            ```

            ![detect_onset](/images/api/detect_onset.svg)

        !!! warning
            `detect_onset` works only for 1-dimensional data.
            For example, you can select a dimension using `analogs.sel(channel='EMG1')` or `analogs.isel(channel=0)`.
        """
        return misc.detect_onset(
            self._obj, threshold, n_above, n_below, threshold2, n_above2
        )

    def detect_outliers(self, threshold: int = 3) -> xr.DataArray:
        """
        Detects data points that are `threshold` times the standard deviation from the mean.

        Arguments:
            threshold: Multiple of standard deviation from which data is considered outlier

        Returns:
            A boolean `xarray.DataArray` containing the outliers

        !!! example
            To get a boolean `xr.DataArray` containing the data that are 3 times the mean +/- standard deviation:

            ```python
            from pyomeca import Analogs

            analogs = Analogs.from_random_data()
            outliers = analogs.meca.detect_outliers(threshold=1)
            ```

            Let's plot the data that are 1 time the mean +/- standard deviation on an analog vector:

            ```python
            import matplotlib.pyplot as plt

            analogs = Analogs.from_random_data(size=(1, 100))

            threshold = 1
            outliers = analogs.meca.detect_outliers(threshold=threshold)

            analogs.plot.line(x="time", color="black", add_legend=False)
            analogs.where(outliers).plot.line(
                x="time", color="red", add_legend=False, marker="o", label="outliers"
            )

            mu = analogs.mean()
            sigma = analogs.std()

            plt.axhline(mu, color="grey")
            plt.axhspan(
                mu - threshold * sigma,
                mu + threshold * sigma,
                color="grey",
                alpha=0.3,
                label=f"mean +/- {threshold} std",
            )

            plt.legend()
            plt.show()
            ```

            ![detect_outliers](/images/api/detect_outliers.svg)

        Note:
            `detect_outliers` is not limited on one-dimensional data and
            can detect outliers for any number of dimensions.
        """
        return misc.detect_outliers(self._obj, threshold)

    @property
    def rotation(self) -> xr.DataArray:
        return rototrans.rotation_getter(self._obj)

    @rotation.setter
    def rotation(self, value) -> None:
        rototrans.rotation_setter(self._obj, value)

    @property
    def translation(self) -> xr.DataArray:
        return rototrans.translation_getter(self._obj)

    @translation.setter
    def translation(self, value) -> None:
        rototrans.translation_setter(self._obj, value)
