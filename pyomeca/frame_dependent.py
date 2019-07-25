from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.io import savemat
from scipy.signal import filtfilt, medfilt, butter


class FrameDependentNpArray(np.ndarray):
    def __new__(cls, array=np.ndarray((0, 0, 0)), *args, **kwargs):
        """
        Convenient wrapper around np.ndarray for time related data
        Parameters
        ----------
        array : np.ndarray
            A 3-dimensions matrix where 3rd dimension is the frame
        -------

        """
        if not isinstance(array, np.ndarray):
            raise TypeError('FrameDependentNpArray must be a numpy array')

        # Sanity check on size
        if len(array.shape) == 1:
            array = array[:, np.newaxis, np.newaxis]
        if len(array.shape) == 2:
            array = array[:, :, np.newaxis]

        # metadata
        obj = np.asarray(array).view(cls, *args, **kwargs)
        obj.__array_finalize__(array)
        return obj

    def __parse_item__(self, item):
        if isinstance(item, int):
            pass
        elif isinstance(item[0], str):
            if len(self.shape) != 3:
                raise RuntimeError("Name slicing is only valid on 3D FrameDependentNpArray")
            item = (slice(None, None, None), self.get_index(item), slice(None, None, None))
        elif len(item) == 3:
            if isinstance(item[1], int):
                if isinstance(item[0], int) and isinstance(item[2], int):
                    pass
                else:
                    item = (item[0], [item[1]], item[2])
            if isinstance(item[1], tuple):
                item = (item[0], list(item[1]), item[2])
            if isinstance(item[1], list):  # If multiple value
                idx = self.get_index(item[1])
                if idx:
                    # Replace the text by number so it can be sliced
                    idx_str = [i for i, it in enumerate(item[1]) if isinstance(it, str)]
                    for i1, i2 in enumerate(idx_str):
                        item[1][i2] = idx[i1]
            elif isinstance(item[1], str):  # If single value
                item = (item[0], self.get_index(item[1]), item[2])
        return item

    def __getitem__(self, item):
        return super(FrameDependentNpArray, self).__getitem__(self.__parse_item__(item))

    def __setitem__(self, key, value):
        return super(FrameDependentNpArray, self).__setitem__(self.__parse_item__(key), value)

    def __array_finalize__(self, obj):
        # Allow slicing
        if obj is None or not isinstance(obj, FrameDependentNpArray):
            self._current_frame = 0
            self.get_first_frame = []
            self.get_last_frame = []
            self.get_time_frames = None
            self.get_rate = []
            self.get_labels = []
            self.get_unit = []
            self.get_nan_idx = None
            self.misc = {}
        else:
            self._current_frame = getattr(obj, '_current_frame')
            self.get_first_frame = getattr(obj, 'get_first_frame')
            self.get_last_frame = getattr(obj, 'get_last_frame')
            self.get_time_frames = getattr(obj, 'get_time_frames')
            self.get_rate = getattr(obj, 'get_rate')
            self.get_labels = getattr(obj, 'get_labels')
            self.get_unit = getattr(obj, 'get_unit')
            self.get_nan_idx = getattr(obj, 'get_nan_idx')
            self.misc = getattr(obj, 'misc')

    def dynamic_child_cast(self, x):
        """
        Dynamically cast the np.array into type of self (which is probably inherited from FrameDependentNpArray)
        Parameters
        ----------
        x : np.array

        Returns
        -------
        x in the same type as self
        """
        casted_x = type(self)(x)
        casted_x.__array_finalize__(self)
        return casted_x

    def __iter__(self):
        self._current_frame = 0  # Reset the counter
        return self

    def __next__(self):
        if self._current_frame < self.shape[2]:
            self._current_frame += 1
            return self.get_frame(self._current_frame-1)  # -1 since it is incremented before hand
        else:
            raise StopIteration

    # --- Utils methods

    @classmethod
    def _get_class_name(cls):
        return cls.__name__

    @staticmethod
    def check_parent_dir(file_name):
        file_name = Path(file_name)
        if not file_name.parents[0].is_dir():
            file_name.parents[0].mkdir()
        return file_name

    def update_misc(self, d):
        """
        Append the misc field with a given dictionary.
        An Optional reference to the internal state is also return in order to chain the operation if needed.

        Parameters
        ----------
        d : dict
            Dictionary to be added to the misc field
        """
        self.misc.update(d)
        return self.dynamic_child_cast(self)

    # --- Get metadata methods

    def get_num_frames(self):
        """

        Returns
        -------
        The number of frames
        """
        s = self.shape
        if len(s) == 2:
            return 1
        else:
            return s[2]

    def get_frame(self, f):
        """
        Return the fth frame of the array
        Parameters
        ----------
        f : int
            index of frame

        Returns
        -------
        frame
        """
        return self[..., f]

    def get_index(self, names):
        """
        Return the index associated to label names
        Parameters
        ----------
        names : list(str)
            names of the label to find the indexes of

        Returns
        -------
        indexes
        """
        if isinstance(names, list) or isinstance(names, tuple):
            # Remove the integer
            names = [n for n in names if isinstance(n, str)]
        else:
            names = [names]

        if not names:  # If all integer, return null
            return []
        else:
            return [self.get_labels.index(name) for name in names]

    # --- Fileio methods (from_*)

    @classmethod
    def from_csv(cls, filename, first_row=0, time_column=None, first_column=None, last_column_to_remove=None, idx=None,
                 header=None, names=None, delimiter=',', prefix=None, skiprows=None, na_values=None):
        """
        Read csv data and convert to Vectors3d format
        Parameters
        ----------
        filename : Union[str, Path]
            Path of file
        first_row : int
            Index of first rows of data (0th indexed)
        first_column : int
            Index of first column of data (0th indexed)
        last_column_to_remove : int
            If for some reason the csv reads extra columns, how many should be ignored
        time_column : int
            Index of the time column, if None time column is the index
        idx : list(int)
            Order of columns given by index
        header : int
            row of the header (0th indexed)
        names : list(str)
            Order of columns given by names, if both names and idx are provided, an error occurs
        delimiter : str
            Delimiter of the CSV file
        prefix : str
            Prefix to remove in the header

        Returns
        -------
        Data set in Vectors3d format
        """

        if names and idx:
            raise ValueError("names and idx can't be set simultaneously, please select only one")

        if not skiprows:
            if not header:
                skiprows = np.arange(1, first_row)
            else:
                skiprows = np.arange(header + 1, first_row)

        data = pd.read_csv(str(filename), sep=delimiter, header=header, skiprows=skiprows, na_values=na_values)
        if time_column is None:
            time_frames = np.arange(0, data.shape[0])
        else:
            time_frames = np.array(data.iloc[:, time_column])

        if first_column:
            data.drop(data.columns[:first_column], axis=1, inplace=True)

        if last_column_to_remove:
            data.drop(data.columns[-last_column_to_remove], axis=1, inplace=True)

        column_names = data.columns.tolist()
        if header and cls._get_class_name()[:9] == 'Markers3d':
            column_names = [icol.split(prefix)[-1] for icol in column_names if
                            not (len(icol) >= 7 and icol[:7] == 'Unnamed')]
        metadata = {'get_first_frame': [], 'get_last_frame': [], 'get_rate': [], 'get_labels': [], 'get_unit': [],
                    'get_time_frames': time_frames}
        if names:
            metadata.update({'get_labels': names})
        else:
            names = column_names

        return cls._to_vectors(data=data.values, idx=idx, all_names=column_names, target_names=names, metadata=metadata)

    @staticmethod
    def _parse_c3d(c3d, prefix):
        """
        Abstract function on how to read c3d header and parameter for markers or analogs.
        Must be implemented for each subclasses of frame_dependent.

        Parameters
        ----------
        c3d : ezc3d class
            Pointer on the read c3d
        prefix : str, optional
            Participant's prefix
        Returns
        -------
        data : np.ndarray
            Actual data
        channel_names : List(string)
            Name of the channels
        metadata
            Structure of properties in the c3d files
        """
        raise NotImplementedError('_parse_c3d_info is an abstract function')

    @classmethod
    def from_c3d(cls, filename, idx=None, names=None, prefix=None):
        """
        Read c3d data and convert to Vectors3d format
        Parameters
        ----------
        filename : Union[str, Path]
            Path of file
        idx : list(int)
            Order of columns given by index
        names : list(str)
            Order of columns given by names, if both names and idx are provided, an error occurs
        prefix : str
            Prefix to remove in the header

        Returns
        -------
        Data set in Vectors3d format or Data set in Vectors3d format and metadata dict if get_metadata is True
        """
        if names and idx:
            raise ValueError("names and idx can't be set simultaneously, please select only one")
        reader = ezc3d.c3d(str(filename)).c3d_swig
        data, channel_names, metadata = cls._parse_c3d(reader, prefix)

        if names:
            metadata.update({'get_labels': names})
        else:
            metadata.update({'get_labels': []})
            names = channel_names

        # Add time frames
        metadata['get_time_frames'] = np.arange(metadata['get_first_frame'] / metadata['get_rate'],
                                                (metadata['get_last_frame'] + 1) / metadata['get_rate'],
                                                1 / metadata['get_rate'])

        return cls._to_vectors(data=data,
                               idx=idx,
                               all_names=channel_names,
                               target_names=names,
                               metadata=metadata)

    @classmethod
    def _to_vectors(cls, data, idx, all_names, target_names, metadata=None):
        if not idx:
            # find names in column_names
            idx = []
            for i, m in enumerate(target_names):
                idx.append([i for i, s in enumerate(all_names) if m in s][0])

        data = cls.__new__(cls, data)  # Dynamically cast the data to fit the child
        data = data.get_specific_data(idx)

        data.get_first_frame = metadata['get_first_frame']
        data.get_last_frame = metadata['get_last_frame']
        data.get_rate = metadata['get_rate']
        data.get_unit = metadata['get_unit']
        data.get_time_frames = metadata['get_time_frames']
        if np.array(idx).ndim == 1 and not metadata['get_labels']:
            data.get_labels = [name for i, name in enumerate(all_names) if i in idx]
        elif metadata['get_labels']:
            data.get_labels = metadata['get_labels']
        return data

    def get_specific_data(self, idx):
        """
        # TODO: description
        Parameters
        ----------
        idx : list(int)
            idx of marker to keep (order is kept in the returned data).
            If idx has more than one row, output is the mean of the markers over the columns.
        Returns
        -------
        numpy.array
            extracted data
        """
        idx = np.array(idx, ndmin=2)
        try:
            data = self[:, idx[0, :], :]
            for i in range(1, idx.shape[0]):
                data += self[:, np.array(idx)[i, :], :]
            data /= idx.shape[0]
        except IndexError:
            raise IndexError('get_specific_data works only on 3xNxF matrices and idx must be a ixj array')
        return data

    # --- Fileio methods (to_*)

    def to_dataframe(self, add_metadata=[]):
        """
        Convert a Vectors3d class to a pandas dataframe

        Parameters
        ----------
        add_metadata : list
            add each metadata specified in this list to the dataframe

        Returns
        -------
        pd.DataFrame
        """
        cols = {}
        for imeta in add_metadata:
            ivalue = getattr(self, imeta)
            if isinstance(ivalue, dict):
                cols.update({key: value for key, value in ivalue.items()})
            else:
                cols.update({imeta: ivalue})
        return pd.DataFrame(self.to_2d(), columns=self.get_2d_labels()).assign(**cols)

    def to_csv(self, file_name, header=False):
        """
        Write a csv file from a FrameDependentNpArray

        Parameters
        ----------
        file_name : str, Path
            path of the file to write
        header : bool, optional
            Write header with labels (default False)
        """
        file_name = self.check_parent_dir(file_name)

        # Get the 2d style labels
        if header:
            header = self.get_2d_labels()

        # Write into the csv file
        pd.DataFrame(self.to_2d()).to_csv(file_name, index=False, header=header)

    def to_mat(self, file_name, metadata=False):
        """
        Write a Matlab's mat file from a FrameDependentNpArray

        Parameters
        ----------
        file_name : str, Path
            path of the file to write
        metadata : bool, optional
            Write data with metadata (default False)

        Returns
        -------

        """
        file_name = self.check_parent_dir(file_name)
        mat_dict = {}
        if metadata:
            mat_dict.update({
                'get_first_frame': self.get_first_frame,
                'get_last_frame': self.get_last_frame,
                'get_time_frames': self.get_time_frames,
                'get_rate': self.get_rate,
                'get_labels': self.get_labels,
                'get_unit': self.get_unit,
                'get_nan_idx': self.get_nan_idx
            })
        mat_dict.update({'data': self})
        savemat(file_name, mat_dict)

    # --- Plot method

    def plot(self, x=None, ax=None, fmt='', lw=1, label=None, alpha=1):
        """
        Plot a pyomeca vector3d (Markers3d, Analogs3d, etc.)

        Parameters
        ----------
        x : np.ndarray, optional
            data to plot on x axis
        ax : matplotlib axe, optional
            axis on which the data will be ploted
        fmt : str
            color of the line
        lw : int
            line width of the line
        label : str
            label associated with the data (useful to plot legend)
        alpha : int, float
            alpha
        """
        if not ax:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for i in range(self.shape[0]):
            data_to_plot = np.squeeze(self[i, :, :]).transpose()
            if np.any(x):
                ax.plot(x, data_to_plot, fmt, lw=lw, label=label, alpha=alpha)
            elif self.get_time_frames is None or self.get_time_frames.shape[0] != self.shape[2]:
                ax.plot(data_to_plot, fmt, lw=lw, label=label, alpha=alpha)
            else:
                ax.plot(self.get_time_frames, data_to_plot, fmt, lw=lw, label=label, alpha=alpha)
        return ax

    # --- Signal processing methods

    def matmul(self, other):
        """
        Matrix product of two arrays.

        Parameters
        ----------
        other : np.ndarray
            Second matrix to multiply

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(np.matmul(self, other))

    def abs(self):
        """
        Get absolute values

        Returns
        -------
        FrameDependentNpArray
        """
        return np.abs(self)

    def square(self):
        """
        Get square of values

        Returns
        -------
        FrameDependentNpArray
        """
        return np.square(self)

    def sqrt(self):
        """
        Get square root of values

        Returns
        -------
        FrameDependentNpArray
        """
        return np.sqrt(self)

    def mean(self, *args, axis=2, **kwargs):
        """
        Get mean values (default over time)

        Returns
        -------
        FrameDependentNpArray
        """
        return super().mean(*args, axis=axis, keepdims=True, **kwargs)

    def nanmean(self, *args, axis=2, **kwargs):
        """
        Get mean values ignoring NaNs (default over time)

        Returns
        -------
        FrameDependentNpArray
        """
        return np.nanmean(self, *args, axis=axis, keepdims=True, **kwargs)

    def rms(self, axis=2):
        """
        Get root-mean-square values

        Returns
        -------
        FrameDependentNpArray
        """
        return self.square().nanmean().sqrt()

    def center(self, mu=None, axis=-1):
        """
        Center a signal (i.e., subtract the mean)

        Parameters
        ----------
        mu : np.ndarray, float, int
            mean of the signal to subtract, optional
        axis : int, optional
            axis along which the means are computed. The default is to compute
            the mean on the last axis.
        Returns
        -------
        FrameDependentNpArray
        """
        if not np.any(mu):
            mu = np.nanmean(self, axis=axis)
        if self.ndim > mu.ndim:
            # add one dimension if the input is a 3d matrix
            mu = np.expand_dims(mu, axis=-1)
        return self - mu

    def max(self, *args, axis=2, **kwargs):
        """
        Get maximal value (default over time)

        Returns
        -------
        float
        """
        return super().max(*args, axis=axis, **kwargs)

    def normalization(self, ref=None, scale=100):
        """
        Normalize a signal against `ref` (x's max if empty) on a scale of `scale`

        Parameters
        ----------
        ref : np.ndarray
            reference value
        scale
            Scale on which to express x (100 by default)

        Returns
        -------
        FrameDependentNpArray
        """
        if not np.any(ref):
            ref = np.nanmax(self, axis=-1)
        # add one dimension
        ref = np.expand_dims(ref, axis=-1)
        return self / (ref / scale)

    def time_normalization(self, time_vector=np.linspace(0, 100, 101), axis=-1):
        """
        Time normalization used for temporal alignment of data

        Parameters
        ----------
        time_vector : np.ndarray
            desired time vector (0 to 100 by step of 1 by default)
        axis : int
            specifies the axis along which to interpolate. Interpolation defaults to the last axis (over frames)

        Returns
        -------
        FrameDependentNpArray
        """
        original_time_vector = np.linspace(time_vector[0], time_vector[-1], self.shape[axis])
        f = interp1d(original_time_vector, self, axis=axis)
        return self.dynamic_child_cast(f(time_vector))

    def fill_values(self, axis=-1):
        """
        Fill values. Warning: this function can be used only for very small gaps in your data.

        Parameters
        ----------
        axis : int
            specifies the axis along which to interpolate. Interpolation defaults to the last axis (over frames)

        Returns
        -------
        FrameDependentNpArray
        """

        def fct(m):
            """Simple function to interpolate along an axis"""
            nans, y = np.isnan(m), lambda z: z.nonzero()[0]
            m[nans] = np.interp(y(nans), y(~nans), m[~nans])
            return m

        if np.any(self.get_nan_idx):
            # do not take nan dimensions
            index = np.ones(self.shape[1], dtype=bool)
            index[self.get_nan_idx] = False
            x = self[:, index, :].copy()

            out = np.apply_along_axis(fct, axis=axis, arr=x)
            # reinsert nan dimensions
            for i in self.get_nan_idx:
                out = np.insert(out, i, np.nan, axis=1)
        else:
            x = self.copy()
            out = np.apply_along_axis(fct, axis=axis, arr=x)
        return self.dynamic_child_cast(out)

    def check_for_nans(self, threshold_channel=10, threshold_consecutive=5):
        """
        1. Check if there is less than `threshold_channel`% of nans on each channel
        2. Check if there is not more than `threshold_consecutive`% of the rate of consecutive nans

        Parameters
        ----------
        threshold_channel : int
            Threshold of tolerated nans on each channel
        threshold_consecutive : int
            Threshold of tolerated consecutive nans on each channel
        """
        # check if there is nans
        if self.get_nan_idx is not None:
            nans = np.isnan(
                self[:, np.setdiff1d(np.arange(self.shape[1]), self.get_nan_idx), :]
            )
        else:
            nans = np.isnan(self)
        if nans.any():
            # check if there is less than `threshold_channel`% of nans on each channel
            percentage = (nans.sum(axis=-1) / self.shape[-1] * 100).ravel()
            above = np.argwhere(percentage > threshold_channel)
            if above.any():
                for iabove in above:
                    if iabove not in self.get_nan_idx:
                        raise ValueError(
                            f"There is more than {threshold_channel}% ({percentage[iabove]}) "
                            f"NaNs on the channel ({iabove})"
                        )

            # check if there is not more than `threshold_consecutive`% of the rate of consecutive nans
            def max_consecutive_nans(a):
                mask = np.concatenate(([False], np.isnan(a), [False]))
                if ~mask.any():
                    return 0
                else:
                    idx = np.nonzero(mask[1:] != mask[:-1])[0]
                    return (idx[1::2] - idx[::2]).max()

            consecutive_nans = np.apply_along_axis(
                max_consecutive_nans, axis=-1, arr=self
            ).ravel()
            above = np.argwhere(consecutive_nans > self.get_rate / threshold_consecutive)
            percentage = (consecutive_nans / self.shape[-1] * 100).ravel()
            if above.any():
                for iabove in above:
                    if iabove not in self.get_nan_idx:
                        raise ValueError(
                            f"There is more than {threshold_consecutive}% ({percentage[iabove]}) "
                            f"consecutive NaNs on the channel ({iabove})"
                        )
            return True
        else:
            return False

    def moving_rms(self, window_size):
        """
        Moving root mean square

        Parameters
        ----------
        window_size : Union(int, float)
            Window size

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(np.sqrt(filtfilt(np.ones(window_size) / window_size, 1, self * self)))

    def moving_average(self, window_size):
        """
        Moving average

        Parameters
        ----------
        window_size : Union(int, float)
            Window size

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(filtfilt(np.ones(window_size) / window_size, 1, self))

    def moving_median(self, window_size):
        """
        Moving median (has a sharper response to abrupt changes than the moving average)

        Parameters
        ----------
        window_size : Union(int, float)
            Window size (use around [3, 11])

        Returns
        -------
        FrameDependentNpArray
        """
        if window_size % 2 == 0:
            raise ValueError(f'window_size should be odd. Add or subtract 1. You provided {window_size}')
        if self.ndim == 3:
            window_size = [1, 1, window_size]
        elif self.ndim == 2:
            window_size = [1, window_size]
        elif self.ndim == 1:
            pass
        else:
            raise ValueError(f'dim should be 1, 2 or 3. You provided an array with {self.ndim} dimensions.')
        return self.dynamic_child_cast(medfilt(self, window_size))

    def _base_filter(self, freq, order, cutoff, btype, interp_nans):
        """
        Butterworth filter

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        order : Int
            Order of the filter
        cutoff : Int
            Cut-off frequency
        interp_nans : bool
            As this function does not work with nans, check if it is safe to interpolate and then interpolate over nans
        btype : str
            Filter type

        Returns
        -------

        """
        check_for_nans = self.check_for_nans()

        if not check_for_nans:
            # if there is no nans
            x = self.dynamic_child_cast(self)
        elif interp_nans and check_for_nans:
            # if there is some nans and it is safe to interpolate
            x = self.dynamic_child_cast(self.fill_values())
        else:
            # there is nans and we don't want to interpolate
            raise ValueError('filters do not work well with nans. Try interp_nans=True flag')

        nyquist = freq / 2
        corrected_freq = np.array(cutoff) / nyquist
        b, a = butter(N=order, Wn=corrected_freq, btype=btype)
        return filtfilt(b, a, x)

    def low_pass(self, freq, order, cutoff, interp_nans=False):
        """
        Low-pass Butterworth filter

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        order : Int
            Order of the filter
        cutoff : Int
            Cut-off frequency
        interp_nans : bool
            As this function does not work with nans, check if it is safe to interpolate and then interpolate over nans

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(
            self._base_filter(freq, order, cutoff, 'low', interp_nans)
        )

    def band_pass(self, freq, order, cutoff, interp_nans=False):
        """
        Band-pass Butterworth filter

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        order : Int
            Order of the filter
        cutoff : List-like
            Cut-off frequencies ([lower, upper])
        interp_nans : bool
            As this function does not work with nans, check if it is safe to interpolate and then interpolate over nans

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(
            self._base_filter(freq, order, cutoff, 'bandpass', interp_nans)
        )

    def band_stop(self, freq, order, cutoff, interp_nans=False):
        """
        Band-stop Butterworth filter

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        order : Int
            Order of the filter
        cutoff : List-like
            Cut-off frequencies ([lower, upper])
        interp_nans : bool
            As this function does not work with nans, check if it is safe to interpolate and then interpolate over nans

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(
            self._base_filter(freq, order, cutoff, 'bandstop', interp_nans)
        )

    def high_pass(self, freq, order, cutoff, interp_nans=False):
        """
        Band-stop Butterworth filter

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        order : Int
            Order of the filter
        cutoff : List-like
            Cut-off frequencies ([lower, upper])
        interp_nans : bool
            As this function does not work with nans, check if it is safe to interpolate and then interpolate over nans

        Returns
        -------
        FrameDependentNpArray
        """
        return self.dynamic_child_cast(
            self._base_filter(freq, order, cutoff, 'high', interp_nans)
        )

    def fft(self, freq, only_positive=True, axis=-1):
        """
        Performs a discrete Fourier Transform and return amplitudes and frequencies

        Parameters
        ----------
        freq : Union(int, float)
            Sample frequency
        only_positive : bool
            Returns only the positives frequencies if true (True by default)
        axis : int
            specifies the axis along which to performs the FFT. Performs defaults to the last axis (over frames)

        Returns
        -------
        amp (numpy.ndarray) and freqs (numpy.ndarray)
        """
        n = self.shape[axis]
        yfft = fftpack.fft(self, n)
        freqs = fftpack.fftfreq(n, 1. / freq)

        if only_positive:
            amp = 2 * np.abs(yfft) / n
            amp = amp[..., :int(np.floor(n / 2))]
            freqs = freqs[:int(np.floor(n / 2))]
        else:
            amp = np.abs(yfft) / n
        return amp, freqs

    def norm(self, axis=(0, 1)):
        """
        Compute the matrix norm. Same as np.sqrt(np.sum(np.power(x, 2), axis=0))

        Parameters
        ----------
        axis : int, tuple
            specifies the axis along which to compute the norm
        Returns
        -------
        FrameDependentNpArray
        """
        return np.linalg.norm(self, axis=axis)

    def detect_onset(self, threshold=0, above=1, below=0, threshold2=None, above2=1):
        """
        Detects onset in vector data. Inspired by Marcos Duarte's works.

        Parameters
        ----------
        threshold : double, optional
            minimum amplitude to detect
        above : double, optional
            minimum sample of continuous samples above `threshold` to detect
        below : double, optional
            minimum sample of continuous samples below `threshold` to ignore
        threshold2 : double, None, optional
            minimum amplitude of `above2` values in `x` to detect.
        above2
            minimum sample of continuous samples above `threshold2` to detect

        Returns
        -------
        idx : np.ndarray
            onset events
        """
        if self.ndim > 1:
            raise ValueError(f'detect_onset works only for vector (ndim < 2). Your data have {self.ndim} dimensions.')
        self[np.isnan(self)] = -np.inf
        idx = np.argwhere(self >= threshold).ravel()

        if np.any(idx):
            # initial & final indexes of almost continuous data
            idx = np.vstack(
                (idx[np.diff(np.hstack((-np.inf, idx))) > below + 1],
                 idx[np.diff(np.hstack((idx, np.inf))) > below + 1])
            ).T
            # indexes of almost continuous data longer or equal to `above`
            idx = idx[idx[:, 1] - idx[:, 0] >= above - 1, :]

            if np.any(idx) and threshold2:
                # minimum amplitude of above2 values in x
                ic = np.ones(idx.shape[0], dtype=bool)
                for irow in range(idx.shape[0]):
                    if np.count_nonzero(self[idx[irow, 0]: idx[irow, 1] + 1] >= threshold2) < above2:
                        ic[irow] = False
                idx = idx[ic, :]

        if not np.any(idx):
            idx = np.array([])
        return idx

    def detect_outliers(self, onset_idx=None, threshold=3):
        """
        Detects data that is `threshold` times the standard deviation calculated on the `onset_idx`

        Parameters
        ----------
        onset_idx : numpy.ndarray
            Array of onset (first column) and offset (second column). You can use detect_onset to have such a table
        threshold : int
            Multiple of standard deviation from which data is considered outlier

        Returns
        -------
        numpy masked array
        """
        if np.any(onset_idx):
            mask = np.zeros(self.shape, dtype='bool')
            for (inf, sup) in onset_idx:
                mask[inf:sup] = 1
            sigma = np.nanstd(self[mask])
            mu = np.nanmean(self[mask])
        else:
            sigma = np.nanstd(self)
            mu = np.nanmean(self)
        return np.ma.masked_where(np.abs(self) > mu + (threshold * sigma), self)

    def derivative(self, window=1):
        """
        Performs a derivative of the data, assuming the get_time_frames variable has the same length as the data,
        otherwise it returns an error

        Parameters
        ----------
        window : int
            Number of frame before and after to use. This amount of frames is therefore lost at begining and end of
            the data

        Returns
        -------
        numpy array
        """
        deriv = self.dynamic_child_cast(np.ndarray(self.shape))
        deriv[:, :, 0:window] = np.nan
        deriv[:, :, -window:] = np.nan
        deriv[:, :, window:-window] = (self[:, :, 2 * window:] - self[:, :, 0:-2 * window]) / \
                                      (self.get_time_frames[2 * window:] - self.get_time_frames[0:-2 * window])
        return deriv


class FrameDependentNpArrayCollection(list):
    """
    Collection of time frame array
    """

    # --- Get metadata methods

    def get_frame(self, f):
        """
        Get fth frame of the collection
        Parameters
        ----------
        f : int
            Frame to get
        Returns
        -------
        Collection of frame f
        """
        coll = FrameDependentNpArrayCollection()
        for element in self:
            coll.append(element.get_frame(f))
        return coll

    def get_num_segments(self):
        """
        Get the number of segments in the collection
        Returns
        -------
        n : int
        Number of segments in the collection
        """
        return len(self)

    def get_num_frames(self):
        """

        Returns
        -------
        The number of frames
        """
        if len(self) > 0:
            if len(self[0].shape) == 2:
                return 1
            else:
                return self[0].shape[2]  # Assume all meshes has the same number of frame, return the first one
        else:
            return -1
