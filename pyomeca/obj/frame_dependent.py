from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import filtfilt, medfilt, butter

from pyomeca.thirdparty import btk


class FrameDependentNpArray(np.ndarray):
    def __new__(cls, array=np.ndarray((0, 0, 0)), *args, **kwargs):
        """
        Convenient wrapper around np.ndarray for time related data
        Parameters
        ----------
        array : np.ndarray
            A 3 dimensions matrix where 3rd dimension is the frame
        -------

        """
        if not isinstance(array, np.ndarray):
            raise TypeError('FrameDependentNpArray must be a numpy array')

        # metadata
        obj = np.asarray(array).view(cls, *args, **kwargs)
        obj.__array_finalize__(array)
        return obj

    def __array_finalize__(self, obj):
        # Allow slicing
        if obj is None or not isinstance(obj, FrameDependentNpArray):
            self._current_frame = 0
            self.get_first_frame = []
            self.get_last_frame = []
            self.get_rate = []
            self.get_labels = []
            self.get_unit = []
        else:
            self._current_frame = getattr(obj, '_current_frame')
            self.get_first_frame = getattr(obj, 'get_first_frame')
            self.get_last_frame = getattr(obj, 'get_last_frame')
            self.get_rate = getattr(obj, 'get_rate')
            self.get_labels = getattr(obj, 'get_labels')
            self.get_unit = getattr(obj, 'get_unit')

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

    def __next__(self):
        if self._current_frame > self.shape[2]:
            raise StopIteration
        else:
            self._current_frame += 1
            return self.get_frame(self._current_frame)

    @classmethod
    def _get_class_name(cls):
        return cls.__name__

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

    # --- Fileio methods (from_*)

    @classmethod
    def from_csv(cls, filename, first_row=None, first_column=0, idx=None,
                 header=None, names=None, delimiter=',', prefix=None):
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
        if not header:
            skiprows = np.arange(1, first_row)
        else:
            skiprows = np.arange(header + 1, first_row)

        data = pd.read_csv(str(filename), delimiter=delimiter, header=header, skiprows=skiprows)
        data.drop(data.columns[:first_column], axis=1, inplace=True)
        column_names = data.columns.tolist()
        if header and cls._get_class_name() == 'Markers3d':
            column_names = [icol.split(prefix)[-1] for icol in column_names if
                            (len(icol) >= 7 and icol[:7] != 'Unnamed')]
        metadata = {'get_first_frame': [], 'get_last_frame': [], 'get_rate': [], 'get_labels': [], 'get_unit': []}
        if names:
            metadata.update({'get_labels': names})
        else:
            names = column_names

        return cls._to_vectors(data=data.values, idx=idx, all_names=column_names, target_names=names, metadata=metadata)

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
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(str(filename))
        reader.Update()
        acq = reader.GetOutput()

        channel_names = []

        current_class = cls._get_class_name()
        if current_class == 'Markers3d':
            flat_data = {i.GetLabel(): i.GetValues() for i in btk.Iterate(acq.GetPoints())}
            metadata = {
                'get_num_markers': acq.GetPointNumber(),
                'get_num_frames': acq.GetPointFrameNumber(),
                'get_first_frame': acq.GetFirstFrame(),
                'get_last_frame': acq.GetLastFrame(),
                'get_rate': acq.GetPointFrequency(),
                'get_unit': acq.GetPointUnit()
            }
            data = np.full([metadata['get_num_frames'], 3 * metadata['get_num_markers']], np.nan)
            for i, (key, value) in enumerate(flat_data.items()):
                data[:, i * 3: i * 3 + 3] = value
                channel_names.append(key.split(prefix)[-1])
        elif current_class == 'Analogs3d':
            flat_data = {i.GetLabel(): i.GetValues() for i in btk.Iterate(acq.GetAnalogs())}
            metadata = {
                'get_num_analogs': acq.GetAnalogNumber(),
                'get_num_frames': acq.GetAnalogFrameNumber(),
                'get_first_frame': acq.GetFirstFrame(),
                'get_last_frame': acq.GetLastFrame(),
                'get_rate': acq.GetAnalogFrequency(),
                'get_unit': []
            }
            data = np.full([metadata['get_num_frames'], metadata['get_num_analogs']], np.nan)
            for i, (key, value) in enumerate(flat_data.items()):
                data[:, i] = value.ravel()
                channel_names.append(key.split(prefix)[-1])
        else:
            raise ValueError('from_c3d should be called from Markers3d or Analogs3d')
        if names:
            metadata.update({'get_labels': names})
        else:
            metadata.update({'get_labels': []})
            names = channel_names

        return cls._to_vectors(data=data,
                               idx=idx,
                               all_names=channel_names,
                               target_names=names,
                               metadata=metadata)

    @classmethod
    def _to_vectors(cls, data, idx, all_names, target_names, metadata=None):
        data[data == 0.0] = np.nan  # because sometimes nan are replace by 0.0
        if not idx:
            # find names in column_names
            idx = []
            for i, m in enumerate(target_names):
                idx.append([i for i, s in enumerate(all_names) if m in s][0])

        data = cls.__new__(cls, data)
        data = data.get_specific_data(idx)

        data.get_first_frame = metadata['get_first_frame']
        data.get_last_frame = metadata['get_last_frame']
        data.get_rate = metadata['get_rate']
        data.get_unit = metadata['get_unit']
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
        idx = np.matrix(idx)
        try:
            data = self[:, np.array(idx)[0, :], :]
            for i in range(1, idx.shape[0]):
                data += self[:, np.array(idx)[i, :], :]
            data /= idx.shape[0]
        except IndexError:
            raise IndexError('get_specific_data works only on 3xNxF matrices and idx must be a ixj array')
        return data

    # --- Fileio methods (to_*)

    def to_csv(self, file_name, header=False):
        """
        Write a csv file from a FrameDependentNpArray
        Parameters
        ----------
        file_name : string
            path of the file to write
        header : bool
            Write header with labels (default False)
        """
        file_name = Path(file_name)
        # Make sure the directory exists, otherwise create it
        if not file_name.parents[0].is_dir():
            file_name.parents[0].mkdir()

        # Convert markers into 2d matrix
        data = pd.DataFrame(self.to_2d())

        # Get the 2d style labels
        if header:
            header = self.get_2d_labels()

        # Write into the csv file
        data.to_csv(file_name, index=False, header=header)

    # --- Plot method

    def plot(self, x=None, ax=None, fmt='k', lw=1, label=None, alpha=1):
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

        if self.shape[0] == 1 and self.shape[1] == 1:
            current = self.squeeze()
        else:
            current = self
        if np.any(x):
            ax.plot(x, current, fmt, lw=lw, label=label, alpha=alpha)
        else:
            ax.plot(current, fmt, lw=lw, label=label, alpha=alpha)
        return ax

    # --- Signal processing methods

    def rectify(self):
        """
        Rectify a signal (i.e., get absolute values)

        Returns
        -------
        FrameDependentNpArray
        """
        return np.abs(self)

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

    def normalization(self, ref=None, scale=100):
        """
        Normalize a signal against `ref` (x's max if empty) on a scale of `scale`

        Parameters
        ----------
        ref : Union(int, float)
            reference value
        scale
            Scale on which to express x (100 by default)

        Returns
        -------
        FrameDependentNpArray
        """
        if not ref:
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
        original_time_vector = np.arange(0, self.shape[axis])
        x = self.copy()

        def fct(m):
            """Simple function to interpolate along an axis"""
            w = np.isnan(m)
            m[w] = 0
            f = UnivariateSpline(original_time_vector, m, w=~w)
            return f(original_time_vector)

        return self.dynamic_child_cast(np.apply_along_axis(fct, axis=axis, arr=x))

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
            raise ValueError(f'x.dim should be 1, 2 or 3. You provided an array with {x.ndim} dimensions.')
        return self.dynamic_child_cast(medfilt(self, window_size))

    def low_pass(self, freq, order, cutoff):
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

        Returns
        -------
        FrameDependentNpArray
        """
        nyquist = freq / 2
        corrected_freq = np.array(cutoff) / nyquist
        b, a = butter(N=order, Wn=corrected_freq, btype='low')
        return self.dynamic_child_cast(filtfilt(b, a, self))

    def band_pass(self, freq, order, cutoff):
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

        Returns
        -------
        FrameDependentNpArray
        """
        nyquist = freq / 2
        corrected_freq = np.array(cutoff) / nyquist
        b, a = butter(N=order, Wn=corrected_freq, btype='bandpass')
        return self.dynamic_child_cast(filtfilt(b, a, self))

    def band_stop(self, freq, order, cutoff):
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

        Returns
        -------
        FrameDependentNpArray
        """
        nyquist = freq / 2
        corrected_freq = np.array(cutoff) / nyquist
        b, a = butter(N=order, Wn=corrected_freq, btype='bandstop')
        return self.dynamic_child_cast(filtfilt(b, a, self))

    def high_pass(self, freq, order, cutoff):
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

        Returns
        -------
        FrameDependentNpArray
        """
        nyquist = freq / 2
        corrected_freq = np.array(cutoff) / nyquist
        b, a = butter(N=order, Wn=corrected_freq, btype='high')
        return self.dynamic_child_cast(filtfilt(b, a, self))

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
        y = np.ma.masked_where(np.abs(self) > mu + (threshold * sigma), self)
        return y


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