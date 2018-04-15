import numpy as np
from pathlib import Path
import pandas as pd

from pyomeca import signal as pyosignal

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

    @staticmethod
    def to_2d():
        raise ValueError('to_2d should be called from a child class (e.g. Markers3d, Analogs3d, etc.)')

    @staticmethod
    def get_2d_labels():
        raise ValueError('get_2d_labels should be called from a child class (e.g. Markers3d, Analogs3d, etc.)')

    # --- Signal processing methods

    def rectify(self):
        return pyosignal.rectify(self)

    def moving_rms(self, window_size, method='filtfilt'):
        return pyosignal.moving_rms(self, window_size, method)

    def moving_average(self, window_size, method='filtfilt'):
        return pyosignal.moving_average(self, window_size, method)

    def moving_median(self, window_size):
        return pyosignal.moving_median(self, window_size)


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
