import numpy as np
from pathlib import Path
import pandas as pd


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
