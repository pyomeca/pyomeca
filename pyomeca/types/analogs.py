from pyomeca.types.frame_dependent import FrameDependentNpArray
import numpy as np


class Analogs3d(FrameDependentNpArray):
    def __new__(cls, data=np.ndarray((1, 0, 0)), names=list(), *args, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            1xNxF matrix of analogs data
        names : list of string
            name of the analogs that correspond to second dimension of the matrix
        """
        if data.ndim == 2:
            data = Analogs3d.from_2d(data)

        if data.ndim == 3:
            s = data.shape
            if s[0] != 1:
                raise IndexError('Analogs3d must have a length of 1 on the first dimension')
            analog = data
        else:
            raise TypeError('Data must be 2d or 3d matrix')

        return super(Analogs3d, cls).__new__(cls, array=analog, *args, **kwargs)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, Analogs3d):
            return

    def get_num_analogs(self):
        """
        Returns
        -------
        The number of analogs
        """
        s = self.shape
        return s[1]

    def to_2d(self):
        """
        Takes a Analogs3d style matrix and returns a tabular matrix
        Returns
        -------
        Tabular matrix
        """
        return np.squeeze(self.T, axis=2)

    @staticmethod
    def from_2d(m):
        """
        Takes a tabular matrix and returns a Vectors3d
        Parameters
        ----------
        m : np.array
            A CSV tabular matrix (Fx3*N)
        Returns
        -------
        Vectors3d of data set
        """
        s = m.shape
        return Analogs3d(np.reshape(m.T, (1, s[1], s[0]), 'F'))

    def get_2d_labels(self):
        """
        Takes a Analogs style labels and returns 2d style labels
        Returns
        -------
        2d style labels
        """
        return self.get_labels
