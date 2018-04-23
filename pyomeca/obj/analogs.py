from pyomeca.obj.frame_dependent import FrameDependentNpArray
import numpy as np
from pyomeca import signal as pyosignal


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

    # --- Get metadata methods

    def get_num_analogs(self):
        """
        Returns
        -------
        The number of analogs
        """
        s = self.shape
        return s[1]

    def get_2d_labels(self):
        """
        Takes a Analogs style labels and returns 2d style labels
        Returns
        -------
        2d style labels
        """
        return self.get_labels

    # --- Fileio methods (from_*)

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

    # --- Fileio methods (to_*)

    def to_2d(self):
        """
        Takes a Analogs3d style matrix and returns a tabular matrix
        Returns
        -------
        Tabular matrix
        """
        return np.squeeze(self.T, axis=2)

    # --- Signal processing methods

    def moving_rms(self, window_size, method='filtfilt'):
        return Analogs3d(pyosignal.moving_rms(self, window_size, method))

    def moving_average(self, window_size, method='filtfilt'):
        return Analogs3d(pyosignal.moving_average(self, window_size, method))

    def moving_median(self, window_size):
        return Analogs3d(pyosignal.moving_median(self, window_size))

    def low_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.low_pass(self, freq, order, cutoff))

    def band_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.band_pass(self, freq, order, cutoff))

    def band_stop(self, freq, order, cutoff):
        return Analogs3d(pyosignal.band_stop(self, freq, order, cutoff))

    def high_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.high_pass(self, freq, order, cutoff))

    def time_normalization(self, time_vector=np.linspace(0, 100, 101), axis=-1):
        return Analogs3d(pyosignal.time_normalization(self, time_vector, axis=axis))

    def fill_values(self, axis=-1):
        return Analogs3d(pyosignal.fill_values(self, axis))
