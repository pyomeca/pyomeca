import ezc3d
import numpy as np

from pyomeca.obj.frame_dependent import FrameDependentNpArray


class Markers3d(FrameDependentNpArray):
    def __new__(cls, data=np.ndarray((3, 0, 0)), names=list(), *args, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            3xNxF matrix of marker positions
        names : list of string
            name of the marker that correspond to second dimension of the positions matrix
        """
        if data.ndim == 2:
            data = np.array(Markers3d.from_2d(data))

        if data.ndim == 3:
            s = data.shape
            if s[0] == 3:
                pos = np.ones((4, s[1], s[2]))
                pos[0:3, :, :] = data
            elif s[0] == 4:
                pos = data
            else:
                raise IndexError('Vectors3d must have a length of 3 on the first dimension')
        else:
            raise TypeError('Data must be 2d or 3d matrix')

        return super(Markers3d, cls).__new__(cls, array=pos, *args, **kwargs)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, Markers3d):
            return

    # --- Get metadata methods

    def get_num_markers(self):
        """
        Returns
        -------
        The number of markers
        """
        s = self.shape
        return s[1]

    def get_2d_labels(self):
        """
        Takes a Markers3d style labels and returns 2d style labels
        Returns
        -------
        2d style labels
        """
        return [i + axe for i in self.get_labels for axe in ['_X', '_Y', '_Z']]

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
        if s[1] % 3 != 0:
            raise IndexError("Number of columns must be divisible by 3")
        return Markers3d(np.reshape(m.T, (3, int(s[1] / 3), s[0]), 'F'))

    # --- Fileio methods (to_*)

    def to_2d(self):
        """
        Takes a Markers3d style matrix and returns a tabular matrix
        Returns
        -------
        Tabular matrix
        """
        return np.reshape(self[0:3, :, :], (3 * self.get_num_markers(), self.get_num_frames()), 'F').T

    @staticmethod
    def _parse_c3d(c3d, prefix):
        """
        Implementation on how to read c3d header and parameter for markers
        Parameters
        ----------
        c3d : ezc3d
            ezc3d class

        prefix : str, optional
            Participant's prefix

        Returns
        -------
        metadata, channel_names, data
        """
        channel_names = [i.c_str().split(prefix)[-1] for i in
                         c3d.parameters().group('POINT').parameter('LABELS').valuesAsString()]
        metadata = {
            'get_num_markers': c3d.header().nb3dPoints(),
            'get_num_frames': c3d.header().nbFrames(),
            'get_first_frame': c3d.header().firstFrame(),
            'get_last_frame': c3d.header().lastFrame(),
            'get_rate': c3d.header().frameRate(),
            'get_unit': c3d.parameters().group('POINT').parameter('UNITS').valuesAsString()[0].c_str()
        }
        data = c3d.get_points()
        return data, channel_names, metadata

    # --- Linear algebra methods

    def rotate(self, rt):
        """
        Parameters
        ----------
        rt : RotoTrans
            Rototranslation matrix to rotate about

        Returns
        -------
        A new Vectors3d rotated
        """
        s_m = self.shape
        s_rt = rt.shape

        if len(s_rt) == 2 and len(s_m) == 2:
            m2 = rt.dot(self)
        elif len(s_rt) == 2 and len(s_m) == 3:
            m2 = np.einsum('ij,jkl->ikl', rt, self)
        elif len(s_rt) == 3 and len(s_m) == 3:
            m2 = np.einsum('ijk,jlk->ilk', rt, self)
        else:
            raise ValueError('Size of RT and M must match')

        return Markers3d(data=m2)

    def norm(self):
        """
        Compute the Euclidean norm of vectors
        Returns:
        -------
        Norm
        """
        square = self[0:3, :, :] ** 2
        sum_square = np.sum(square, axis=0)
        norm = np.sqrt(sum_square)
        return norm
