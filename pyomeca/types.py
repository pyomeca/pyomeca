# -*- coding: utf-8 -*-
"""

Definition of different container in pyomeca

"""

import numpy as np


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

        # Finally, we must return the newly created object:
        cls.current_frame = 0
        return np.asarray(array).view(cls, *args, **kwargs)

    def n_frames(self):
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
        return self[:, :, f]

    def __next__(self):
        if self.current_frame > self.shape[2]:
            raise StopIteration
        else:
            self.current_frame += 1
            return self.get_frame(self.current_frame)


class RotoTransCollection(list):
    """
    List of RotoTrans
    """

    def get_rt(self, i):
        """
        Get a specific RotoTrans of the collection
        Parameters
        ----------
        i : int
            Index of the RotoTrans in the collection

        Returns
        -------
        All frame of RotoTrans of index i
        """
        return self[i]

    def get_frame(self, f):
        """
        Get the RotoTransCollection for frame f
        Parameters
        ----------
        f : int
            Frame to get

        Returns
        -------
        RotoTransCollection of frame f
        """
        rt_coll = RotoTransCollection()
        for rt in self:
            rt_coll.append(rt.get_frame(f))
        return rt_coll

    def n_rt(self):
        """
        Get the number of RotoTrans in the collection
        Returns
        -------
        n : int
        Number of RotoTrans in the collection
        """
        return len(self)


class RotoTrans(FrameDependentNpArray):
    def __new__(cls, rt=np.eye(4), angles=(0, 0, 0), angle_sequence="", translations=(0, 0, 0), *args, **kwargs):
        """

        Parameters
        ----------
        rt : FrameDependentNpArray (4x4xF)
            Rototranslation matrix sorted in 4x4xF, default is the matrix that don't rotate nor translate the system, is
            ineffective if angles is provided
        angles : tuple of angle (floats)
            Euler angles of the rototranslation, angles parameter is ineffective if angles_sequence if not defined, but
            will override rt
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of 3 axes (e.g. "xyz", "yzx", ...)
        translations : tuple of translation (floats)
            First 3 rows of 4th row, translation is ineffective if angles is not provided
        """

        # Determine if we construct RotoTrans from rt or angles/translations
        if angle_sequence:
            rt = cls.rt_from_euler_angles(angles=angles, angle_sequence=angle_sequence, translations=translations)

        else:
            s = rt.shape
            if s[0] != 4 or s[1] != 4:
                raise IndexError('RotoTrans must by a 4x4xF matrix')
            # Make sure last line reads [0, 0, 0, 1]
            if len(s) == 2:
                rt[3, :] = np.array([0, 0, 0, 1])
            else:
                rt[3, 0:3, :] = 0
                rt[3, 3, :] = 1

        # Finally, we must return the newly created object:
        return super(RotoTrans, cls).__new__(cls, array=rt, *args, **kwargs)

    def get_euler_angles(self, angle_sequence):
        """

        Parameters
        ----------
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
        Returns
        -------
        angles : Markers3d
            Euler angles associated with RotoTrans
        """
        if self.n_frames() > 1:
            raise NotImplementedError("get_euler_angles on more than one frame at a time is not implemented yet")

        angles = np.ndarray(shape=(len(angle_sequence), 1))

        if angle_sequence == "x":
            angles[0] = np.arcsin(self[2, 1])
        elif angle_sequence == "y":
            angles[0] = np.arcsin(self[0, 2])
        elif angle_sequence == "z":
            angles[0] = np.arcsin(self[1, 0])
        elif angle_sequence == "xy":
            angles[0] = np.arcsin(self[2, 1])
            angles[1] = np.arcsin(self[0, 2])
        elif angle_sequence == "xz":
            angles[0] = -np.arcsin(self[1, 2])
            angles[1] = -np.arcsin(self[0, 1])
        elif angle_sequence == "yx":
            angles[0] = -np.arcsin(self[2, 0])
            angles[1] = -np.arcsin(self[1, 2])
        elif angle_sequence == "yz":
            angles[0] = np.arcsin(self[0, 2])
            angles[1] = np.arcsin(self[1, 0])
        elif angle_sequence == "zx":
            angles[0] = np.arcsin(self[1, 0])
            angles[1] = np.arcsin(self[2, 1])
        elif angle_sequence == "zy":
            angles[0] = -np.arcsin(self[0, 1])
            angles[1] = -np.arcsin(self[2, 0])
        elif angle_sequence == "xyz":
            angles[0] = np.arctan2(self[1, 2], self[2, 2])
            angles[1] = np.arcsin(self[0, 1])
            angles[2] = np.arctan2(-self[0, 1], self[0, 0])
        elif angle_sequence == "xzy":
            angles[0] = np.arctan2(self[2, 1], self[1, 1])
            angles[2] = np.arctan2(self[0, 2], self[0, 0])
            angles[1] = -np.arcsin(self[0, 1])
        elif angle_sequence == "xzy":
            angles[1] = -np.arcsin(self[1, 2])
            angles[0] = np.arctan2(self[0, 2], self[2, 2])
            angles[2] = np.arctan2(self[1, 0], self[1, 1])
        elif angle_sequence == "yzx":
            angles[2] = np.arctan2(-self[1, 2], self[1, 1])
            angles[0] = np.arctan2(-self[2, 0], self[0, 0])
            angles[1] = np.arcsin(self[1, 2])
        elif angle_sequence == "zxy":
            angles[1] = np.arcsin(self[2, 1])
            angles[2] = np.arctan2(-self[2, 0], self[2, 2])
            angles[0] = np.arctan2(-self[0, 1], self[1, 1])
        elif angle_sequence == "zyz":
            angles[0] = np.arctan2(self[1, 2], self[0, 2])
            angles[1] = np.arccos(self[2, 2])
            angles[2] = np.arctan2(self[2, 1], -self[2, 0])
        elif angle_sequence == "zxz":
            angles[0] = np.arctan2(self[0, 2], -self[1, 2])
            angles[1] = np.arccos(self[2, 2])
            angles[2] = np.arctan2(self[2, 0], self[2, 1])
        elif angle_sequence == "zyzz":
            angles[0] = np.arctan2(self[1, 2], self[0, 2])
            angles[1] = np.arccos(self[2, 2])
            angles[2] = np.arctan2(self[2, 1], -self[2, 0])

        return angles

    @staticmethod
    def rt_from_euler_angles(angles=(0, 0, 0), angle_sequence="", translations=(0, 0, 0)):
        """

        Parameters
        ----------
        angles : tuple of angle (floats)
            Euler angles of the rototranslation
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
        translations

        Returns
        -------
        rt : RotoTrans
            The rototranslation associated to the input parameters
        """
        if angle_sequence == "zyzz":
            angles = (angles[0], angles[1], angles[2] - angles[0])
            angle_sequence = "zyz"

        if len(angles) is not len(angle_sequence):
            raise IndexError("angles and angles_sequence must be the same size")

        matrix_to_prod = list()
        try:
            for i in range(len(angles)):
                if angle_sequence[i] == "x":
                    a = angles[i]
                    matrix_to_prod.append(np.array([[1, 0, 0],
                                                    [0, np.cos(a), np.sin(a)],
                                                    [0, -np.sin(a), np.cos(a)]]).T)
                elif angle_sequence[i] == "y":
                    a = angles[i]
                    matrix_to_prod.append(np.array([[np.cos(a), 0, -np.sin(a)],
                                                    [0, 1, 0],
                                                    [np.sin(a), 0, np.cos(a)]]).T)
                elif angle_sequence[i] == "z":
                    a = angles[i]
                    matrix_to_prod.append(np.array([[np.cos(a), np.sin(a), 0],
                                                    [-np.sin(a), np.cos(a), 0],
                                                    [0, 0, 1]]).T)
                else:
                    raise ValueError("angle_sequence must be a permutation of axes (e.g. ""xyz"", ""yzx"", ...)")
        except IndexError:
            raise ValueError("angle_sequence must be a permutation of axes (e.g. ""xyz"", ""yzx"", ...)")

        r = np.eye(3)
        for i in range(len(angles)):
            r = r.dot(matrix_to_prod[i])

        rt = np.eye(4)
        rt[0:3, 0:3] = r
        rt[0:3, 3] = translations[0:3]

        return RotoTrans(rt)

    def rotation(self):
        """
        Returns
        -------
        Rotation part of the RotoTrans
        """
        return self[0:3, 0:3]

    def set_rotation(self, r):
        """
        Set rotation part of the RotoTrans
        Parameters
        ----------
        r : np.array
            A 3x3 rotation matrix
        """
        self[0:3, 0:3] = r

    def translation(self):
        """
        Returns
        -------
        Translation part of the RotoTrans
        """
        return self[0:3, 3]

    def set_translation(self, t):
        """
        Set translation part of the RotoTrans
        Parameters
        ----------
        t : np.array
            A 3x1 vector
        """
        self[0:3, 3, :] = t[0:3, :, :].reshape(3, t.shape[2])

    def transpose(self):
        """

        Returns
        -------
        Rt_t : RotoTrans
            Transposed RotoTrans matrix ([R.T -R.T*t],[0 0 0 1])
        """
        # Create a matrix with the transposed rotation part
        rt_t = RotoTrans(rt=np.ndarray((4, 4, self.n_frames())))
        rt_t[0:3, 0:3, :] = np.transpose(self[0:3, 0:3, :], (1, 0, 2))

        # Fill the last column and row with 0 and bottom corner with 1
        rt_t[3, 0:3, :] = 0
        rt_t[0:3, 3, :] = 0
        rt_t[3, 3, :] = 1

        # Transpose the translation part
        t = Markers3d(data=np.reshape(self[0:3, 3, :], (3, 1, self.n_frames())))
        rt_t[0:3, 3, :] = t.rotate(-rt_t)[0:3, :].reshape((3, self.n_frames()))

        # Return transposed matrix
        return rt_t

    def inverse(self):
        """

        Returns
        -------
        Inverse of the RotoTrans matrix (which is by definition the transposed matrix)
        """
        return self.transpose()


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

        s = data.shape
        if s[0] == 3:
            pos = np.ones((4, s[1], s[2]))
            pos[0:3, :, :] = data
        elif s[0] == 4:
            pos = data
        else:
            raise IndexError('Vectors3d must have a length of 3 on the first dimension')
        return super(Markers3d, cls).__new__(cls, array=pos, *args, **kwargs)

    def n_markers(self):
        """
        Returns
        -------
        Get the number of markers
        """
        s = self.shape
        return s[1]

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


class GeneralizedCoordinate(FrameDependentNpArray):
    def __new__(cls, q=np.ndarray((0, 1, 0)), *args, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            nQxNxF matrix of marker positions
        """

        # Reshape if the user sent a 2d instead of 3d shape
        if len(q.shape) == 2:
            q = np.reshape(q, (q.shape[0], 1, q.shape[1]))

        if q.shape[1] != 1:
            raise IndexError('Generalized coordinates can''t have multiple columns')

        return super(GeneralizedCoordinate, cls).__new__(cls, array=q, *args, **kwargs)


class Analogs3d(FrameDependentNpArray):
    def __new__(cls, data=np.ndarray((3, 0, 0)), names=list(), *args, **kwargs):

        """
        Parameters
        ----------
        data : np.ndarray
            1xNxF matrix of analogs data
        names : list of string
            name of the analogs that correspond to second dimension of the matrix
        """
        return super(Analogs3d, cls).__new__(cls, array=data, *args, **kwargs)
