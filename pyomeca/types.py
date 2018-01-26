# -*- coding: utf-8 -*-
"""

Definition of different container in PyoMeca library

"""

import numpy as np


class Vectors3d(np.ndarray):
    """

    Definition of a matrix of vector

    """

    def __new__(cls, positions=np.ndarray((3, 0, 0)), names=list()):
        """
        Parameters
        ----------
        names : list of string
            name of the marker that correspond to second dimension of the positions matrix

        positions : np.ndarray
            3xNxF matrix of marker positions
        """

        # Add positions
        if not isinstance(positions, np.ndarray):
            raise TypeError('Vectors3d must be a numpy array')
        s = positions.shape
        if s[0] != 3:
            raise IndexError('Vectors3d must have a length of 3 on the first dimension')
        pos = np.ones((4, s[1], s[2]))
        pos[0:3, :, :] = positions

        obj = np.asarray(pos).view(cls)

        # Finally, we must return the newly created object:
        return obj

    def number_frames(self):
        s = self.shape
        if len(s) == 2:
            return 1
        else:
            return s[2]

    def number_markers(self):
        s = self.shape
        return s[1]
