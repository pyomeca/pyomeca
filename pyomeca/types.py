# -*- coding: utf-8 -*-
"""

Definition of different container in PyoMeca library

"""

import numpy as np


class Vectors3d:
    """

    Definition of a matrix of vector

    """
    def __init__(self, names, positions):
        """

        Parameters
        ----------
        names : list of string
            name of the marker that correspond to second dimension of the positions matrix

        position : np.ndarray
            3xNxF matrix of marker positions
        """

        if not isinstance(positions, np.ndarray):
            raise TypeError('')

        self.names = names
        self.pos = positions
