# -*- coding: utf-8 -*-
"""

Matrix manipulation of S2M python library
Written by: Pariterre & Romain Martinez
Date: January 2018

"""

import numpy as np
from pyomeca.types import Vectors3d


def reshape_2d_to_3d_matrix(m):
    """
    Convert a Fx3*N into a 3xNxF matrix
    :param m: Fx3*N matrix
    :type m: numpy.array
    :return a 3xNxF matrix
    """

    s = m.shape
    if int(s[1]/3) != s[1]/3:
        raise IndexError("Number of columns must be divisible by 3")
    return Vectors3d(np.reshape(m.T, (3, int(s[1]/3), s[0]), 'F'))


def reshape_3d_to_2d_matrix(m):
    """Convert a 3xNxF into a Fx3*N matrix
    :param numpy.array m: 3xNxF matrix
    :type m: numpy.array
    :return a Fx3*N matrix
    """

    return np.reshape(m[0:3, :, :], (3 * m.number_markers(), m.number_frames()), 'A').T


def define_axes(axis1, axis2, axesName, keep, origin):
    """Creates and returns 4x4xF systems of axes made from axis1 and axis2 data.
    :param axis1: matrix of markers (3xN or 3x1xN) that defines the first axis
    :type axis1: numpy.array
    :param axis2: matrix of markers (3xN or 3x1xN) that defines the second axis
    :type axis2: numpy.array
    :param axesName: a string that defines the axis ('xy', 'yx', 'xz', 'zx', 'yz' or 'zy')
    :type axesName: string
    :param keep: index of kept axis
    :type keep: int
    :param origin: matrix of markers (3xN or 3x1xN) that defines the origin position
    :type origin: numpy.array
    :return homogenous hypermatrix (4x4xF)
    """

    # axis1 = s2m_data.extract_data(data_set, idx_xaxis)
    # axis2 = s2m_data.extract_data(data_set, idx_yaxis)
    # origin = s2m_data.extract_data(data_set, idx_origin)
