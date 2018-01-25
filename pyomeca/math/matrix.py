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

    s = m.shape
    return np.reshape(m[0:3, :, :], (s[0] * s[1], s[2]), 'A').T


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


def rotate(rt, m):
    """Rotate markers m about rt
    :param rt: an homogeneous matrix of rototranslation (4x4xF). If F == 1 then the matrix is repeated to fit m size
    :type rt: numpy.array
    :param m: matrix of markers (3x4xF)
    :type m: numpy.array
    :return m rotated by rt
    """

    s_m = m.shape
    s_rt = rt.shape
    if s_rt[0] != s_m[0]:
        raise ValueError('Size of RT and M must match')

    if len(s_rt) == 2 and len(s_m) == 2:
        m2 = rt.dot(m)
    elif len(s_rt) == 2 and len(s_m) == 3:
        m2 = np.einsum('ij,jkl->ikl', rt, m)
    elif len(s_rt) == 3 and len(s_m) == 3:
        m2 = np.einsum('ijk,jlk->ilk', rt, m)
    else:
        raise ValueError('Size of RT and M must match coucou')

    return m2


def inv_rt(rt):
    """Returns the transposed (and, by definition, inverted) homogeneous 4x4xF matrix
    :param rt: an homogeneous matrix of rototranslation (4x4xF)
    :type rt: numpy.array
    :return rt transposed
    """

    print('coucou')
    return rt
