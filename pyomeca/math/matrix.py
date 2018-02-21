# -*- coding: utf-8 -*-
"""

Matrix manipulation in pyomeca

"""

import numpy as np

from pyomeca import fileio as pyio
from pyomeca.types import RotoTrans
from pyomeca.types import Vectors3d


def reshape_2d_to_3d_matrix(m):
    """
    Takes a tabular matrix and returns a Vectors3d
    Parameters
    ----------
    m : np.array
        A CSV style matrix (Fx3*N)

    Returns
    -------
    Vectors3d of data set
    """

    s = m.shape
    if int(s[1] / 3) != s[1] / 3:
        raise IndexError("Number of columns must be divisible by 3")
    return Vectors3d(np.reshape(m.T, (3, int(s[1] / 3), s[0]), 'F'))


def reshape_3d_to_2d_matrix(m):
    """
    Takes a Vectors3d style matrix and returns a tabular matrix
    Parameters
    ----------
    m : Vectors3d
        Matrix to be reshaped

    Returns
    -------
    tabular matrix
    """

    return np.reshape(m[0:3, :, :], (3 * m.number_markers(), m.number_frames()), 'F').T


def define_axes(data_set, idx_axis1, idx_axis2, axes_name, axis_to_recalculate, idx_origin):
    """
    This function creates system of axes from axis1 and axis2
    Parameters
    ----------
    data_set : Vectors3d
        Whole data set
    idx_axis1 : list(int)
        First column is the beginning of the axis, second is the end. Rows are the markers to be mean
    idx_axis2 : list(int)
        First column is the beginning of the axis, second is the end. Rows are the markers to be mean
    axes_name : str
        Name of the axis1 and axis2 in that order ("xy", "yx", "xz", ...)
    axis_to_recalculate : str
        Which of the 3 axes to recalculate
    idx_origin : list(int)
        Markers to be mean to define the origin of the system of axes

    Returns
    -------
    System of axes
    """

    # Extract mean of each required axis indexes
    idx_axis1 = np.matrix(idx_axis1)
    idx_axis2 = np.matrix(idx_axis2)

    axis1 = pyio.extract_markers(data_set, idx_axis1[:, 1]) - pyio.extract_markers(data_set, idx_axis1[:, 0])
    axis2 = pyio.extract_markers(data_set, idx_axis2[:, 1]) - pyio.extract_markers(data_set, idx_axis2[:, 0])
    origin = pyio.extract_markers(data_set, np.matrix(idx_origin).reshape((len(idx_origin), 1)))

    axis1 = axis1[0:3, :, :].reshape(3, axis1.shape[2]).T
    axis2 = axis2[0:3, :, :].reshape(3, axis2.shape[2]).T

    # If we inverse axes_names, inverse axes as well
    axes_name_tp = ''.join(sorted(axes_name))
    if axes_name != axes_name_tp:
        axis1_copy = axis1
        axis1 = axis2
        axis2 = axis1_copy
        axes_name = axes_name_tp

    if axes_name[0] == "x":
        x = axis1
        if axes_name[1] == "y":
            y = axis2
            z = np.cross(x, y)
        elif axes_name[1] == "z":
            z = axis2
            y = np.cross(z, x)
        else:
            raise ValueError("Axes names should be 2 values of ""x"", ""y"" and ""z"" permutations)")

    elif axes_name[0] == "y":
        y = axis1
        if axes_name[1] == "z":
            z = axis2
            x = np.cross(y, z)
        else:
            raise ValueError("Axes names should be 2 values of ""x"", ""y"" and ""z"" permutations)")
    else:
        raise ValueError("Axes names should be 2 values of ""x"", ""y"" and ""z"" permutations)")

    # Normalize each vector
    x = x / np.matrix(np.linalg.norm(x, axis=1)).T
    y = y / np.matrix(np.linalg.norm(y, axis=1)).T
    z = z / np.matrix(np.linalg.norm(z, axis=1)).T

    # # Recalculate the temporary axis
    if axis_to_recalculate == "x":
        x = np.cross(y, z)
    elif axis_to_recalculate == "y":
        y = np.cross(z, x)
    elif axis_to_recalculate == "z":
        z = np.cross(x, y)
    else:
        raise ValueError("Axis to recalculate must be ""x"", ""y"" or ""z""")

    rt = RotoTrans(rt=np.zeros((4, 4, data_set.shape[2])))
    rt[0:3, 0, :] = x.T
    rt[0:3, 1, :] = y.T
    rt[0:3, 2, :] = z.T
    rt.set_translation(origin)
    return rt
