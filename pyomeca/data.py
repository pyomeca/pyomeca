# -*- coding: utf-8 -*-
"""

File IO of the S2M python library
Written by: Pariterre & Romain Martinez
Date: January 2018
"""

import numpy as np
import pandas
from pyomeca.math import matrix


def load_data(file_name, header=None, mark_idx=list(), mark_names=None):
    """
    Load CSV or C3D data
    Parameters
    ----------
    file_name : str
        Path of file
    mark_names : list(str)
        Order of markers given by names, if both mark_names and mark_idx are provided, mark_idx prevails
    mark_idx : list(int)
        Order of markers given by index,
    header : int
        Number of rows in the csv file header, this parameter is ignored when the file is a C3D

    Returns
    -------
    Data set in Vectors3d format
    """

    data_frame = pandas.read_csv(file_name, header=header)
    data_set = matrix.reshape_2d_to_3d_matrix(data_frame.values)
    if not mark_names and not mark_idx:
        return data_set
    elif not mark_names:
        return extract_data(data_set, mark_idx)
    else:
        raise NotImplementedError('the use of mark_name is not implemented yet')


def extract_data(m, mark_idx):
    """

    Parameters
    ----------
    m : numpy.array
        a Fx3*N or 3xNxF matrix of marker position
    mark_idx : list(int)
        idx of marker to keep (order is kept in the returned data).
        If mark_idx has more than one row, output is the mean of the markers over the columns.
    Returns
    -------
    numpy.array
        extracted data
    """
    mark_idx = np.matrix(mark_idx)

    try:
        data = m[:, np.array(mark_idx)[0, :], :]
        for i in range(1, mark_idx.shape[0]):
            data += m[:, np.array(mark_idx)[i, :], :]
        data /= mark_idx.shape[0]
    except IndexError:
        raise IndexError('extract_data works only on 3xNxF matrices and mark_idx must be a ixj array')
    return data
