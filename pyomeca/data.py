# -*- coding: utf-8 -*-
"""

File IO of the S2M python library
Written by: Pariterre & Romain Martinez
Date: January 2018
"""

import numpy as np
import pandas
from pyomeca import math


def load_data(file_name, mark_names=None, mark_idx=list(), header=None):
    """ This function reads CSV or C3D files and classes them according to the mark_names string list or
    the mark_idx integer list
    :param file_name: path + name of the file (including extension)
    :type file_name: string
    :param mark_names: list depicting the marker to keep (the order of the marker is kept in return data)
    :type mark_names: list of string
    :param list mark_idx: list depicting the marker to keep (the order of the marker is kept in return data)
    :type mark_idx: list of integer
    :param header: number of line of header as needed by pandas.read_csv
    :type header: int
    :return data
    """

    data_frame = pandas.read_csv(file_name, header=header)
    data_set = math.matrix.reshape_2d_to_3d_matrix(data_frame.values)
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
    mark_idx : int
        idx of marker to keep (order is kept in the returned data)
    Returns
    -------
    numpy.array
        extracted data
    """

    s = len(numpy.shape(mark_idx))
    if s == 2:
        data = numpy.mean(m[:, numpy.mean(mark_idx, axis=0).astype(int), :])
    elif s == 3:
        data = m[:, mark_idx, :]
    else:
        raise NotImplementedError('extract_data works only on Fx3*N or 3xNxF matrices')
    return data

