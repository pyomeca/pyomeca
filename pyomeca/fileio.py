# -*- coding: utf-8 -*-
"""

File IO in PyoMeca library

"""

import numpy as np
import pandas as pd

from pyomeca.math import matrix


def read_csv(file_name, first_row=None, first_column=0, idx=None, header=None, names=None, kind='markers',
             delimiter=',', prefix=None):
    """
    Read CSV data and convert to Vectors3d format
    Parameters
    ----------
    file_name : Union[str, Path]
        Path of file
    first_row : int
        Index of first rows of data (0th indexed)
    first_column : int
        Index of first column of data (0th indexed)
    idx : list(int)
        Order of columns given by index
    header : int
        row of the header (0th indexed)
    names : list(str)
        Order of columns given by names, if both names and idx are provided, an error occurs
    kind : str
        Kind of data to read (markers or analogs)
    delimiter : str
        Delimiter of the CSV file
    prefix : str
        Prefix to remove in the header

    Returns
    -------
    Data set in Vectors3d format
    """

    if names and idx:
        raise ValueError("names and idx can't be set simultaneously, please select only one")

    # read the file
    data = pd.read_csv(file_name, delimiter=delimiter, header=header, skiprows=np.arange(header + 1, first_row))
    data.drop(data.columns[:first_column], axis=1, inplace=True)
    # get column names
    column_names = data.columns.tolist()
    # separate and delete empty names
    column_names = [icol.split(prefix)[1] for icol in column_names if icol[:7] != 'Unnamed']

    if kind == 'markers':
        data = matrix.reshape_2d_to_3d_matrix(data.values)
        if not idx:
            # find names in column_names
            idx = np.argwhere(np.in1d(np.array(column_names),
                                      np.array(names))).ravel()
        data = extract_markers(data, idx)
    elif kind == 'analogs':
        # TODO: implements for analogs
        pass
    else:
        raise ValueError('kind should be "markers" or "analogs"')

    return data


def read_c3d():
    # TODO: implements for c3d
    pass


def extract_markers(m, mark_idx):
    """
    # TODO: description
    Parameters
    ----------
    m : Vectors3D
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
        raise IndexError('extract_markers works only on 3xNxF matrices and mark_idx must be a ixj array')
    return data


if __name__ == '__main__':
    FILENAME = '/home/romain/Documents/codes/pyomeca/test/data/markers.csv'
    # read_csv(file_name=FILENAME, first_row=5, first_column=2, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], header=2,
    #          prefix=':')

    m_csv_4 = read_csv(FILENAME, first_row=5, first_column=2, header=2,
                       names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], prefix=':')
