# -*- coding: utf-8 -*-
"""

File IO in pyomeca library

"""

import numpy as np
import pandas as pd

from pyomeca.math import matrix
from pyomeca.thirdparty import btk


def read_csv(file_name, first_row=None, first_column=0, idx=None, header=None, names=None, kind='markers',
             delimiter=',', prefix=None):
    """
    Read csv data and convert to Vectors3d format
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
    data = pd.read_csv(str(file_name), delimiter=delimiter, header=header, skiprows=np.arange(header + 1, first_row))
    data.drop(data.columns[:first_column], axis=1, inplace=True)
    # get column names
    column_names = data.columns.tolist()
    # separate and delete empty names
    column_names = [icol.split(prefix)[1] for icol in column_names if icol[:7] != 'Unnamed']
    # return data in pyomeca format
    return _to_vectors(data=data.values,
                       kind=kind,
                       idx=idx,
                       actual_names=column_names,
                       target_names=names)


def read_c3d(file_name, idx=None, names=None, kind='markers', prefix=None, get_metadata=False):
    """
    Read c3d data and convert to Vectors3d format
    Parameters
    ----------
    file_name : Union[str, Path]
        Path of file
    idx : list(int)
        Order of columns given by index
    names : list(str)
        Order of columns given by names, if both names and idx are provided, an error occurs
    kind : str
        Kind of data to read (markers or analogs)
    prefix : str
        Prefix to remove in the header
    get_metadata : bool
        Return a dict with metadata if true

    Returns
    -------
    Data set in Vectors3d format and metadata dict
    """
    # TODO: metadata in dict?
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(str(file_name))
    reader.Update()
    acq = reader.GetOutput()

    channel_names = []

    if kind == 'markers':
        flat_data = {i.GetLabel(): i.GetValues() for i in btk.Iterate(acq.GetPoints())}
        metadata = {'n_points': acq.GetPointNumber(), 'n_frames': acq.GetPointFrameNumber()}
        if get_metadata:
            metadata.update({
                'first_frame': acq.GetFirstFrame(),
                'last_frame': acq.GetLastFrame(),
                'point_rate': acq.GetPointFrequency()
            })
        data = np.ndarray((metadata['n_frames'], 3 * metadata['n_points']))
        for i, (key, value) in enumerate(flat_data.items()):
            data[:, i * 3: i * 3 + 3] = value
            channel_names.append(key.split(prefix)[1])
    else:
        # TODO: implements for analogs
        all_data = acq.GetAnalogs()

    data = _to_vectors(data=data,
                       kind=kind,
                       idx=idx,
                       actual_names=channel_names,
                       target_names=names)
    return (data, metadata) if get_metadata else data

    # read the file
    # get column names
    # separate and delete empty names
    # return data in pyomeca format


def _to_vectors(data, kind, idx, actual_names, target_names):
    if kind == 'markers':
        data = matrix.reshape_2d_to_3d_matrix(data)
        if not idx:
            # find names in column_names
            idx = np.argwhere(np.in1d(np.array(actual_names),
                                      np.array(target_names))).ravel()
        data = extract_markers(data, idx)
    elif kind == 'analogs':
        # TODO: implements for analogs
        pass
    else:
        raise ValueError('kind should be "markers" or "analogs"')
    return data


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
