# -*- coding: utf-8 -*-
"""

File IO in pyomeca

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
    if not header:
        skiprows = np.arange(1, first_row)
    else:
        skiprows = np.arange(header + 1, first_row)

    data = pd.read_csv(str(file_name), delimiter=delimiter, header=header, skiprows=skiprows)
    data.drop(data.columns[:first_column], axis=1, inplace=True)
    column_names = data.columns.tolist()
    if kind == 'markers' and header:
        column_names = [icol.split(prefix)[-1] for icol in column_names if (len(icol) >= 7 and icol[:7] != 'Unnamed')]
    if not names:
        names = column_names

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
        Return a dict with metadata if True

    Returns
    -------
    Data set in Vectors3d format or Data set in Vectors3d format and metadata dict if get_metadata is True
    """
    if names and idx:
        raise ValueError("names and idx can't be set simultaneously, please select only one")
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
        data = np.full([metadata['n_frames'], 3 * metadata['n_points']], np.nan)
        for i, (key, value) in enumerate(flat_data.items()):
            data[:, i * 3: i * 3 + 3] = value
            channel_names.append(key.split(prefix)[-1])
    elif kind == 'analogs':
        flat_data = {i.GetLabel(): i.GetValues() for i in btk.Iterate(acq.GetAnalogs())}
        metadata = {'n_analogs': acq.GetAnalogNumber(), 'n_frames': acq.GetAnalogFrameNumber()}
        if get_metadata:
            metadata.update({
                'first_frame': acq.GetFirstFrame(),
                'last_frame': acq.GetLastFrame(),
                'analog_rate': acq.GetAnalogFrequency()
            })
        data = np.full([metadata['n_frames'], metadata['n_analogs']], np.nan)
        for i, (key, value) in enumerate(flat_data.items()):
            data[:, i] = value.ravel()
            channel_names.append(key.split(prefix)[-1])
    if not names:
        names = channel_names

    data = _to_vectors(data=data,
                       kind=kind,
                       idx=idx,
                       actual_names=channel_names,
                       target_names=names)
    return (data, metadata) if get_metadata else data


def _to_vectors(data, kind, idx, actual_names, target_names):
    data[data == 0.0] = np.nan  # because nan are replace by 0.0 sometimes
    if not idx:
        # find names in column_names
        idx = np.argwhere(np.in1d(np.array(actual_names), np.array(target_names))).ravel()

    if kind == 'markers':
        data = matrix.reshape_2d_to_3d_matrix(data)
    elif kind == 'analogs':
        data = matrix.reshape_2d_to_3d_matrix(data, kind='analogs')
    else:
        raise ValueError('kind should be "markers" or "analogs"')
    data = extract_markers(data, idx)
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
