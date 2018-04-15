# -*- coding: utf-8 -*-
"""

File IO in pyomeca

"""

import numpy as np
import pandas as pd

from pyomeca.thirdparty import btk
from pyomeca.types import Markers3d, Analogs3d


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
    metadata = {'get_first_frame': [], 'get_last_frame': [], 'get_rate': [], 'get_labels': [], 'get_unit': []}
    if names:
        metadata.update({'get_labels': names})
    else:
        names = column_names

    return _to_vectors(data=data.values,
                       kind=kind,
                       idx=idx,
                       all_names=column_names,
                       target_names=names,
                       metadata=metadata)


def read_c3d(file_name, idx=None, names=None, kind='markers', prefix=None):
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
        metadata = {
            'get_num_markers': acq.GetPointNumber(),
            'get_num_frames': acq.GetPointFrameNumber(),
            'get_first_frame': acq.GetFirstFrame(),
            'get_last_frame': acq.GetLastFrame(),
            'get_rate': acq.GetPointFrequency(),
            'get_unit': acq.GetPointUnit()
        }
        data = np.full([metadata['get_num_frames'], 3 * metadata['get_num_markers']], np.nan)
        for i, (key, value) in enumerate(flat_data.items()):
            data[:, i * 3: i * 3 + 3] = value
            channel_names.append(key.split(prefix)[-1])
    elif kind == 'analogs':
        flat_data = {i.GetLabel(): i.GetValues() for i in btk.Iterate(acq.GetAnalogs())}
        metadata = {
            'get_num_analogs': acq.GetAnalogNumber(),
            'get_num_frames': acq.GetAnalogFrameNumber(),
            'get_first_frame': acq.GetFirstFrame(),
            'get_last_frame': acq.GetLastFrame(),
            'get_rate': acq.GetAnalogFrequency(),
            'get_unit': []
        }
        data = np.full([metadata['get_num_frames'], metadata['get_num_analogs']], np.nan)
        for i, (key, value) in enumerate(flat_data.items()):
            data[:, i] = value.ravel()
            channel_names.append(key.split(prefix)[-1])
    if names:
        metadata.update({'get_labels': names})
    else:
        metadata.update({'get_labels': []})
        names = channel_names

    return _to_vectors(data=data,
                       kind=kind,
                       idx=idx,
                       all_names=channel_names,
                       target_names=names,
                       metadata=metadata)


def _to_vectors(data, kind, idx, all_names, target_names, metadata=None):
    data[data == 0.0] = np.nan  # because nan are replace by 0.0 sometimes
    if not idx:
        # find names in column_names
        idx = []
        for i, m in enumerate(target_names):
            idx.append([i for i, s in enumerate(all_names) if m in s][0])
    if kind == 'markers':
        data = Markers3d(data)
    elif kind == 'analogs':
        data = Analogs3d(data)
    else:
        raise ValueError('kind should be "markers" or "analogs"')
    data = data.get_specific_data(idx)

    data.get_first_frame = metadata['get_first_frame']
    data.get_last_frame = metadata['get_last_frame']
    data.get_rate = metadata['get_rate']
    data.get_unit = metadata['get_unit']
    if np.array(idx).ndim == 1 and not metadata['get_labels']:
        data.get_labels = [name for i, name in enumerate(all_names) if i in idx]
    elif metadata['get_labels']:
        data.get_labels = metadata['get_labels']
    return data
