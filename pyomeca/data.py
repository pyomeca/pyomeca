# -*- coding: utf-8 -*-
"""

File IO in PyoMeca library

"""

from thirdparty import btk
import numpy as np
import pandas as pd
from pyomeca.math import matrix
import os


def load_marker_data(file_name, mark_idx=list(), mark_names=None,
                     csv_first_row=None, csv_first_column=0, csv_row_of_mark_names=None):
    """
    Load CSV or C3D data
    Parameters
    ----------
    file_name : str
        Path of file
    mark_idx : list(int)
        Order of markers given by index,
    mark_names : list(str)
        Order of markers given by names, if both mark_names and mark_idx are provided, mark_idx prevails
    csv_first_row : int
        Index of first rows of data in the csv file, this parameter is ignored when the file is a C3D (0th indexed)
    csv_first_column : int
        Index of first column of data in the csv file, this parameter is ignored when the file is a C3D (0th indexed)
    csv_row_of_mark_names : int
        row of the marker names in the csv file (0th indexed)

    Returns
    -------
    Data set in Vectors3d format
    """

    # if mark_names is used, find the proper indexes
    if mark_names and mark_idx:
        raise ValueError("mark_names and mark_idx can't be set simultanously, please select only one")

    def read_from_csv():
        """
        Read a CSV file using panda
        Returns
        -------
        2d matrix of markers, all marker names
        """
        # Read the markers header if needed
        if mark_names:
            if not csv_row_of_mark_names:
                raise ValueError("csv_header_mark_names_row must be provided when mark_names is used with a csv file")

            with open(file_name, 'r') as f:
                for _ in range(csv_row_of_mark_names+1):
                    header_mark_names = f.readline()

            # Separate into marker names
            all_mark_names = list()
            for t in header_mark_names.split(','):
                if not t:  # if between 2 ","
                    continue
                all_mark_names.append(t)
        else:
            all_mark_names = []

        # Read the file
        data = pd.read_csv(file_name, header=csv_first_row)
        data = data.values[:, csv_first_column:]

        return data, all_mark_names

    def read_from_c3d():
        """
        Read a C3D file using btk
        Returns
        -------
        2d matrix of markers, all marker names
        """
        # Open the file
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(file_name)
        reader.Update()
        acq = reader.GetOutput()

        # Get the model name and extract all markers names (by removing the model name)
        all_markers = acq.GetPoints()
        all_mark_names = list()
        for marker in btk.Iterate(all_markers):
            # Store the name removing the first part which is the name of the model
            all_mark_names.append(marker.GetLabel())

        # Put all data in a single matrix in the order of all_mark_names
        nbre_mark = len(all_mark_names)
        nbre_frames = acq.GetLastFrame() - acq.GetFirstFrame() + 1
        d = np.ndarray((nbre_frames, 3 * nbre_mark))
        for i, m in enumerate(all_mark_names):
            d[:, i * 3:i * 3 + 3] = acq.GetPoint(m).GetValues()

        return d, all_mark_names

    # Get the file extension and call the right opening function
    _, file_extension = os.path.splitext(file_name)
    if file_extension.lower() == ".c3d":
        data_frame, all_mark_names = read_from_c3d()
    elif file_extension.lower() == ".csv":
        data_frame, all_mark_names = read_from_csv()
    else:
        raise NotImplementedError("Only .c3d or .csv files can be read")
    data_set = matrix.reshape_2d_to_3d_matrix(data_frame)

    if mark_names:
        # Get indexes from names
        for i, m in enumerate(mark_names):
            mark_idx.append([i for i, s in enumerate(all_mark_names) if m in s][0])
        data_set = extract_data(data_set, mark_idx)
    elif mark_idx:
        data_set = extract_data(data_set, mark_idx)

    return data_set


def extract_data(m, mark_idx):
    """

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
        raise IndexError('extract_data works only on 3xNxF matrices and mark_idx must be a ixj array')
    return data
