"""
Test and example script for file IO
"""

from pathlib import Path

import numpy as np

from pyomeca import fileio as pyoio

# Path to data
DATA_FOLDER = Path('.') / 'data'
markers_csv = DATA_FOLDER / 'markers.csv'
markers_analogs_c3d = DATA_FOLDER / 'markers_analogs.c3d'
analogs_csv = DATA_FOLDER / 'analogs.csv'


def check_array(arr, expected_shape, expected_values, text_str=None, kind='markers'):
    """
    Check if array respect some conditions
    Parameters
    ----------
    arr : numpy.array
        Array to check
    expected_shape : Tuple
        Expected shape
    expected_values : list
        Expected values
    kind : str
        Type of array tested
    text_str : str
        String to print to identify the current test
    """
    shape_condition = arr.shape == expected_shape
    if kind == 'markers':
        value_condition = np.array_equal(np.round(arr[:, 0, -1], decimals=5),
                                         expected_values)
    elif kind == 'analogs':
        # TODO: implement for analogs
        pass
    if shape_condition and value_condition:
        print(f'{text_str}: OK')
    else:
        raise ValueError(f'{text_str}: FAILED')


print('File IO tests:')

# 1. Load markers in csv
print('\tmarkers in csv')
# 1.1. 11 markers
m_csv_1 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                         idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
check_array(m_csv_1,
            expected_shape=(4, 11, 580),
            expected_values=[99.2596, -259.171, 903.981, 1.],
            text_str='\t\t11 markers', kind='markers')

# 1.2. Mean of 1st and 4th
m_csv_2 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                         idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

check_array(m_csv_2,
            expected_shape=(4, 3, 580),
            expected_values=[99.2596, -259.171, 903.981, 1.],
            text_str='\t\tmean of 1st and 4th', kind='markers')

# 1.3. Mean of first 3 markers
m_csv_3 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                         idx=[[0], [1], [2]], prefix=':')
check_array(m_csv_3,
            expected_shape=(4, 1, 580),
            expected_values=[48.55783, -114.27667, 903.79767, 1.],
            text_str='\t\tmean of first 3 markers', kind='markers')

# 1.4. With mark_names
m_csv_4 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                         names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], prefix=':')
check_array(m_csv_4,
            expected_shape=(4, 3, 580),
            expected_values=[879.73, 177.838, 223.66, 1.],
            text_str='\t\twith names', kind='markers')

# 2. Load markers in c3d
print('\tmarkers in c3d')
# 2.1. 11 markers
m_c3d_1 = pyoio.read_c3d(markers_analogs_c3d, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         kind='markers', prefix=':')
check_array(m_c3d_1,
            expected_shape=(4, 11, 580),
            expected_values=[99.25964, -259.17093, 903.9809, 1.],
            text_str='\t\t11 markers', kind='markers')

# 2.2. Mean of 1st and 4th
m_c3d_2 = pyoio.read_c3d(markers_analogs_c3d, idx=[[0, 1, 2], [0, 4, 2]],
                         kind='markers', prefix=':')
check_array(m_c3d_2,
            expected_shape=(4, 3, 580),
            expected_values=[99.25964, -259.17093, 903.9809, 1.],
            text_str='\t\tmean of 1st and 4th', kind='markers')

# 2.3. Mean of first 3 markers
m_c3d_3 = pyoio.read_c3d(markers_analogs_c3d, idx=[[0], [1], [2]],
                         kind='markers', prefix=':')
check_array(m_c3d_3,
            expected_shape=(4, 1, 580),
            expected_values=[48.55782, -114.2767, 903.79779, 1.],
            text_str='\t\tmean of first 3 markers', kind='markers')

# 2.4. With mark_names and metadata
m_c3d_4, meta = pyoio.read_c3d(markers_analogs_c3d, names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'],
                               kind='markers', prefix=':', get_metadata=True)
check_array(m_c3d_4,
            expected_shape=(4, 3, 580),
            expected_values=[879.7298, 177.83847, 223.6602, 1.],
            text_str='\t\twith names and metadata', kind='markers')
