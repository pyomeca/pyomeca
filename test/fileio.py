"""
Test and example script for file IO
"""

from pathlib import Path

import numpy as np

from pyomeca import fileio as pyoio

# Path to data
DATA_FOLDER = Path('.') / 'data'
MARKERS_CSV = DATA_FOLDER / 'markers.csv'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'
ANALOGS_CSV = DATA_FOLDER / 'analogs.csv'


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
    value_condition = np.array_equal(np.round(arr[:, 0, -1], decimals=5),
                                     expected_values)
    if shape_condition and value_condition:
        print(f'{text_str}: OK')
    else:
        raise ValueError(f'{text_str}: FAILED')


print('File IO tests:')

# 1. markers in csv
print('\t1. markers in csv')
# 1.1. 11 first
m_csv_1 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
check_array(m_csv_1,
            expected_shape=(4, 11, 580),
            expected_values=[99.2596, -259.171, 903.981, 1.],
            text_str='\t\t1.1. 11 first', kind='markers')

# 1.2. mean of 1st and 4th
m_csv_2 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

check_array(m_csv_2,
            expected_shape=(4, 3, 580),
            expected_values=[99.2596, -259.171, 903.981, 1.],
            text_str='\t\t1.2. mean of 1st and 4th', kind='markers')

# 1.3. mean of first 3 markers
m_csv_3 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[[0], [1], [2]], prefix=':')
check_array(m_csv_3,
            expected_shape=(4, 1, 580),
            expected_values=[48.55783, -114.27667, 903.79767, 1.],
            text_str='\t\t1.3. mean of first 3', kind='markers')

# 1.4. with names
m_csv_4 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], prefix=':')
check_array(m_csv_4,
            expected_shape=(4, 3, 580),
            expected_values=[879.73, 177.838, 223.66, 1.],
            text_str='\t\t1.4. with names', kind='markers')

# 2. markers in c3d
print('\t2. markers in c3d')
# 2.1. 11 first
m_c3d_1 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         kind='markers', prefix=':')
check_array(m_c3d_1,
            expected_shape=(4, 11, 580),
            expected_values=[99.25964, -259.17093, 903.9809, 1.],
            text_str='\t\t2.1. 11 first', kind='markers')

# 2.2. mean of 1st and 4th
m_c3d_2 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0, 1, 2], [0, 4, 2]],
                         kind='markers', prefix=':')
check_array(m_c3d_2,
            expected_shape=(4, 3, 580),
            expected_values=[99.25964, -259.17093, 903.9809, 1.],
            text_str='\t\t2.2. mean of 1st and 4th', kind='markers')

# 2.3. mean of first 3 markers
m_c3d_3 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0], [1], [2]],
                         kind='markers', prefix=':')
check_array(m_c3d_3,
            expected_shape=(4, 1, 580),
            expected_values=[48.55782, -114.2767, 903.79779, 1.],
            text_str='\t\t2.3. mean of first 3', kind='markers')

# 2.4. with names and metadata
m_c3d_4, meta = pyoio.read_c3d(MARKERS_ANALOGS_C3D, names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'],
                               kind='markers', prefix=':', get_metadata=True)
check_array(m_c3d_4,
            expected_shape=(4, 3, 580),
            expected_values=[879.7298, 177.83847, 223.6602, 1.],
            text_str='\t\t2.4. with names and metadata', kind='markers')

# 3. analogs in csv
print('\t3. analogs in csv')
# 3.1. 11 first
a_csv_1 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
check_array(a_csv_1,
            expected_shape=(1, 11, 11600),
            expected_values=[0.],
            text_str='\t\t3.1. 11 first', kind='analogs')

# 3.2. mean of 1st and 4th
a_csv_2 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

check_array(a_csv_2,
            expected_shape=(1, 3, 11600),
            expected_values=[0.],
            text_str='\t\t3.2. mean of 1st and 4th', kind='analogs')

# 3.3. mean of first 3
a_csv_3 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[[0], [1], [2]], prefix=':')
check_array(a_csv_3,
            expected_shape=(1, 1, 11600),
            expected_values=[0.],
            text_str='\t\t3.3. mean of first 3', kind='analogs')

# 3.4. with names
a_csv_4 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         names=['EMG1', 'EMG11', 'EMG5', 'EMG13'], prefix=':')
check_array(a_csv_4,
            expected_shape=(1, 4, 11600),
            expected_values=[-2.e-05],
            text_str='\t\t3.4. with names', kind='analogs')

# 4. analogs in c3d
print('\t4. analogs in c3d')
# 4.1. 11 first
a_c3d_1 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         kind='analogs', prefix=':')
check_array(a_c3d_1,
            expected_shape=(1, 11, 11600),
            expected_values=[0.],
            text_str='\t\t4.1. 11 first', kind='analogs')

# 4.2. mean of 1st and 4th
a_c3d_2 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0, 1, 2], [0, 4, 2]],
                         kind='analogs', prefix=':')
check_array(a_c3d_2,
            expected_shape=(1, 3, 11600),
            expected_values=[0.],
            text_str='\t\t4.2. mean of 1st and 4th', kind='analogs')

# 4.3. mean of first 3
a_c3d_3 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0], [1], [2]],
                         kind='analogs', prefix=':')
check_array(a_c3d_3,
            expected_shape=(1, 1, 11600),
            expected_values=[0.],
            text_str='\t\t4.3. mean of first 3', kind='analogs')

# 4.4. with names and metadata
a_c3d_4, meta = pyoio.read_c3d(MARKERS_ANALOGS_C3D,
                               names=['Delt_ant.EMG1', 'Subscap.EMG11', 'Triceps.EMG5', 'Gd_dors.IM EMG13'],
                               kind='analogs', prefix=':', get_metadata=True)
check_array(a_c3d_4,
            expected_shape=(1, 4, 11600),
            expected_values=[-2.e-05],
            text_str='\t\t4.4. with names and metadata', kind='analogs')
