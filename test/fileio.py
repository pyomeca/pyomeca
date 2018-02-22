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


def check_array(arr, expected_shape, expected_values, text=None):
    np.testing.assert_equal(arr.shape, expected_shape,
                            err_msg=f'{text} [shape]: FAILED')
    print(f'{text} [shape]: OK')

    np.testing.assert_almost_equal(arr[:, 0, int(arr.shape[2] / 2)], expected_values,
                                   decimal=5,
                                   err_msg=f'{text} [value]: FAILED')
    print(f'{text} [value]: OK')


def compare_arrays(arr1, arr2, text):
    np.testing.assert_allclose(arr1[:-1], arr2[:-1],
                               atol=1e-2,
                               equal_nan=True,
                               err_msg=f'{text}: FAILED')
    print(f'{text}: OK')


print('File IO tests:')

# 1. markers in csv
print('\t1. markers in csv')
# 1.1. 11 first
m_csv_1 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
check_array(m_csv_1,
            expected_shape=(4, 11, 580),
            expected_values=[3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00],
            text='\t\t1.1. 11 first')

# 1.2. mean of 1st and 4th
m_csv_2 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

check_array(m_csv_2,
            expected_shape=(4, 3, 580),
            expected_values=[3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00],
            text='\t\t1.2. mean of 1st and 4th')

# 1.3. mean of first 3 markers
m_csv_3 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         idx=[[0], [1], [2]], prefix=':')
check_array(m_csv_3,
            expected_shape=(4, 1, 580),
            expected_values=[2.62055670e+02, -2.65073300e+01, 1.04641333e+03, 1.00000000e+00],
            text='\t\t1.3. mean of first 3')

# 1.4. with names
m_csv_4 = pyoio.read_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                         names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], prefix=':')
check_array(m_csv_4,
            expected_shape=(4, 4, 580),
            expected_values=[791.96, 295.588, 682.808, 1.],
            text='\t\t1.4. with names')

# 2. markers in c3d
print('\t2. markers in c3d')
# 2.1. 11 first
m_c3d_1 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         kind='markers', prefix=':')
check_array(m_c3d_1,
            expected_shape=(4, 11, 580),
            expected_values=[3.18461360e+02, -1.69002700e+02, 1.05422009e+03, 1.00000000e+00],
            text='\t\t2.1. 11 first')

# 2.2. mean of 1st and 4th
m_c3d_2 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0, 1, 2], [0, 4, 2]],
                         kind='markers', prefix=':')
check_array(m_c3d_2,
            expected_shape=(4, 3, 580),
            expected_values=[3.18461360e+02, -1.69002700e+02, 1.05422009e+03, 1.00000000e+00],
            text='\t\t2.2. mean of 1st and 4th')

# 2.3. mean of first 3 markers
m_c3d_3 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0], [1], [2]],
                         kind='markers', prefix=':')
check_array(m_c3d_3,
            expected_shape=(4, 1, 580),
            expected_values=[2.62055570e+02, -2.65075200e+01, 1.04641496e+03, 1.00000000e+00],
            text='\t\t2.3. mean of first 3')

# 2.4. with names and metadata
m_c3d_4, meta = pyoio.read_c3d(MARKERS_ANALOGS_C3D, names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'],
                               kind='markers', prefix=':', get_metadata=True)
check_array(m_c3d_4,
            expected_shape=(4, 4, 580),
            expected_values=[791.96002197, 295.58773804, 682.80767822, 1.],
            text='\t\t2.4. with names and metadata')

# 3. analogs in csv
print('\t3. analogs in csv')
# 3.1. 11 first
a_csv_1 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
check_array(a_csv_1,
            expected_shape=(1, 11, 11600),
            expected_values=[-0.01396],
            text='\t\t3.1. 11 first')

# 3.2. mean of 1st and 4th
a_csv_2 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

check_array(a_csv_2,
            expected_shape=(1, 3, 11600),
            expected_values=[-0.01396],
            text='\t\t3.2. mean of 1st and 4th')

# 3.3. mean of first 3
a_csv_3 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         idx=[[0], [1], [2]], prefix=':')
check_array(a_csv_3,
            expected_shape=(1, 1, 11600),
            expected_values=[-0.10447],
            text='\t\t3.3. mean of first 3')

# 3.4. with names
a_csv_4 = pyoio.read_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, kind='analogs',
                         names=['EMG1', 'EMG11', 'EMG5', 'EMG13'], prefix=':')
check_array(a_csv_4,
            expected_shape=(1, 4, 11600),
            expected_values=[-0.00039],
            text='\t\t3.4. with names')

# 4. analogs in c3d
print('\t4. analogs in c3d')
# 4.1. 11 first
a_c3d_1 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         kind='analogs', prefix=':')
check_array(a_c3d_1,
            expected_shape=(1, 11, 11600),
            expected_values=[-0.01396],
            text='\t\t4.1. 11 first')

# 4.2. mean of 1st and 4th
a_c3d_2 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0, 1, 2], [0, 4, 2]],
                         kind='analogs', prefix=':')
check_array(a_c3d_2,
            expected_shape=(1, 3, 11600),
            expected_values=[-0.01396],
            text='\t\t4.2. mean of 1st and 4th')

# 4.3. mean of first 3
a_c3d_3 = pyoio.read_c3d(MARKERS_ANALOGS_C3D, idx=[[0], [1], [2]],
                         kind='analogs', prefix=':')
check_array(a_c3d_3,
            expected_shape=(1, 1, 11600),
            expected_values=[-0.10447],
            text='\t\t4.3. mean of first 3')

# 4.4. with names and metadata
a_c3d_4, _ = pyoio.read_c3d(MARKERS_ANALOGS_C3D,
                            names=['Delt_ant.EMG1', 'Subscap.EMG11', 'Triceps.EMG5', 'Gd_dors.IM EMG13'],
                            kind='analogs', prefix=':', get_metadata=True)
check_array(a_c3d_4,
            expected_shape=(1, 4, 11600),
            expected_values=[-0.00039],
            text='\t\t4.4. with names and metadata')

# 5. constancy c3d and csv markers
print('\t5. compare c3d and csv markers')
# 5.1. 11 first
compare_arrays(m_csv_1, m_c3d_1, text='\t\t11 first')
# 5.2. mean of 1st and 4th
compare_arrays(m_csv_2, m_c3d_2, text='\t\tmean of 1st and 4th')
# 5.3. mean of first 3
compare_arrays(m_csv_3, m_c3d_3, text='\t\tmean of first 3')
# 5.4. with names and metadata
compare_arrays(m_csv_4, m_c3d_4, text='\t\twith names')

# 6. constancy c3d and csv analogs
print('\t6. compare c3d and csv analogs')
# 6.1. 11 first
compare_arrays(a_csv_1, a_c3d_1, text='\t\t11 first')
# 6.2. mean of 1st and 4th
compare_arrays(a_csv_2, a_c3d_2, text='\t\tmean of 1st and 4th')
# 6.3. mean of first 3
compare_arrays(a_csv_3, a_c3d_3, text='\t\tmean of first 3')
# 6.4. with names and metadata
compare_arrays(a_csv_4, a_c3d_4, text='\t\twith names')
