"""
Test and example script for File IO
"""

from pyomeca import data as PyoData

TEST_FILENAME = {
    # TODO: add csv with header
    'markers_without_header': './data/markers_without_header.csv',
    'c3d': './data/markers_and_analogs.c3d'
}

print('FileIO')

# csv without header
points_1 = PyoData.load_data(TEST_FILENAME['markers_without_header'],
                             mark_idx=[0, 1, 2, 3, 4, 5])
print('\tcsv without header: OK')

# csv with header

# c3d
