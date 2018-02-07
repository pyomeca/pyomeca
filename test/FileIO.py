"""
Test and example script for File IO
"""

from pyomeca import data as PyoData

TEST_FILENAME_CSV = './data/markers_without_header.csv'
TEST_FILENAME_C3D = './data/markers_and_analogs.c3d'

print('FileIO')

points_1 = PyoData.load_data(TEST_FILENAME_CSV, mark_idx=[0, 1, 2, 3, 4, 5])  # all markers
print('\tcsv without header: OK')

points_2 =
