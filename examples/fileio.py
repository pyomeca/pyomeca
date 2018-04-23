"""
Example script for file IO
"""

from pathlib import Path

import numpy as np
from pyomeca.types.analogs import Analogs3d
from pyomeca.types.markers import Markers3d

# Path to data
DATA_FOLDER = Path('..') / 'tests' / 'data'
MARKERS_CSV = DATA_FOLDER / 'markers.csv'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'
ANALOGS_CSV = DATA_FOLDER / 'analogs.csv'

# read 11 first markers of a csv file
markers_1 = Markers3d.from_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                               idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')

# mean of 1st and 4th markers of a csv file
markers_2 = Markers3d.from_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                               idx=[[0, 1, 2], [0, 4, 2]], prefix=':')

# get markers by names in a csv file
markers_3 = Markers3d.from_csv(MARKERS_CSV, first_row=5, first_column=2, header=2,
                               names=['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], prefix=':')

# write a csv file from a Markers3d types
markers_3.to_csv('../Misc/mtest.csv', header=False)

# read 4 first markers of a c3d file
markers_4 = Markers3d.from_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3])

# get 5 first analogs of a csv file
analogs_1 = Analogs3d.from_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3,
                               idx=[[0, 1, 2], [0, 4, 2]])

# get analogs by names in a c3d file
analogs_2 = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, prefix=':',
                               names=['Delt_ant.EMG1', 'Subscap.EMG11', 'Triceps.EMG5', 'Gd_dors.IM EMG13'])

# write analogs to a csv file without header
analogs_2.to_csv('../Misc/atest.csv', header=True)
