from pyomeca import signal as pyosignal
from pyomeca.types.analogs import Analogs3d
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

# Path to data
DATA_FOLDER = Path('..') / 'tests' / 'data'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'

# read first analogs of a c3d file
a = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, idx=[0])

# moving rms
WINDOW_SIZE = 100
a_rms_conv = pyosignal.moving_rms(a.ravel(), window_size=WINDOW_SIZE, method='convolution')
# try another filtfilt and with the method
a_rms_filt = a.moving_rms(window_size=WINDOW_SIZE)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(a.rectify().squeeze(), 'k-', label='raw')
ax.plot(a_rms_filt.squeeze(), 'r-', label='rms with filtfilt')
ax.plot(a_rms_conv.squeeze(), 'b-', label='rms with conv')

ax.legend(fontsize=12)
plt.show()
