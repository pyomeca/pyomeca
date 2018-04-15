from pyomeca import signal as pyosignal
from pyomeca.types.analogs import Analogs3d
from pathlib import Path

import matplotlib.pyplot as plt

# Path to data
DATA_FOLDER = Path('..') / 'tests' / 'data'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'

# read an emg from a c3d file
a = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, names=['EMG1'])

# --- Moving rms
WINDOW_SIZE = 100

mv_rms = {
    # standard filtfilt method
    'filt': pyosignal.moving_rms(a, window_size=WINDOW_SIZE),
    # with the Analogs3d's method
    'filt2': a.moving_rms(window_size=WINDOW_SIZE),
    # with the convolution method (works only for one dimensional array)
    'conv': pyosignal.moving_rms(a.squeeze(), window_size=WINDOW_SIZE, method='convolution'),
}

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(a.squeeze(), 'k-', label='raw')
ax.plot(mv_rms['filt'].squeeze(), 'r-', label='with filtfilt')
ax.plot(mv_rms['conv'].squeeze(), 'b-', label='with conv')
ax.set_title(f'Moving RMS (window = {WINDOW_SIZE})')
ax.legend(fontsize=12)
plt.show()

# --- Moving average
b = Analogs3d(a.moving_rms(window_size=10))

mv_mu = {
    # standard filtfilt method
    'filtfilt': pyosignal.moving_average(b, window_size=WINDOW_SIZE, method='filtfilt'),
    # with the Analogs3d's method
    'filtfilt2': b.moving_average(window_size=WINDOW_SIZE),
    'cumsum': pyosignal.moving_average(b, window_size=WINDOW_SIZE, method='cumsum'),
    'conv': pyosignal.moving_average(b.squeeze(), window_size=WINDOW_SIZE, method='convolution')
}

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(b.squeeze(), 'k-', label='raw')
ax.plot(mv_mu['cumsum'].squeeze(), 'r-', label='with cumsum')
ax.plot(mv_mu['filtfilt'].squeeze(), 'b-', label='with filtfilt')
ax.plot(mv_mu['conv'].squeeze(), 'g-', label='with conv')
ax.set_title(f'Moving average (window = {WINDOW_SIZE})')
ax.legend(fontsize=12)
plt.show()
