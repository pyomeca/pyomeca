from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyomeca import signal as pyosignal
from pyomeca.types.analogs import Analogs3d

# Path to data
DATA_FOLDER = Path('..') / 'tests' / 'data'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'

# read an emg from a c3d file
a = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, names=['EMG1'])

# --- Pyomeca types method implementation

# every function described below are implemented as method in pyomeca types and can be chained:
amp_, freqs_ = a \
    .rectify() \
    .center() \
    .moving_rms(window_size=100) \
    .moving_average(window_size=100) \
    .moving_median(window_size=100 - 1) \
    .low_pass(freq=a.get_rate, order=2, cutoff=5) \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 200]) \
    .band_stop(freq=a.get_rate, order=4, cutoff=[49.9, 50.1]) \
    .high_pass(freq=a.get_rate, order=4, cutoff=30) \
    .fft(freq=a.get_rate)

# --- Rectify and center
b = a + 2 * a.mean()
_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(b.squeeze(), 'k-', label='raw')
ax.plot(b.center().squeeze(), 'b-', label='centered', alpha=0.7)
ax.plot(b.rectify().squeeze(), 'r-', label='rectified (abs)', alpha=0.7)
ax.set_title(f'Rectify and center')
ax.legend(fontsize=12)
plt.show()

# --- Moving rms
WINDOW_SIZE = 100

mv_rms = {
    # standard filtfilt method
    'filt': pyosignal.moving_rms(a, window_size=WINDOW_SIZE),
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

# --- Moving median (sharper response to abrupt changes than the moving average)
mv_med = pyosignal.moving_median(b, window_size=WINDOW_SIZE - 1)

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(b.squeeze(), 'k-', label='raw')
ax.plot(mv_med.squeeze(), 'r-', label='moving median')
ax.plot(mv_mu['filtfilt'].squeeze(), 'b-', label='moving average')
ax.set_title(f'Moving median (window = {WINDOW_SIZE - 1})')
ax.legend(fontsize=12)
plt.show()

# --- Low-pass filter
freq = 100
t = np.arange(0, 1, .01)
w = 2 * np.pi * 1
y = np.sin(w * t) + 0.1 * np.sin(10 * w * t)

low_pass = pyosignal.low_pass(y, freq=freq, order=2, cutoff=5)

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(y, 'k-', label='raw')
ax.plot(low_pass, 'r-', label='low-pass @ 5Hz')
ax.set_title(f'Low-pass Butterworth filter')
ax.legend(fontsize=12)
plt.show()

# --- Band-pass filter
band_pass = pyosignal.band_pass(a, freq=a.get_rate, order=4, cutoff=[10, 200])

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(a.squeeze(), 'k-', label='raw')
ax.plot(band_pass.squeeze(), 'r-', label='band-pass @ 10-200Hz', alpha=0.7)
ax.set_title(f'Band-pass Butterworth filter')
ax.legend(fontsize=12)
plt.show()

# --- Band-stop filter (useful to remove the 50Hz noise for example)
band_stop = pyosignal.band_stop(a, freq=a.get_rate, order=2, cutoff=[49.9, 50.1])

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(a.squeeze(), 'k-', label='raw')
ax.plot(band_stop.squeeze(), 'r-', label='band-stop @ 49.9-50.1Hz', alpha=0.7)
ax.set_title(f'Band-stop Butterworth filter')
ax.legend(fontsize=12)
plt.show()

# --- High-pass filter
high_pass = pyosignal.high_pass(a, freq=a.get_rate, order=2, cutoff=100)

_, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(a.squeeze(), 'k-', label='raw')
ax.plot(high_pass.squeeze(), 'r-', label='high-pass @ 30Hz', alpha=0.7)
ax.set_title(f'High-pass Butterworth filter')
ax.legend(fontsize=12)
plt.show()

# --- EMG: a complete example

emg = a \
    .center() \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 425]) \
    .rectify() \
    .low_pass(freq=a.get_rate, order=4, cutoff=5) \
    .normalization()

_, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(a.squeeze(), 'k-')
ax[0].set_title('Raw data')

ax[1].plot(emg.squeeze(), 'r-')
ax[1].set_title('Processed data')
plt.show()

# --- FFT

# fft on raw data
amp, freqs = pyosignal.fft(y, freq=freq)
# compare with low-pass filtered data
amp_filtered, freqs_filtered = pyosignal.fft(low_pass, freq=freq)

_, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(y, 'k-', label='raw')
ax[0].plot(low_pass, 'r-', label='low-pass @ 5Hz')
ax[0].set_title('Raw data')

ax[1].plot(freqs, amp, 'b-', label='raw')
ax[1].plot(freqs_filtered, amp_filtered, 'r-', label='low-pass @ 5Hz')
ax[1].set_title('Frequency domain')
ax[1].legend(fontsize=12)
plt.show()

# fft on real data
emg_without_low_pass = a \
    .center() \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 425]) \
    .rectify()

amp_a, freqs_a = pyosignal.fft(emg_without_low_pass.squeeze(), freq=freq)
# compare with low-pass filtered data
amp_a_filtered, freqs_a_filtered = pyosignal.fft(emg.squeeze(), freq=a.get_rate)

_, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(emg_without_low_pass.squeeze(), 'k-', label='raw', alpha=0.7)
ax[0].plot(emg.squeeze(), 'r-', label='low-pass @ 5Hz', alpha=0.7)
ax[0].set_title('Raw data')
ax[0].legend(fontsize=12)

ax[1].plot(freqs_a, amp_a, 'k-', label='raw')
ax[1].plot(freqs_a_filtered, amp_a_filtered, 'r-', label='low-pass @ 5Hz')
ax[1].set_title('Frequency domain')
ax[1].legend(fontsize=12)
ax[1].set_xlim(-2, 40)
plt.show()
