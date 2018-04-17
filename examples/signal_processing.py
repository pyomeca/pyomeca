""""
Signal processing examples in pyomeca
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyomeca import plot as pyoplot
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
    .time_normalization() \
    .normalization() \
    .fft(freq=a.get_rate)

# --- Rectify and center
b = a + 2 * a.mean()

_, ax = plt.subplots(nrows=1, ncols=1)
b.plot(ax=ax, label='raw')
b.center().plot(ax=ax, fmt='b-', alpha=.7, label='centered')
b.rectify().plot(ax=ax, fmt='g-', alpha=.7, label='rectified (abs)')
ax.legend()

ax.set_title('Rectify and center')
ax.legend()
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

a.plot(ax=ax, fmt='k-', label='raw')
pyoplot.plot_vector3d(mv_rms['filt'], ax=ax, fmt='r-', lw=2, label='with filtfilt')
pyoplot.plot_vector3d(mv_rms['conv'], ax=ax, fmt='b-', lw=2, label='with convolution')

ax.set_title(f'Moving RMS (window = {WINDOW_SIZE})')
ax.legend()
plt.show()

# --- Moving average
b = Analogs3d(a.moving_rms(window_size=10))

mv_mu = {
    # standard filtfilt method
    'filtfilt': pyosignal.moving_average(b, window_size=WINDOW_SIZE, method='filtfilt'),
    'cumsum': pyosignal.moving_average(b.squeeze(), window_size=WINDOW_SIZE, method='cumsum'),
    'conv': pyosignal.moving_average(b.squeeze(), window_size=WINDOW_SIZE, method='convolution')
}

_, ax = plt.subplots(nrows=1, ncols=1)

b.plot(ax=ax, fmt='k-', label='raw')
pyoplot.plot_vector3d(mv_mu['cumsum'], ax=ax, fmt='r-', lw=2, label='with cumsum')
pyoplot.plot_vector3d(mv_mu['filtfilt'], ax=ax, fmt='b-', lw=2, label='with filtfilt')
pyoplot.plot_vector3d(mv_mu['conv'], ax=ax, fmt='g-', lw=2, label='with convolution')

ax.set_title(f'Moving average (window = {WINDOW_SIZE})')
ax.legend()
plt.show()

# --- Moving median (sharper response to abrupt changes than the moving average)
mv_med = pyosignal.moving_median(b, window_size=WINDOW_SIZE - 1)

_, ax = plt.subplots(nrows=1, ncols=1)

b.plot(ax=ax, fmt='k-', label='raw')
pyoplot.plot_vector3d(mv_med, ax=ax, fmt='r-', lw=2, label='moving median')
pyoplot.plot_vector3d(mv_mu['filtfilt'], ax=ax, fmt='b-', lw=2, label='moving average')

ax.set_title(f'Moving median (window = {WINDOW_SIZE - 1})')
ax.legend()
plt.show()

# --- Low-pass filter
freq = 100
t = np.arange(0, 1, .01)
w = 2 * np.pi * 1
y = np.sin(w * t) + 0.1 * np.sin(10 * w * t)

low_pass = pyosignal.low_pass(y, freq=freq, order=2, cutoff=5)

_, ax = plt.subplots(nrows=1, ncols=1)

pyoplot.plot_vector3d(y, ax=ax, fmt='k-', label='raw')
pyoplot.plot_vector3d(low_pass, ax=ax, fmt='r-', label='low-pass @ 5Hz')

ax.set_title('Low-pass Butterworth filter')
ax.legend()
plt.show()

# --- Band-pass filter
band_pass = a.band_pass(freq=a.get_rate, order=4, cutoff=[10, 200])

_, ax = plt.subplots(nrows=1, ncols=1)

a.plot(ax=ax, fmt='k-', label='raw')
band_pass.plot(ax=ax, fmt='r-', alpha=.7, label='band-pass @ 10-200Hz')

ax.set_title('Band-pass Butterworth filter')
ax.legend()
plt.show()

# --- Band-stop filter (useful to remove the 50Hz noise for example)
band_stop = a.band_stop(freq=a.get_rate, order=2, cutoff=[49.9, 50.1])

_, ax = plt.subplots(nrows=1, ncols=1)

a.plot(ax=ax, fmt='k-', label='raw')
band_stop.plot(ax=ax, fmt='r-', alpha=.7, label='band-stop @ 49.9-50.1Hz')

ax.set_title('Band-stop Butterworth filter')
ax.legend()
plt.show()

# --- High-pass filter
high_pass = a.high_pass(freq=a.get_rate, order=2, cutoff=100)

_, ax = plt.subplots(nrows=1, ncols=1)

a.plot(ax=ax, fmt='k-', label='raw')
high_pass.plot(ax=ax, fmt='r-', alpha=.7, label='high-pass @ 30Hz')

ax.set_title('High-pass Butterworth filter')
ax.legend()
plt.show()

# --- Time normalization

time_normalized = pyosignal.time_normalization(a, time_vector=np.linspace(0, 100, 101))

# --- Amplitude normalization

amp_normalized = pyosignal.normalization(a)

# --- EMG: a complete example

emg = a \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 425]) \
    .center() \
    .rectify() \
    .low_pass(freq=a.get_rate, order=4, cutoff=5) \
    .normalization() \
    .time_normalization()

_, ax = plt.subplots(nrows=2, ncols=1)

a.plot(ax=ax[0], fmt='k-')
ax[0].set_title('Raw data')

emg.plot(ax=ax[1], fmt='r-')
ax[1].set_title('Processed data')

plt.show()

# --- FFT

# fft on raw data
amp, freqs = pyosignal.fft(y, freq=freq)
# compare with low-pass filtered data
amp_filtered, freqs_filtered = pyosignal.fft(low_pass, freq=freq)

_, ax = plt.subplots(nrows=2, ncols=1)

pyoplot.plot_vector3d(y, ax=ax[0], fmt='k-', label='raw')
pyoplot.plot_vector3d(low_pass, ax=ax[0], fmt='r-', alpha=.7, label='low-pass @ 5Hz')
ax[0].set_title('Temporal domain')

pyoplot.plot_vector3d(x=freqs, y=amp, ax=ax[1], fmt='k-', label='raw')
pyoplot.plot_vector3d(x=freqs_filtered, y=amp_filtered, ax=ax[1], fmt='r-', label='low-pass @ 5Hz')
ax[1].set_title('Frequency domain')

ax[1].legend()
plt.show()

# fft on real data
emg_without_low_pass = a \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 425]) \
    .center() \
    .rectify()

emg_with_a_low_pass = a \
    .band_pass(freq=a.get_rate, order=4, cutoff=[10, 425]) \
    .center() \
    .rectify() \
    .low_pass(freq=a.get_rate, order=4, cutoff=5)

amp_a, freqs_a = pyosignal.fft(emg_without_low_pass.squeeze(), freq=freq)

# compare with low-pass filtered data
amp_a_filtered, freqs_a_filtered = pyosignal.fft(emg_with_a_low_pass.squeeze(), freq=a.get_rate)

_, ax = plt.subplots(nrows=2, ncols=1)

emg_without_low_pass.plot(ax=ax[0], fmt='k-', label='raw', alpha=.7)
emg_with_a_low_pass.plot(ax=ax[0], fmt='r-', label='low-pass @ 5Hz', alpha=.7)
ax[0].set_title('Temporal domain')

pyoplot.plot_vector3d(x=freqs_a, y=amp_a, ax=ax[1], fmt='k-', label='raw')
pyoplot.plot_vector3d(x=freqs_a_filtered, y=amp_a_filtered, ax=ax[1], fmt='r-', label='low-pass @ 5Hz')
ax[1].set_xlim(-2, 10)
ax[1].set_title('Frequency domain')
ax[1].legend()

plt.show()

ax[0].plot(emg_without_low_pass.squeeze(), 'k-', label='raw', alpha=0.7)
ax[0].plot(emg_with_a_low_pass.squeeze(), 'r-', label='low-pass @ 5Hz', alpha=0.7)
ax[0].set_title('Raw data')
ax[0].legend(fontsize=12)

ax[1].plot(freqs_a, amp_a, 'k-', label='raw')
ax[1].plot(freqs_a_filtered, amp_a_filtered, 'r-', label='low-pass @ 5Hz')
ax[1].set_title('Frequency domain')
ax[1].legend(fontsize=12)

plt.show()
