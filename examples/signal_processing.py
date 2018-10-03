""""
Signal processing examples in pyomeca
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyomeca import Analogs3d

# Path to data
DATA_FOLDER = Path('..') / 'tests' / 'data'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'

# read an emg from a c3d file
a = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, names=['EMG1'])
a.plot()
plt.show()

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

mv_rms = a.moving_rms(window_size=WINDOW_SIZE)

_, ax = plt.subplots(nrows=1, ncols=1)

a.plot(ax=ax, fmt='k-', label='raw')
mv_rms.plot(ax=ax, fmt='r-', lw=2, label='moving rms')

ax.set_title(f'Moving RMS (window = {WINDOW_SIZE})')
ax.legend()
plt.show()

# --- Moving average
b = Analogs3d(a.moving_rms(window_size=10))

mv_mu = b.moving_average(window_size=WINDOW_SIZE)

_, ax = plt.subplots(nrows=1, ncols=1)

b.plot(ax=ax, fmt='k-', label='raw')
mv_mu.plot(ax=ax, fmt='b-', lw=2, label='moving average')

ax.set_title(f'Moving average (window = {WINDOW_SIZE})')
ax.legend()
plt.show()

# --- Moving median (sharper response to abrupt changes than the moving average)
mv_med = b.moving_median(window_size=WINDOW_SIZE - 1)

_, ax = plt.subplots(nrows=1, ncols=1)

b.plot(ax=ax, fmt='k-', label='raw')
mv_rms.plot(ax=ax, fmt='r-', lw=2, label='moving rms')
mv_mu.plot(ax=ax, fmt='g-', lw=2, label='moving average')
mv_med.plot(ax=ax, fmt='m-', lw=2, label='moving median')

ax.set_title(f'Comparison of moving methods (window = {WINDOW_SIZE - 1})')
ax.legend()
plt.show()

# --- Low-pass filter
freq = 100
t = np.arange(0, 1, .01)
w = 2 * np.pi * 1
y = np.sin(w * t) + 0.1 * np.sin(10 * w * t)
y = Analogs3d(y.reshape(1, 1, -1))

low_pass = y.low_pass(freq=freq, order=2, cutoff=5)

_, ax = plt.subplots(nrows=1, ncols=1)

y.plot(ax=ax, fmt='k-', label='raw')
low_pass.plot(ax=ax, fmt='r-', label='low-pass @ 5Hz')

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

time_normalized = a.time_normalization(time_vector=np.linspace(0, 100, 101))

# --- Amplitude normalization

amp_normalized = a.normalization()

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
amp, freqs = y.fft(freq=freq)
# compare with low-pass filtered data
amp_filtered, freqs_filtered = low_pass.fft(freq=freq)

_, ax = plt.subplots(nrows=2, ncols=1)

y.plot(ax=ax[0], fmt='k-', label='raw')
low_pass.plot(ax=ax[0], fmt='r-', alpha=.7, label='low-pass @ 5Hz')
ax[0].set_title('Temporal domain')

ax[1].plot(freqs, amp.squeeze(), 'k-', label='raw')
ax[1].plot(freqs_filtered, amp_filtered.squeeze(), 'r-', label='low-pass @ 5Hz')
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

amp_a, freqs_a = emg_without_low_pass.fft(freq=freq)
amp_a_filtered, freqs_a_filtered = emg_with_a_low_pass.fft(freq=freq)

_, ax = plt.subplots(nrows=2, ncols=1)

emg_without_low_pass.plot(ax=ax[0], fmt='k-', label='raw', alpha=.7)
emg_with_a_low_pass.plot(ax=ax[0], fmt='r-', label='low-pass @ 5Hz', alpha=.7)
ax[0].set_title('Temporal domain')

ax[1].plot(freqs_a, amp_a.squeeze(), 'k-', label='raw')
ax[1].plot(freqs_a_filtered, amp_a_filtered.squeeze(), 'r-', label='low-pass @ 5Hz')
ax[1].set_xlim(-2, 10)
ax[1].set_title('Frequency domain')
ax[1].legend()

plt.show()

# --- Norm

b = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, idx=[0, 1, 2, 3, 4, 5])

# offset on one second, then compute the norm
norm = b \
    .center(mu=np.nanmean(b[..., :int(b.get_rate)], axis=-1), axis=-1) \
    .norm()

plt.plot(norm)
plt.show()

# --- Onset detection
two_norm = np.hstack((norm, norm / 4))
two_norm = Analogs3d(two_norm.reshape(1, 1, -1))

# threshold = mean during the first second
idx = two_norm[0, 0, :].detect_onset(
    threshold=np.nanmean(two_norm[..., :int(b.get_rate)]),
    above=int(b.get_rate) / 2,
    below=3,
    threshold2=np.nanmean(two_norm[..., :int(b.get_rate)]) * 2,
    above2=5
)

_, ax = plt.subplots(nrows=1, ncols=1)
two_norm.plot(ax=ax)
for (inf, sup) in idx:
    ax.axvline(x=inf, color='r', lw=2, ls='--')
    ax.axvline(x=sup, color='r', lw=2, ls='--')
ax.set_title('Onset detection')
plt.show()
