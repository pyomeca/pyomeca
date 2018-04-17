""""
Signal processing in pyomeca
"""

import numpy as np
from scipy import fftpack
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import filtfilt, medfilt, butter


def rectify(x):
    """
    Rectify a signal (i.e., get absolute values)

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data

    Returns
    -------
    Rectified x
    """
    return np.abs(x)


def center(x, axis=-1):
    """
    Center a signal (i.e., subtract the mean)

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    axis : int, optional
        axis along which the means are computed. The default is to compute
        the mean on the last axis.
    Returns
    -------
    Centered x
    """
    mu = np.nanmean(x, axis=axis)
    if x.ndim > mu.ndim:
        # add one dimension if the input is a 3d matrix
        mu = np.expand_dims(mu, axis=-1)
    return x - mu


def normalization(x, ref=None, scale=100):
    """
    Normalize a signal against `ref` (x's max if empty) on a scale of `scale`

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    ref : Union(int, float)
        reference value
    scale
        Scale on which to express x (100 by default)

    Returns
    -------
    x normalized
    """
    if not ref:
        ref = np.nanmax(x, axis=-1)
        ref = np.expand_dims(ref, axis=-1)
    return x / (ref / scale)


def time_normalization(x, time_vector=np.linspace(0, 100, 101), axis=-1):
    """
    Time normalization used for temporal alignment of data

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    time_vector : np.ndarray
        desired time vector (0 to 100 by step of 1 by default)
    axis : int
        specifies the axis along which to interpolate. Interpolation defaults to the last axis (over frames)

    Returns
    -------
    Time normalized x
    """
    original_time_vector = np.linspace(time_vector[0], time_vector[-1], x.shape[axis])
    f = interp1d(original_time_vector, x, axis=axis)
    return f(time_vector)


def fill_values(x, axis=-1):
    """
    Fill values. Warning: this function can be used only for very small gaps in your data.

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    axis : int
        specifies the axis along which to interpolate. Interpolation defaults to the last axis (over frames)

    Returns
    -------
    Filled x
    """
    original_time_vector = np.arange(0, x.shape[axis])
    x = x.copy()

    def fct(m):
        w = np.isnan(m)
        m[w] = 0
        f = UnivariateSpline(original_time_vector, m, w=~w)
        return f(original_time_vector)

    return np.apply_along_axis(fct, axis=axis, arr=x)


def moving_rms(x, window_size, method='filtfilt'):
    """
    Moving root mean square

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    window_size : Union(int, float)
        Window size
    method : str
        method to use:
            - 'convolution': faster and behaves better to abrupt changes, but works only for one dimensional array.
            - 'filtfilt': the go-to solution.

    Returns
    -------
    Moving root mean square of `x` with window size `window_size`
    """
    if method == 'convolution':
        if x.ndim > 1:
            raise ValueError(f'moving_rms with convolution take only one dimension array')
        window = 2 * window_size + 1
        return np.sqrt(np.convolve(x * x, np.ones(window) / window, 'same'))
    elif method == 'filtfilt':
        return np.sqrt(filtfilt(np.ones(window_size) / window_size, 1, x * x))
    else:
        raise ValueError(f'method should be filtfilt or convolution. You provided {method}')


def moving_average(x, window_size, method='filtfilt'):
    """
    Moving average

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    window_size : Union(int, float)
        Window size
    method : str
        method to use:
            - 'cumsum': fastest method.
            - 'convolution': produces a result without a lag between the input and the output.
            - 'filtfilt': The go-to method.

    Returns
    -------
    Moving average of `x` with window size `window_size`
    """
    if method == 'cumsum':
        if x.ndim > 2:
            raise ValueError(f'moving_average with cumsum take only one or two dimensions array')
        xsum = np.cumsum(x)
        xsum[window_size:] = xsum[window_size:] - xsum[:-window_size]
        return xsum[window_size - 1:] / window_size
    elif method == 'convolution':
        if x.ndim > 1:
            raise ValueError(f'moving_average with convolution take only one dimension array')
        return np.convolve(x, np.ones(window_size) / window_size, 'same')
    elif method == 'filtfilt':
        return filtfilt(np.ones(window_size) / window_size, 1, x)
    else:
        raise ValueError(f'method should be filtfilt, cumsum or convolution. You provided {method}')


def moving_median(x, window_size):
    """
    Moving median (has a sharper response to abrupt changes than the moving average)

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    window_size : Union(int, float)
        Window size (use around [3, 11])

    Returns
    -------
    Moving average of `x` with window size `window_size`
    """
    if window_size % 2 == 0:
        raise ValueError(f'window_size should be odd. Add or substract 1. You provided {window_size}')
    if x.ndim == 3:
        window_size = [1, 1, window_size]
    elif x.ndim == 2:
        window_size = [1, window_size]
    elif x.ndim == 1:
        pass
    else:
        raise ValueError(f'x.dim should be 1, 2 or 3. You provided an array with {x.ndim} dimensions.')
    return medfilt(x, window_size)


def low_pass(x, freq, order, cutoff):
    """
    Low-pass Butterworth filter

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    freq : Union(int, float)
        Sample frequency
    order : Int
        Order of the filter
    cutoff : Int
        Cut-off frequency

    Returns
    -------
    Filtered `x`
    """
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype='low')
    return filtfilt(b, a, x)


def band_pass(x, freq, order, cutoff):
    """
    Band-pass Butterworth filter

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    freq : Union(int, float)
        Sample frequency
    order : Int
        Order of the filter
    cutoff : List-like
        Cut-off frequencies ([lower, upper])

    Returns
    -------
    Filtered `x`
    """
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype='bandpass')
    return filtfilt(b, a, x)


def band_stop(x, freq, order, cutoff):
    """
    Band-stop Butterworth filter

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    freq : Union(int, float)
        Sample frequency
    order : Int
        Order of the filter
    cutoff : List-like
        Cut-off frequencies ([lower, upper])

    Returns
    -------
    Filtered `x`
    """
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype='bandstop')
    return filtfilt(b, a, x)


def high_pass(x, freq, order, cutoff):
    """
    Band-stop Butterworth filter

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    freq : Union(int, float)
        Sample frequency
    order : Int
        Order of the filter
    cutoff : List-like
        Cut-off frequencies ([lower, upper])

    Returns
    -------
    Filtered `x`
    """
    nyquist = freq / 2
    corrected_freq = np.array(cutoff) / nyquist
    b, a = butter(N=order, Wn=corrected_freq, btype='high')
    return filtfilt(b, a, x)


def fft(x, freq, only_positive=True, axis=-1):
    """
    Performs a discrete Fourier Transform and return amplitudes and frequencies

    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    freq : Union(int, float)
        Sample frequency
    only_positive : bool
        Returns only the positives frequencies if true (True by default)
    axis : int
        specifies the axis along which to performs the FFT. Performs defaults to the last axis (over frames)

    Returns
    -------

    """
    n = x.shape[axis]
    yfft = fftpack.fft(x, n)
    freqs = fftpack.fftfreq(n, 1. / freq)

    if only_positive:
        amp = 2 * np.abs(yfft) / n
        amp = amp[:int(np.floor(n / 2))]
        freqs = freqs[:int(np.floor(n / 2))]
    else:
        amp = np.abs(yfft) / n
    return amp, freqs

# todo:
# residual_analysis (bmc)
# ensemble_average (bmc)
