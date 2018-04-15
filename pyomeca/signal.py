""""

Signal processing in pyomeca

"""
import numpy as np
from scipy.signal import filtfilt, lfilter


def rectify(x):
    """
    Rectify a signal (i.e., get absolute values)
    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    Returns
    -------
    np.array
    """
    return np.abs(x)


def moving_rms(x, window_size, method='filtfilt'):
    """
    Moving root mean square
    Parameters
    ----------
    x : np.ndarray
        vector or matrix of data
    window_size : Union[int, float]
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
    window_size : Union[int, float]
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
#
# def high_pass():
#     pass
#
#
# def band_pass():
#     pass
#
#
# def band_stop():
#     pass
#
# def frame_interpolation():
#     pass
#
