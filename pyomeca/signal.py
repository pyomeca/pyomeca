""""

Signal processing in pyomeca

"""
import numpy as np
from scipy.signal import filtfilt


def rectify(x):
    """
    Rectify a signal (i.e., get absolute values)
    Parameters
    ----------
    x : np.ndarray
        1xNxF matrix of data

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
        1xNxF matrix of data
    window_size : Union[int, float]
        Window size
    method : str
        method to use. Can be 'convolution' (faster) or 'filtfilt' (works on array of any dimensions)
    Returns
    -------
    Moving average of `x` with window size `window`
    """
    if method == 'convolution':
        if x.ndim > 1:
            raise ValueError(f'moving_rms with convolution take only one dimension array')
        window = 2 * window_size + 1
        return np.sqrt(np.convolve(x * x, np.ones(window) / window, 'same'))
    elif method == 'filtfilt':
        return np.sqrt(filtfilt(np.ones(window_size) / window_size, [1], x * x))
    else:
        raise ValueError(f'method should be filtfilt or convolution. You provided {method}')

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
#

#
#
# def frame_interpolation():
#     pass
#
# def moving_rms():
#     pass
