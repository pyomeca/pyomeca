"""
Test for file IO
"""
from pathlib import Path

import numpy as np
import pytest
from pyomeca import fileio as pyoio

# Path to data
DATA_FOLDER = Path('.') / 'data'
MARKERS_CSV = DATA_FOLDER / 'markers.csv'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'
ANALOGS_CSV = DATA_FOLDER / 'analogs.csv'

markers_shapes = [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, (4, 11, 580)),
    ([[0, 1, 2], [0, 4, 2]], None, (4, 3, 580)),
    ([[0], [1], [2]], None, (4, 1, 580)),
    (None, ['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], (4, 4, 580)),
]

markers_values = [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, [3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00]),
    ([[0, 1, 2], [0, 4, 2]], None, [3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00]),
    ([[0], [1], [2]], None, [2.62055670e+02, -2.65073300e+01, 1.04641333e+03, 1.00000000e+00]),
    (None, ['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'], [791.96, 295.588, 682.808, 1.]),
]

analogs_shapes = [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, (1, 11, 11600)),
    ([[0, 1, 2], [0, 4, 2]], None, (1, 3, 11600)),
    ([[0], [1], [2]], None, (1, 1, 11600)),
    (None, ['EMG1', 'EMG11', 'EMG5', 'EMG13'], (1, 4, 11600)),
]

analogs_values = [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, [-0.01396]),
    ([[0, 1, 2], [0, 4, 2]], None, [-0.01396]),
    ([[0], [1], [2]], None, [-0.10447]),
    (None, ['EMG1', 'EMG11', 'EMG5', 'EMG13'], [-0.00039]),
]


def read_csv(file, idx=None, names=None, kind='markers'):
    """Read a csv file."""
    if kind == 'markers':
        data = pyoio.read_csv(file, kind='markers', first_row=5, first_column=2, header=2, prefix=':',
                              idx=idx, names=names)
    elif kind == 'analogs':
        data = pyoio.read_csv(file, kind='analogs', first_row=5, first_column=2, header=3, prefix=':',
                              idx=idx, names=names)
    else:
        ValueError(f'kind should be "markers", "analogs". You provided {kind}')
        data = []
    return data


def read_c3d(file, idx=None, names=None, kind='markers'):
    """Read a c3d file."""
    if kind == 'markers':
        data = pyoio.read_c3d(file, kind='markers', prefix=':',
                              idx=idx,
                              names=names)
    elif kind == 'analogs':
        data = pyoio.read_c3d(file, kind='analogs', prefix=':',
                              idx=idx,
                              names=names)
    else:
        ValueError(f'kind should be "markers", "analogs". You provided {kind}')
        data = []
    return data


def extend_with_different_extensions(l, extensions):
    """Extend a pytest parametrize list with given extensions"""
    case = len(l)
    n_extensions = len(extensions)
    output = []
    idx = 0
    for i, irow in enumerate(l * n_extensions):
        output.append(irow + (extensions[idx],))
        if i == case - 1:
            idx += 1
    return output


@pytest.mark.parametrize('idx, names, expected_shape, extension',
                         extend_with_different_extensions(markers_shapes, extensions=['csv', 'c3d']))
def test_markers_shapes(idx, names, expected_shape, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = read_csv(file=MARKERS_CSV, idx=idx, names=names, kind='markers')
    elif extension == 'c3d':
        arr = read_c3d(file=MARKERS_ANALOGS_C3D, idx=idx, names=names, kind='markers')
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    np.testing.assert_equal(arr.shape, expected_shape)


@pytest.mark.parametrize('idx, names, expected_values, extension',
                         extend_with_different_extensions(markers_values, extensions=['csv', 'c3d']))
def test_markers_values(idx, names, expected_values, extension):
    """Assert markers values."""
    if extension == 'csv':
        arr = read_csv(file=MARKERS_CSV, idx=idx, names=names, kind='markers')
    elif extension == 'c3d':
        arr = read_c3d(file=MARKERS_ANALOGS_C3D, idx=idx, names=names, kind='markers')
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)


@pytest.mark.parametrize('idx, names, expected_shape, extension',
                         extend_with_different_extensions(analogs_shapes, extensions=['csv', 'c3d']))
def test_analogs_shapes(idx, names, expected_shape, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = read_csv(file=ANALOGS_CSV, idx=idx, names=names, kind='analogs')
    elif extension == 'c3d':
        arr = read_c3d(file=MARKERS_ANALOGS_C3D, idx=idx, names=names, kind='analogs')
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    np.testing.assert_equal(arr.shape, expected_shape)


@pytest.mark.parametrize('idx, names, expected_values, extension',
                         extend_with_different_extensions(analogs_values, extensions=['csv', 'c3d']))
def test_analogs_values(idx, names, expected_values, extension):
    """Assert markers values."""
    if extension == 'csv':
        arr = read_csv(file=ANALOGS_CSV, idx=idx, names=names, kind='analogs')
    elif extension == 'c3d':
        arr = read_c3d(file=MARKERS_ANALOGS_C3D, idx=idx, names=names, kind='analogs')
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)

# TODO: write_csv