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


def compare_csv(file_name, kind):
    """Assert analogs's to_csv method."""
    idx = [0, 1, 2, 3]
    out = Path('.') / 'temp.csv'

    if kind == 'markers':
        header = 2
    else:
        header = 3

    arr_ref = pyoio.read_csv(file_name, kind=kind, first_row=5, first_column=2, header=header, prefix=':',
                             idx=idx)
    arr_ref.to_csv(out, header=False)
    arr = pyoio.read_csv(out, kind=kind, first_row=0, first_column=0, header=0, prefix=':', idx=idx)

    out.unlink()

    a = arr_ref[:, :, 1:100]
    b = arr[:, :, 1:100]

    np.testing.assert_equal(a.shape, b.shape)
    np.testing.assert_almost_equal(a, b, decimal=1)


# --- Test markers data
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


@pytest.mark.parametrize('idx, names, expected_shape', markers_shapes)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_markers_shapes(idx, names, expected_shape, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = pyoio.read_csv(MARKERS_CSV, kind='markers', first_row=5, first_column=2, header=2, prefix=':',
                             idx=idx, names=names)
    elif extension == 'c3d':
        arr = pyoio.read_c3d(MARKERS_ANALOGS_C3D, kind='markers', prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    np.testing.assert_equal(arr.shape, expected_shape)


@pytest.mark.parametrize('idx, names, expected_values', markers_values)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_markers_values(idx, names, expected_values, extension):
    """Assert markers values."""
    if extension == 'csv':
        arr = pyoio.read_csv(MARKERS_CSV, kind='markers', first_row=5, first_column=2, header=2, prefix=':',
                             idx=idx, names=names)
    elif extension == 'c3d':
        arr = pyoio.read_c3d(MARKERS_ANALOGS_C3D, kind='markers', prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)


def test_markers_to_csv():
    """Assert analogs's to_csv method."""
    compare_csv(file_name=MARKERS_CSV, kind='analogs')


# --- Test analogs data

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


@pytest.mark.parametrize('idx, names, expected_shape', analogs_shapes)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_analogs_shapes(idx, names, expected_shape, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = pyoio.read_csv(ANALOGS_CSV, kind='analogs', first_row=5, first_column=2, header=3, prefix=':',
                             idx=idx, names=names)
    elif extension == 'c3d':
        arr = pyoio.read_c3d(MARKERS_ANALOGS_C3D, kind='analogs', prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    np.testing.assert_equal(arr.shape, expected_shape)


@pytest.mark.parametrize('idx, names, expected_values', analogs_values)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_analogs_values(idx, names, expected_values, extension):
    """Assert markers values."""
    if extension == 'csv':
        arr = pyoio.read_csv(ANALOGS_CSV, kind='analogs', first_row=5, first_column=2, header=3, prefix=':',
                             idx=idx, names=names)
    elif extension == 'c3d':
        arr = pyoio.read_c3d(MARKERS_ANALOGS_C3D, kind='analogs', prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)


def test_analogs_to_csv():
    """Assert analogs's to_csv method."""
    compare_csv(file_name=ANALOGS_CSV, kind='analogs')
