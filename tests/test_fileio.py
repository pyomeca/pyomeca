"""
Test for file IO
"""
from pathlib import Path

import numpy as np
import pytest

from pyomeca.types.markers import Markers3d
from pyomeca.types.analogs import Analogs3d

# Path
if Path.cwd().parts[-1] == 'pyomeca':
    PROJECT_FOLDER = Path('.')
else:
    # if launched from the terminal
    PROJECT_FOLDER = Path('..')
DATA_FOLDER = PROJECT_FOLDER / 'tests' / 'data'

MARKERS_CSV = DATA_FOLDER / 'markers.csv'
MARKERS_ANALOGS_C3D = DATA_FOLDER / 'markers_analogs.c3d'
ANALOGS_CSV = DATA_FOLDER / 'analogs.csv'


def compare_csv(filename, kind):
    """Assert analogs's to_csv method."""
    idx = [0, 1, 2, 3]
    out = Path('.') / 'temp.csv'

    if kind == 'markers':
        header = 2
        arr_ref, arr = Markers3d(), Markers3d()
    else:
        header = 3
        arr_ref, arr = Analogs3d(), Analogs3d()
    # read a csv
    arr_ref = arr_ref.from_csv(filename, first_row=5, first_column=2, header=header, prefix=':', idx=idx)
    # write a csv
    arr_ref.to_csv(out, header=False)
    # read the generated csv
    arr = arr.from_csv(out, first_row=0, first_column=0, header=0, prefix=':', idx=idx)

    out.unlink()

    a = arr_ref[:, :, 1:100]
    b = arr[:, :, 1:100]

    np.testing.assert_equal(a.shape, b.shape)
    np.testing.assert_almost_equal(a, b, decimal=1)


# --- Test markers data
idx = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [[0, 1, 2], [0, 4, 2]], [[0], [1], [2]], None)
names = (None, None, None, ['CLAV_post', 'PSISl', 'STERr', 'CLAV_post'])
expected_shape = ((4, 11, 580), (4, 3, 580), (4, 1, 580), (4, 4, 580))
expected_values = (
    [3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00],
    [3.18461e+02, -1.69003e+02, 1.05422e+03, 1.00000e+00],
    [2.62055670e+02, -2.65073300e+01, 1.04641333e+03, 1.00000000e+00],
    [791.96, 295.588, 682.808, 1.]
)
markers_param = [(idx[i], names[i], expected_shape[i], expected_values[i]) for i in range(len(idx))]


@pytest.mark.parametrize('idx, names, expected_shape, expected_values', markers_param)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_markers(idx, names, expected_shape, expected_values, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = Markers3d.from_csv(MARKERS_CSV, first_row=5, first_column=2, header=2, prefix=':',
                                 idx=idx, names=names)
    elif extension == 'c3d':
        arr = Markers3d.from_c3d(MARKERS_ANALOGS_C3D, prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    # test shape
    np.testing.assert_equal(arr.shape, expected_shape)
    # test values
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)


def test_markers_to_csv():
    """Assert analogs's to_csv method."""
    compare_csv(filename=MARKERS_CSV, kind='analogs')


# --- Test analogs data
names = (None, None, None, ['EMG1', 'EMG11', 'EMG5', 'EMG13'])
expected_shape = ((1, 11, 11600), (1, 3, 11600), (1, 1, 11600), (1, 4, 11600))
expected_values = ([-0.01396], [-0.01396], [-0.10447], [-0.00039])
analogs_param = [(idx[i], names[i], expected_shape[i], expected_values[i]) for i in range(len(idx))]


@pytest.mark.parametrize('idx, names, expected_shape, expected_values', analogs_param)
@pytest.mark.parametrize('extension', ['c3d', 'csv'])
def test_analogs_shapes(idx, names, expected_shape, expected_values, extension):
    """Assert markers shape."""
    if extension == 'csv':
        arr = Analogs3d.from_csv(ANALOGS_CSV, first_row=5, first_column=2, header=3, prefix=':',
                                 idx=idx, names=names)
    elif extension == 'c3d':
        arr = Analogs3d.from_c3d(MARKERS_ANALOGS_C3D, prefix=':', idx=idx, names=names)
    else:
        raise ValueError(f'extension should be "csv", "c3d". You provided {extension}')
    # test shape
    np.testing.assert_equal(arr.shape, expected_shape)
    # test values
    d = arr[:, 0, int(arr.shape[2] / 2)]
    np.testing.assert_almost_equal(d, expected_values, decimal=2)


def test_analogs_to_csv():
    """Assert analogs's to_csv method."""
    compare_csv(filename=ANALOGS_CSV, kind='analogs')
