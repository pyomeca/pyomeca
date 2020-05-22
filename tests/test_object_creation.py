import numpy as np
import pytest
import xarray as xr

from pyomeca import Analogs, Markers, Angles, Rototrans
from ._constants import ANALOGS_DATA, MARKERS_DATA, EXPECTED_VALUES
from .utils import is_expected_array


def test_analogs_creation():
    dims = ("channel", "time")
    array = Analogs()
    np.testing.assert_array_equal(x=array, y=xr.DataArray())
    assert array.dims == dims

    array = Analogs(ANALOGS_DATA.values)
    is_expected_array(array, **EXPECTED_VALUES[56])

    size = 10, 100
    array = Analogs.from_random_data(size=size)
    assert array.shape == size
    assert array.dims == dims

    with pytest.raises(ValueError):
        Analogs(MARKERS_DATA)


def test_markers_creation():
    dims = ("axis", "channel", "time")
    array = Markers()
    np.testing.assert_array_equal(x=array, y=xr.DataArray())
    assert array.dims == dims

    array = Markers(MARKERS_DATA.values)
    is_expected_array(array, **EXPECTED_VALUES[57])

    size = 3, 10, 100
    array = Markers.from_random_data(size=size)
    assert array.shape == (4, size[1], size[2])
    assert array.dims == dims

    with pytest.raises(ValueError):
        Markers(ANALOGS_DATA)


def test_angles_creation():
    dims = ("axis", "channel", "time")
    array = Angles()
    np.testing.assert_array_equal(x=array, y=xr.DataArray())
    assert array.dims == dims

    array = Angles(MARKERS_DATA.values, time=MARKERS_DATA.time)
    is_expected_array(array, **EXPECTED_VALUES[57])

    size = 10, 10, 100
    array = Angles.from_random_data(size=size)
    assert array.shape == size
    assert array.dims == dims

    with pytest.raises(ValueError):
        Angles(ANALOGS_DATA)


def test_rototrans_creation():
    dims = ("row", "col", "time")
    array = Rototrans()
    np.testing.assert_array_equal(x=array, y=xr.DataArray(np.eye(4)[..., np.newaxis]))
    assert array.dims == dims

    array = Rototrans(MARKERS_DATA.values, time=MARKERS_DATA.time)
    is_expected_array(array, **EXPECTED_VALUES[67])

    size = 4, 4, 100
    array = Rototrans.from_random_data(size=size)
    assert array.shape == size
    assert array.dims == dims

    with pytest.raises(ValueError):
        Angles(ANALOGS_DATA)
