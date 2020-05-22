import numpy as np
import pytest
import xarray as xr

from pyomeca.processing import misc
from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_fft():
    is_expected_array(
        ANALOGS_DATA.meca.fft(freq=ANALOGS_DATA.rate), **EXPECTED_VALUES[40]
    )
    is_expected_array(
        ANALOGS_DATA.meca.fft(freq=ANALOGS_DATA.rate, only_positive=False),
        **EXPECTED_VALUES[41]
    )

    is_expected_array(
        MARKERS_DATA.meca.fft(freq=ANALOGS_DATA.rate), **EXPECTED_VALUES[42]
    )
    is_expected_array(
        MARKERS_DATA.meca.fft(freq=ANALOGS_DATA.rate, only_positive=False),
        **EXPECTED_VALUES[43]
    )


def test_proc_detect_onset():
    m = MARKERS_DATA[0, 0, :]
    r = xr.DataArray(m.meca.detect_onset(threshold=m.mean() + m.std()))
    is_expected_array(r, **EXPECTED_VALUES[49])

    r = xr.DataArray(
        m.meca.detect_onset(
            threshold=m.mean(), n_below=10, threshold2=m.mean() + m.std()
        )
    )
    is_expected_array(r, **EXPECTED_VALUES[50])

    np.testing.assert_array_equal(x=m.meca.detect_onset(threshold=m.mean() * 10), y=0)

    with pytest.raises(ValueError):
        MARKERS_DATA[0, :, :].meca.detect_onset(threshold=0)

    with pytest.raises(ValueError):
        MARKERS_DATA[:, :, :].meca.detect_onset(threshold=0)


def test_proc_detect_outliers():
    is_expected_array(
        MARKERS_DATA.meca.detect_outliers(threshold=3), **EXPECTED_VALUES[51]
    )
    is_expected_array(
        MARKERS_DATA.meca.detect_outliers(threshold=1), **EXPECTED_VALUES[52]
    )

    is_expected_array(
        ANALOGS_DATA.meca.detect_outliers(threshold=3), **EXPECTED_VALUES[53]
    )
    is_expected_array(
        ANALOGS_DATA.meca.detect_outliers(threshold=1), **EXPECTED_VALUES[54]
    )


def test_has_correct_name():
    misc.has_correct_name(MARKERS_DATA, "markers")

    with pytest.raises(ValueError):
        misc.has_correct_name(MARKERS_DATA, "rototrans")
