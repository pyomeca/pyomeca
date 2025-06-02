import numpy as np

from tests._constants import ANALOGS_DATA, EXPECTED_VALUES, MARKERS_DATA
from tests.utils import is_expected_array


def test_proc_time_normalize():
    is_expected_array(MARKERS_DATA.meca.time_normalize(), **EXPECTED_VALUES[26])
    is_expected_array(
        MARKERS_DATA.meca.time_normalize(n_frames=1000), **EXPECTED_VALUES[27]
    )
    time_vector = np.linspace(MARKERS_DATA.time[0], MARKERS_DATA.time[100], 100)
    is_expected_array(
        MARKERS_DATA.meca.time_normalize(time_vector=time_vector), **EXPECTED_VALUES[28]
    )
    is_expected_array(
        MARKERS_DATA.meca.time_normalize(norm_time=True).time, **EXPECTED_VALUES[55]
    )

    is_expected_array(ANALOGS_DATA.meca.time_normalize(), **EXPECTED_VALUES[29])
    is_expected_array(
        ANALOGS_DATA.meca.time_normalize(n_frames=1000), **EXPECTED_VALUES[30]
    )
    time_vector = np.linspace(ANALOGS_DATA.time[0], ANALOGS_DATA.time[100], 100)
    is_expected_array(
        ANALOGS_DATA.meca.time_normalize(time_vector=time_vector), **EXPECTED_VALUES[31]
    )



def test_proc_interpolate():

    # Fake data
    marker_data_with_nans = MARKERS_DATA.copy()
    marker_data_with_nans.values[0, 0, 5:10] = [1, np.nan, np.nan, np.nan, 2]
    analog_data_with_nans = ANALOGS_DATA.copy()
    analog_data_with_nans.values[0, 5:10] = [1, np.nan, np.nan, np.nan, 2]

    # Test that it has NaNs
    np.testing.assert_array_equal(marker_data_with_nans.values[0, 0, 5:10], [1, np.nan, np.nan, np.nan, 2])
    np.testing.assert_array_equal(analog_data_with_nans.values[0, 5:10], [1, np.nan, np.nan, np.nan, 2])

    # Test that it filled up the nans
    np.testing.assert_almost_equal(marker_data_with_nans.meca.interpolate_missing_data().values[0, 0, 5:10], [1, 1.25, 1.5, 1.75, 2])
    np.testing.assert_almost_equal(analog_data_with_nans.meca.interpolate_missing_data().values[0, 5:10], [1, 1.25, 1.5, 1.75, 2])
