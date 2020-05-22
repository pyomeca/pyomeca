import numpy as np

from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
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
