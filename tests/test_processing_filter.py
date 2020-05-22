from tests._constants import ANALOGS_DATA, MARKERS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_filters():
    freq = ANALOGS_DATA.rate
    order = 2

    is_expected_array(
        ANALOGS_DATA.meca.low_pass(freq=freq, order=order, cutoff=5),
        **EXPECTED_VALUES[32],
    )
    is_expected_array(
        ANALOGS_DATA.meca.high_pass(freq=freq, order=order, cutoff=100),
        **EXPECTED_VALUES[33],
    )
    is_expected_array(
        ANALOGS_DATA.meca.band_pass(freq=freq, order=order, cutoff=[10, 200]),
        **EXPECTED_VALUES[34],
    )
    is_expected_array(
        ANALOGS_DATA.meca.band_stop(freq=freq, order=order, cutoff=[40, 60]),
        **EXPECTED_VALUES[35],
    )

    freq = MARKERS_DATA.rate
    is_expected_array(
        MARKERS_DATA.meca.low_pass(freq=freq, order=order, cutoff=5),
        **EXPECTED_VALUES[36],
    )
    is_expected_array(
        MARKERS_DATA.meca.high_pass(freq=freq, order=order, cutoff=10),
        **EXPECTED_VALUES[37],
    )
    is_expected_array(
        MARKERS_DATA.meca.band_pass(freq=freq, order=order, cutoff=[1, 10]),
        **EXPECTED_VALUES[38],
    )
    is_expected_array(
        MARKERS_DATA.meca.band_stop(freq=freq, order=order, cutoff=[5, 6]),
        **EXPECTED_VALUES[39],
    )
