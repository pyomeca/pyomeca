import numpy as np

from pyomeca import Markers
from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_abs():
    is_expected_array(ANALOGS_DATA.meca.abs(), **EXPECTED_VALUES[1])
    is_expected_array(MARKERS_DATA.meca.abs(), **EXPECTED_VALUES[2])


def test_proc_matmul():
    m = MARKERS_DATA.sel(axis="x", channel="PSISl").meca.matmul(
        MARKERS_DATA.sel(axis="y", channel="PSISl")
    )
    m_ref = MARKERS_DATA.sel(axis="x", channel="PSISl") @ MARKERS_DATA.sel(
        axis="y", channel="PSISl"
    )
    np.testing.assert_array_almost_equal(m, 87604420.17968313, decimal=6)
    np.testing.assert_array_almost_equal(m, m_ref, decimal=6)

    a = ANALOGS_DATA.sel(channel="EMG1").meca.matmul(ANALOGS_DATA.sel(channel="EMG10"))
    a_ref = ANALOGS_DATA.sel(channel="EMG1") @ ANALOGS_DATA.sel(channel="EMG10")
    np.testing.assert_array_almost_equal(a, -1.41424996e-06, decimal=6)
    np.testing.assert_array_almost_equal(a, a_ref, decimal=6)


def test_proc_square_sqrt():
    is_expected_array(MARKERS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES[3])

    is_expected_array(ANALOGS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES[4])


def test_proc_norm():
    n_frames = 100
    n_markers = 10
    m = Markers(np.random.rand(3, n_markers, n_frames))

    # norm by hand
    expected_norm = np.linalg.norm(m[:3, ...], axis=0)

    # norm with pyomeca
    computed_norm = m.meca.norm(dim="axis")

    np.testing.assert_almost_equal(computed_norm, expected_norm, decimal=10)

    is_expected_array(MARKERS_DATA.meca.norm(dim="axis"), **EXPECTED_VALUES[44])
    is_expected_array(MARKERS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES[45])
    is_expected_array(MARKERS_DATA.meca.norm(dim="time"), **EXPECTED_VALUES[46])

    is_expected_array(ANALOGS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES[47])
    is_expected_array(ANALOGS_DATA.meca.norm(dim="time"), **EXPECTED_VALUES[48])


def test_proc_norm_marker():
    n_frames = 100
    n_markers = 10
    random_marker = Markers.from_random_data(size=(3, n_markers, n_frames))

    norm = random_marker.meca.norm(dim="axis")

    norm_without_ones = random_marker.drop_sel(axis="ones").meca.norm(dim="axis")

    np.testing.assert_array_equal(norm, norm_without_ones)

    expected_norm = np.ndarray((n_markers, n_frames))
    for marker in range(n_markers):
        for frame in range(n_frames):
            expected_norm[marker, frame] = np.sqrt(
                random_marker[0:3, marker, frame].dot(random_marker[0:3, marker, frame])
            )

    np.testing.assert_array_equal(norm, expected_norm)


def test_proc_rms():
    m = MARKERS_DATA.meca.rms()
    a = ANALOGS_DATA.meca.rms()

    np.testing.assert_array_almost_equal(m, 496.31764559, decimal=6)
    np.testing.assert_array_almost_equal(a, 0.00011321, decimal=6)


def test_proc_center():
    is_expected_array(MARKERS_DATA.meca.center(), **EXPECTED_VALUES[5])
    is_expected_array(
        MARKERS_DATA.meca.center(MARKERS_DATA.isel(time=0)), **EXPECTED_VALUES[6]
    )

    is_expected_array(ANALOGS_DATA.meca.center(), **EXPECTED_VALUES[7])
    is_expected_array(ANALOGS_DATA.meca.center(mu=2), **EXPECTED_VALUES[8])
    is_expected_array(
        ANALOGS_DATA.meca.center(ANALOGS_DATA.isel(time=0)), **EXPECTED_VALUES[9]
    )


def test_proc_normalize():
    is_expected_array(MARKERS_DATA.meca.normalize(), **EXPECTED_VALUES[20])
    is_expected_array(MARKERS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES[21])
    is_expected_array(
        MARKERS_DATA.meca.normalize(ref=MARKERS_DATA.sel(time=5.76)),
        **EXPECTED_VALUES[22]
    )

    is_expected_array(ANALOGS_DATA.meca.normalize(), **EXPECTED_VALUES[23])
    is_expected_array(ANALOGS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES[24])
    is_expected_array(
        ANALOGS_DATA.meca.normalize(ref=ANALOGS_DATA.sel(time=5.76)),
        **EXPECTED_VALUES[25]
    )
