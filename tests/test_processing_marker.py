import numpy as np
import pytest

from pyomeca import Angles, Rototrans, Markers


def test_rotate():
    n_frames = 100
    n_markers = 10

    angles = Angles.from_random_data(size=(3, 1, n_frames))
    rt = Rototrans.from_euler_angles(angles, "xyz")
    markers = Markers.from_random_data(size=(3, n_markers, n_frames))

    rotated_markers = Markers.from_rototrans(markers, rt)

    expected_rotated_marker = np.ndarray((4, n_markers, n_frames))
    for marker in range(n_markers):
        for frame in range(n_frames):
            expected_rotated_marker[:, marker, frame] = np.dot(
                rt.isel(time=frame), markers.isel(channel=marker, time=frame),
            )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    rotated_markers = Markers.from_rototrans(markers.isel(time=0), rt.isel(time=0))
    expected_rotated_marker = np.ndarray(rotated_markers.shape)
    for marker in range(n_markers):
        expected_rotated_marker[:, marker] = np.dot(
            rt.isel(time=0), markers.isel(channel=marker, time=0)
        )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    rotated_markers = Markers.from_rototrans(markers, rt.isel(time=0))
    expected_rotated_marker = np.ndarray(rotated_markers.shape)
    for marker in range(n_markers):
        expected_rotated_marker[:, marker] = np.dot(
            rt.isel(time=0), markers.isel(channel=marker)
        )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    with pytest.raises(ValueError):
        Markers.from_rototrans(markers.isel(time=0), rt)
