from itertools import permutations

import numpy as np
import pytest

from pyomeca import Angles, Rototrans, Markers

SEQ = (
    ["".join(p) for i in range(1, 4) for p in permutations("xyz", i)]
    + ["zyzz"]
    + ["zxz"]
)
SEQ = [s for s in SEQ if s not in ["yxz"]]
EPSILON = 1e-12
ANGLES = Angles(np.random.rand(4, 1, 100))


@pytest.mark.parametrize("seq", SEQ)
def test_euler2rot_rot2euleur(seq, angles=ANGLES, epsilon=EPSILON):
    if seq == "zyzz":
        angles_to_test = angles[:3, ...]
    else:
        angles_to_test = angles[: len(seq), ...]
    r = Rototrans.from_euler_angles(angles=angles_to_test, angle_sequence=seq)
    a = Angles.from_rototrans(rt=r, angle_sequence=seq)

    np.testing.assert_array_less((a - angles_to_test).meca.abs().sum(), epsilon)


def test_construct_rt():
    eye = Rototrans()
    np.testing.assert_equal(eye.time.size, 1)
    np.testing.assert_equal(eye.sel(time=0), np.eye(4))

    eye = Rototrans.from_euler_angles()
    np.testing.assert_equal(eye.time.size, 1)
    np.testing.assert_equal(eye.sel(time=0), np.eye(4))

    # Test the way to create a rt, but not when providing bot angles and sequence
    nb_frames = 10
    random_vector = Angles(np.random.rand(3, 1, nb_frames))

    # with angles
    rt_random_angles = Rototrans.from_euler_angles(
        angles=random_vector, angle_sequence="xyz"
    )
    np.testing.assert_equal(rt_random_angles.time.size, nb_frames)
    np.testing.assert_equal(
        rt_random_angles[:-1, -1:, :], np.zeros((3, 1, nb_frames))
    )  # Translation is 0

    # with translation
    rt_random_translation = Rototrans.from_euler_angles(translations=random_vector)
    np.testing.assert_equal(rt_random_translation.time.size, nb_frames)
    np.testing.assert_equal(
        rt_random_translation[:3, :3, :],
        np.repeat(np.eye(3)[:, :, np.newaxis], nb_frames, axis=2),
    )  # rotation is eye3
    np.arange(0, rt_random_angles.time.size / 0.5, 1 / 0.5)

    rt_with_time = Rototrans(
        rt_random_angles, time=np.arange(0, rt_random_angles.time.size / 100, 1 / 100),
    )
    assert rt_with_time.time[-1] == 0.09

    with pytest.raises(IndexError):
        Rototrans(data=np.zeros(1))

    with pytest.raises(IndexError):
        Rototrans.from_euler_angles(
            angles=random_vector[..., :5],
            translations=random_vector,
            angle_sequence="x",
        )

    with pytest.raises(IndexError):
        Rototrans.from_euler_angles(angles=random_vector, angle_sequence="x")

    with pytest.raises(ValueError):
        Rototrans.from_euler_angles(angles=random_vector, angle_sequence="nop")


def test_rt_from_markers():
    all_m = Markers.from_random_data()

    rt_xy = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 1]),
        axis_2=all_m.isel(channel=[0, 2]),
        axes_name="xy",
        axis_to_recalculate="y",
    )

    rt_yx = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 2]),
        axis_2=all_m.isel(channel=[0, 1]),
        axes_name="yx",
        axis_to_recalculate="y",
    )

    rt_xy_x_recalc = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 1]),
        axis_2=all_m.isel(channel=[0, 2]),
        axes_name="yx",
        axis_to_recalculate="x",
    )
    rt_xy_x_recalc = rt_xy_x_recalc.isel(col=[1, 0, 2, 3])
    rt_xy_x_recalc[:, 2, :] = -rt_xy_x_recalc[:, 2, :]

    rt_yz = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 1]),
        axis_2=all_m.isel(channel=[0, 2]),
        axes_name="yz",
        axis_to_recalculate="z",
    )

    rt_zy = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 2]),
        axis_2=all_m.isel(channel=[0, 1]),
        axes_name="zy",
        axis_to_recalculate="z",
    )
    rt_xy_from_yz = rt_yz.isel(col=[1, 2, 0, 3])

    rt_xz = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 1]),
        axis_2=all_m.isel(channel=[0, 2]),
        axes_name="xz",
        axis_to_recalculate="z",
    )

    rt_zx = Rototrans.from_markers(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 2]),
        axis_2=all_m.isel(channel=[0, 1]),
        axes_name="zx",
        axis_to_recalculate="z",
    )
    rt_xy_from_zx = rt_xz.isel(col=[0, 2, 1, 3])
    rt_xy_from_zx[:, 2, :] = -rt_xy_from_zx[:, 2, :]

    np.testing.assert_array_equal(rt_xy, rt_xy_x_recalc)
    np.testing.assert_array_equal(rt_xy, rt_yx)
    np.testing.assert_array_equal(rt_yz, rt_zy)
    np.testing.assert_array_equal(rt_xz, rt_zx)
    np.testing.assert_array_equal(rt_xy, rt_xy_from_yz)
    np.testing.assert_array_equal(rt_xy, rt_xy_from_zx)

    # Produce one that we know the solution
    ref_m = Markers(np.array(((1, 2, 3), (4, 5, 6), (6, 5, 4))).T[:, :, np.newaxis])
    rt_xy_from_known_m = Rototrans.from_markers(
        origin=ref_m.isel(channel=[0]),
        axis_1=ref_m.isel(channel=[0, 1]),
        axis_2=ref_m.isel(channel=[0, 2]),
        axes_name="xy",
        axis_to_recalculate="y",
    )

    rt_xy_expected = Rototrans(
        np.array(
            [
                [0.5773502691896257, 0.7071067811865475, -0.408248290463863, 1.0],
                [0.5773502691896257, 0.0, 0.816496580927726, 2.0],
                [0.5773502691896257, -0.7071067811865475, -0.408248290463863, 3.0],
                [0, 0, 0, 1.0],
            ]
        )
    )

    np.testing.assert_array_equal(rt_xy_from_known_m, rt_xy_expected)

    exception_default_params = dict(
        origin=all_m.isel(channel=[0]),
        axis_1=all_m.isel(channel=[0, 1]),
        axis_2=all_m.isel(channel=[0, 2]),
        axes_name="xy",
        axis_to_recalculate="y",
    )
    with pytest.raises(ValueError):
        Rototrans.from_markers(
            **{**exception_default_params, **dict(origin=all_m.isel(channel=[0, 1]))}
        )

    with pytest.raises(ValueError):
        Rototrans.from_markers(
            **{**exception_default_params, **dict(axis_1=all_m.isel(channel=[0]))}
        )

    with pytest.raises(ValueError):
        Rototrans.from_markers(
            **{**exception_default_params, **dict(axis_2=all_m.isel(channel=[0]))}
        )

    with pytest.raises(ValueError):
        Rototrans.from_markers(
            **{
                **exception_default_params,
                **dict(axis_1=all_m.isel(channel=[0, 1], time=slice(None, 50))),
            }
        )

    with pytest.raises(ValueError):
        Rototrans.from_markers(**{**exception_default_params, **dict(axes_name="yyz")})

    with pytest.raises(ValueError):
        Rototrans.from_markers(**{**exception_default_params, **dict(axes_name="xxz")})

    with pytest.raises(ValueError):
        Rototrans.from_markers(**{**exception_default_params, **dict(axes_name="zzz")})

    with pytest.raises(ValueError):
        Rototrans.from_markers(
            **{**exception_default_params, **dict(axis_to_recalculate="h")}
        )


def test_rt_transpose():
    n_frames = 10
    angles = Angles.from_random_data(size=(3, 1, n_frames))
    rt = Rototrans.from_euler_angles(angles, angle_sequence="xyz")

    rt_t = Rototrans.from_transposed_rototrans(rt)

    rt_t_expected = np.zeros((4, 4, n_frames))
    rt_t_expected[3, 3, :] = 1
    for row in range(rt.row.size):
        for col in range(rt.col.size):
            for frame in range(rt.time.size):
                rt_t_expected[col, row, frame] = rt[row, col, frame]

    for frame in range(rt.time.size):
        rt_t_expected[:3, 3, frame] = -rt_t_expected[:3, :3, frame].dot(
            rt[:3, 3, frame]
        )

    np.testing.assert_array_almost_equal(rt_t, rt_t_expected, decimal=10)


def test_average_rt():
    # TODO: investigate why this does not work
    # angles = Angles.from_random_data(size=(3, 1, 100))
    # or
    # angles = Angles(np.arange(300).reshape((3, 1, 100)))
    angles = Angles(np.random.rand(3, 1, 100))
    seq = "xyz"

    rt = Rototrans.from_euler_angles(angles, seq)
    rt_mean = Rototrans.from_averaged_rototrans(rt)
    angles_mean = Angles.from_rototrans(rt_mean, seq).isel(time=0)

    angles_mean_ref = Angles.from_rototrans(rt, seq).mean(dim="time")

    np.testing.assert_array_almost_equal(angles_mean, angles_mean_ref, decimal=2)
