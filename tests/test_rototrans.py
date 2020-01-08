"""
Test for euler to rot and rot to euler
"""
import numpy as np
import pytest

from pyomeca import RotoTrans, FrameDependentNpArray, Markers3d

# Define all the possible angle_sequence to tests
SEQ = [
    "x",
    "y",
    "z",
    "xy",
    "xz",
    "yx",
    "yz",
    "zx",
    "zy",
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "zyzz",
]
# If the difference between the initial and the final angles are less than epsilon, tests is success
EPSILON = 1e-12
# Define some random data to tests
ANGLES = FrameDependentNpArray(np.random.rand(40, 1, 100))


def test_construct_rt():
    # Test the ways to create an eye matrix
    eye = RotoTrans()
    np.testing.assert_equal(eye.get_num_frames(), 1)
    np.testing.assert_equal(eye[:, :, 0], np.eye(4))

    eye = RotoTrans(RotoTrans.rt_from_euler_angles())
    np.testing.assert_equal(eye.get_num_frames(), 1)
    np.testing.assert_equal(eye[:, :, 0], np.eye(4))

    # Test the way to create a rt, but not when providing bot angles and sequence
    # this is done in test_euler2rot_rot2euler
    nb_frames = 10
    random_vector = FrameDependentNpArray(np.random.rand(3, 1, nb_frames))

    random_from_angles = RotoTrans(
        RotoTrans.rt_from_euler_angles(angles=random_vector, angle_sequence="xyz")
    )
    np.testing.assert_equal(random_from_angles.get_num_frames(), nb_frames)
    np.testing.assert_equal(
        random_from_angles[0:3, 3, :], np.zeros((3, 1, nb_frames))
    )  # Translation is 0

    random_from_translations = RotoTrans(
        RotoTrans.rt_from_euler_angles(translations=random_vector)
    )
    np.testing.assert_equal(random_from_translations.get_num_frames(), nb_frames)
    np.testing.assert_equal(
        random_from_translations[0:3, 0:3, :],
        np.repeat(np.eye(3)[:, :, np.newaxis], nb_frames, axis=2),
    )  # rotation is eye3


def test_rt_from_markers():
    # Produce different RT that should be the same
    all_m = Markers3d(np.random.rand(3, 4, 100))

    rt_xy = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 1), :], all_m[:, (0, 2), :], "xy", "y"
    )
    rt_yx = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 2), :], all_m[:, (0, 1), :], "yx", "y"
    )
    rt_xy_x_recalc = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 1), :], all_m[:, (0, 2), :], "yx", "x"
    )
    rt_xy_x_recalc = rt_xy_x_recalc[:, (1, 0, 2, 3), :]
    rt_xy_x_recalc[:, 2, :] = -rt_xy_x_recalc[:, 2, :]

    rt_yz = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 1), :], all_m[:, (0, 2), :], "yz", "z"
    )
    rt_zy = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 2), :], all_m[:, (0, 1), :], "zy", "z"
    )
    rt_xy_from_yz = rt_yz[:, (1, 2, 0, 3), :]

    rt_xz = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 1), :], all_m[:, (0, 2), :], "xz", "z"
    )
    rt_zx = RotoTrans.define_axes(
        all_m[:, 0, :], all_m[:, (0, 2), :], all_m[:, (0, 1), :], "zx", "z"
    )
    rt_xy_from_zx = rt_xz[:, (0, 2, 1, 3), :]
    rt_xy_from_zx[:, 2, :] = -rt_xy_from_zx[:, 2, :]

    np.testing.assert_equal(rt_xy, rt_xy_x_recalc)
    np.testing.assert_equal(rt_xy, rt_yx)
    np.testing.assert_equal(rt_yz, rt_zy)
    np.testing.assert_equal(rt_xz, rt_zx)
    np.testing.assert_equal(rt_xy, rt_xy_from_yz)
    np.testing.assert_equal(rt_xy, rt_xy_from_zx)

    # Produce one that we are sure of the solution (this validate every others at the same time)
    m = np.ndarray((3, 3, 1))
    m[:, :, 0] = np.array(((1, 2, 3), (4, 5, 6), (6, 5, 4))).T
    rt_m_xy = RotoTrans.define_axes(
        m[:, 0, :], m[:, (0, 1), :], m[:, (0, 2), :], "xy", "y"
    )
    expected_rt_m_xy = np.array(
        [
            [0.5773502691896257, 0.7071067811865475, -0.408248290463863, 1.0],
            [0.5773502691896257, 0.0, 0.816496580927726, 2.0],
            [0.5773502691896257, -0.7071067811865475, -0.408248290463863, 3.0],
            [0, 0, 0, 1.0],
        ]
    )
    expected_rt_m_xy = expected_rt_m_xy[:, :, np.newaxis]
    np.testing.assert_equal(rt_m_xy, expected_rt_m_xy)


@pytest.mark.parametrize("seq", SEQ)
@pytest.mark.parametrize("angles", [ANGLES])
@pytest.mark.parametrize("epsilon", [EPSILON])
def test_euler2rot_rot2euler(seq, angles, epsilon):
    """Test euler to RotoTrans and RotoTrans to euler."""
    # Extract the right amount of angle relative to sequence length
    if seq != "zyzz":
        angles_to_test = angles[0 : len(seq), :, :]
    else:
        angles_to_test = angles[0:3, :, :]
    # Get a RotoTrans from euler angles
    p = RotoTrans(angles=angles_to_test, angle_sequence=seq)
    # Get euler angles back from RotoTrans
    a = p.get_euler_angles(angle_sequence=seq)

    np.testing.assert_array_less((a - angles_to_test).sum(), epsilon)


@pytest.mark.parametrize("angles", [ANGLES])
def test_rt_mean(angles):
    seq = "xyz"
    angles_to_test = angles[0:3, :, :]
    p = RotoTrans(angles=angles_to_test, angle_sequence=seq)
    angles_mean = p.mean().get_euler_angles(seq)

    # Test the difference with a very relax tolerance since angles_to_compare is false by definition
    # but should not be too far most of the time
    angles_to_compare = p.get_euler_angles(angle_sequence=seq).mean()
    np.testing.assert_array_less((angles_mean - angles_to_compare).sum(), 1e-1)


def test_rt_transpose():
    # Convert random angles to RotoTrans
    nb_frames = 10
    random_angles = FrameDependentNpArray(np.random.rand(3, 1, nb_frames))
    rt = RotoTrans.rt_from_euler_angles(random_angles, "xyz")

    # Prepare expected values for transpose
    rt_matrix = np.array(rt)
    rt_t_expected = np.zeros((4, 4, nb_frames))
    rt_t_expected[3, 3, :] = 1
    for i in range(4):
        for j in range(4):
            for k in range(nb_frames):
                rt_t_expected[j, i, k] = rt_matrix[i, j, k]
    for k in range(nb_frames):
        rt_t_expected[0:3, 3, k] = -rt_t_expected[0:3, 0:3, k].dot(rt_matrix[0:3, 3, k])

    # Compute the transpose using pyomeca
    rt_t = rt.transpose()

    # Compare the two
    np.testing.assert_almost_equal(rt_t, rt_t_expected, decimal=10)


def test_rt_inverse():
    # Convert random angles to RotoTrans
    nb_frames = 10
    random_angles = FrameDependentNpArray(np.random.rand(3, 1, nb_frames))
    rt = RotoTrans.rt_from_euler_angles(random_angles, "xyz")

    # Prepare expected values for inverse
    rt_matrix = np.array(rt)
    rt_t_expected = np.zeros((4, 4, nb_frames))
    rt_t_expected[3, 3, :] = 1
    for i in range(4):
        for j in range(4):
            for k in range(nb_frames):
                rt_t_expected[j, i, k] = rt_matrix[i, j, k]
    for k in range(nb_frames):
        rt_t_expected[0:3, 3, k] = -rt_t_expected[0:3, 0:3, k].dot(rt_matrix[0:3, 3, k])

    # Compute the inverse using pyomeca
    rt_t = rt.inverse()

    # Compare the two
    np.testing.assert_almost_equal(rt_t, rt_t_expected, decimal=10)
