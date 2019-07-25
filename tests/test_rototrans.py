"""
Test for euler to rot and rot to euler
"""
import numpy as np
import pytest

from pyomeca import RotoTrans, FrameDependentNpArray

# Define all the possible angle_sequence to tests
SEQ = ["x", "y", "z", "xy", "xz", "yx", "yz", "zx", "zy", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "zyzz"]
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

    random_from_angles = RotoTrans(RotoTrans.rt_from_euler_angles(angles=random_vector, angle_sequence="xyz"))
    np.testing.assert_equal(random_from_angles.get_num_frames(), nb_frames)
    np.testing.assert_equal(random_from_angles[0:3, 3, :], np.zeros((3, 1, nb_frames)))  # Translation is 0

    random_from_translations = RotoTrans(RotoTrans.rt_from_euler_angles(translations=random_vector))
    np.testing.assert_equal(random_from_translations.get_num_frames(), nb_frames)
    np.testing.assert_equal(random_from_translations[0:3, 0:3, :],
                            np.repeat(np.eye(3)[:, :, np.newaxis], nb_frames, axis=2))  # rotation is eye3


@pytest.mark.parametrize('seq', SEQ)
@pytest.mark.parametrize('angles', [ANGLES])
@pytest.mark.parametrize('epsilon', [EPSILON])
def test_euler2rot_rot2euler(seq, angles, epsilon):
    """Test euler to RotoTrans and RotoTrans to euler."""
    # Extract the right amount of angle relative to sequence length
    if seq != "zyzz":
        angles_to_test = angles[0:len(seq), :, :]
    else:
        angles_to_test = angles[0:3, :, :]
    # Get a RotoTrans from euler angles
    p = RotoTrans(angles=angles_to_test, angle_sequence=seq)
    # Get euler angles back from RotoTrans
    a = p.get_euler_angles(angle_sequence=seq)

    np.testing.assert_array_less((a - angles_to_test).sum(), epsilon)


@pytest.mark.parametrize('angles', [ANGLES])
def test_rt_mean(angles):
    seq = "xyz"
    angles_to_test = angles[0:3, :, :]
    p = RotoTrans(angles=angles_to_test, angle_sequence=seq)
    angles_mean = p.mean().get_euler_angles(seq)

    # Test the difference with a very relax tolerance since angles_to_compare is false by definition
    # but should not be too far most of the time
    angles_to_compare = p.get_euler_angles(angle_sequence=seq).mean()
    np.testing.assert_array_less((angles_mean - angles_to_compare).sum(), 1e-1)
