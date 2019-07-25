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
