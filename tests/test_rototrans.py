"""
Test for euler to rot and rot to euler
"""
import numpy as np
import pytest

from pyomeca import RotoTrans

# Define all the possible angle_sequence to tests
SEQ = ["x", "y", "z", "xy", "xz", "yx", "yz", "zx", "zy", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "zyzz"]
# If the difference between the initial and the final angles are less than epsilon, tests is success
EPSILON = 1e-14
# Define some random data to tests
ANGLES = np.random.rand(40, 1)


@pytest.mark.parametrize('seq', SEQ)
@pytest.mark.parametrize('angles', [ANGLES])
@pytest.mark.parametrize('epsilon', [EPSILON])
def test_euler2rot_rot2euler(seq, angles, epsilon):
    """Test euler to RotoTrans and RotoTrans to euler."""
    # Extract the right amount of angle relative to sequence length
    angles_to_test = angles[0:len(seq)]
    # Get a RotoTrans from euler angles
    p = RotoTrans(angles=angles_to_test, angle_sequence=seq)
    # Get euler angles back from RotoTrans
    a = p.get_euler_angles(angle_sequence=seq)

    np.testing.assert_array_less((a - angles_to_test).sum(), epsilon)
