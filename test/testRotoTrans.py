"""
Unittest and example script for euler to rot and rot to euler
"""
from pyomeca.types import RotoTrans
import numpy as np

# Define all the possible angle_sequence to test
angles_seq = ["x", "y", "z", "xy", "xz", "yx", "yz", "zx", "zy", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "zyzz"]

# Define some random data to test
angles = np.random.rand(4, 1)

# Test all sequences
for angle_seq in angles_seq:
    # Extract the right amount of angle relative to sequence length
    angles_to_test = angles[0:len(angle_seq)]

    # Get a RotoTrans from euler angles
    p = RotoTrans(angles=angles_to_test, angle_sequence=angle_seq)

    # Get euler angles back from RotoTrans
    a = p.get_euler_angles(angle_sequence=angle_seq)

    # If the difference between the initial and the final angles are less than epsilon, test is success
    if (a - angles_to_test).sum() < 1e-14:
        print(f'Test successfully passed for {angle_seq}')
    else:
        print(f'Test failed for {angle_seq}')
