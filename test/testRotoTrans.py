from pyomeca.types import RotoTrans as PyRot
import numpy as np

angles_seq = ["x", "y", "z", "xy", "xz", "yx", "yz", "zx", "zy", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "zyzz"]
angles = np.random.rand(4, 1)
for angle_seq in angles_seq:
    angles_to_test= angles[0:len(angle_seq)]
    p = PyRot(angles=angles_to_test, angle_sequence=angle_seq)
    a = p.get_euler_angles(angle_sequence=angle_seq)
    if (a - angles_to_test).sum() < 1e-14:
        print("Test successfully passed for " + angle_seq)
    else:
        print("Test failed for " + angle_seq)

