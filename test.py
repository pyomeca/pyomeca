from pyomeca import Angles, Rototrans

angles = Angles.from_random_data(size=(1, 1, 100))
rt = Rototrans.from_euler_angles(angles, angle_sequence="x")

Rototrans(rt.data)
