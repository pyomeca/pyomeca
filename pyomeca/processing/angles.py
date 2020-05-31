from typing import Callable

import numpy as np
import xarray as xr


def angles_from_rototrans(
    caller: Callable, rt: xr.DataArray, angle_sequence: str
) -> xr.DataArray:
    if angle_sequence == "zyzz":
        angles = caller(np.ndarray(shape=(3, 1, rt.time.size)))
    else:
        angles = caller(np.ndarray(shape=(len(angle_sequence), 1, rt.time.size)))

    if angle_sequence == "x":
        angles[0, :, :] = np.arcsin(rt[2, 1, :])
    elif angle_sequence == "y":
        angles[0, :, :] = np.arcsin(rt[0, 2, :])
    elif angle_sequence == "z":
        angles[0, :, :] = np.arcsin(rt[1, 0, :])
    elif angle_sequence == "xy":
        angles[0, :, :] = np.arcsin(rt[2, 1, :])
        angles[1, :, :] = np.arcsin(rt[0, 2, :])
    elif angle_sequence == "xz":
        angles[0, :, :] = -np.arcsin(rt[1, 2, :])
        angles[1, :, :] = -np.arcsin(rt[0, 1, :])
    elif angle_sequence == "yx":
        angles[0, :, :] = -np.arcsin(rt[2, 0, :])
        angles[1, :, :] = -np.arcsin(rt[1, 2, :])
    elif angle_sequence == "yz":
        angles[0, :, :] = np.arcsin(rt[0, 2, :])
        angles[1, :, :] = np.arcsin(rt[1, 0, :])
    elif angle_sequence == "zx":
        angles[0, :, :] = np.arcsin(rt[1, 0, :])
        angles[1, :, :] = np.arcsin(rt[2, 1, :])
    elif angle_sequence == "zy":
        angles[0, :, :] = -np.arcsin(rt[0, 1, :])
        angles[1, :, :] = -np.arcsin(rt[2, 0, :])
    elif angle_sequence == "xyz":
        angles[0, :, :] = np.arctan2(-rt[1, 2, :], rt[2, 2, :])
        angles[1, :, :] = np.arcsin(rt[0, 2, :])
        angles[2, :, :] = np.arctan2(-rt[0, 1, :], rt[0, 0, :])
    elif angle_sequence == "xzy":
        angles[0, :, :] = np.arctan2(rt[2, 1, :], rt[1, 1, :])
        angles[2, :, :] = np.arctan2(rt[0, 2, :], rt[0, 0, :])
        angles[1, :, :] = np.arcsin(-rt[0, 1, :])
    elif angle_sequence == "yzx":
        angles[2, :, :] = np.arctan2(-rt[1, 2, :], rt[1, 1, :])
        angles[0, :, :] = np.arctan2(-rt[2, 0, :], rt[0, 0, :])
        angles[1, :, :] = np.arcsin(rt[1, 0, :])
    elif angle_sequence == "zxy":
        angles[1, :, :] = np.arcsin(rt[2, 1, :])
        angles[2, :, :] = np.arctan2(-rt[2, 0, :], rt[2, 2, :])
        angles[0, :, :] = np.arctan2(-rt[0, 1, :], rt[1, 1, :])
    elif angle_sequence in ["zyz", "zyzz"]:
        angles[0, :, :] = np.arctan2(rt[1, 2, :], rt[0, 2, :])
        angles[1, :, :] = np.arccos(rt[2, 2, :])
        angles[2, :, :] = np.arctan2(rt[2, 1, :], -rt[2, 0, :])
    elif angle_sequence == "zyx":
        angles[2, :, :] = np.arctan2(rt[2, 1, :], rt[2, 2, :])
        angles[1, :, :] = np.arcsin(-rt[2, 0, :])
        angles[0, :, :] = np.arctan2(rt[1, 0, :], rt[0, 0, :])
    elif angle_sequence == "zxz":
        angles[0, :, :] = np.arctan2(rt[0, 2, :], -rt[1, 2, :])
        angles[1, :, :] = np.arccos(rt[2, 2, :])
        angles[2, :, :] = np.arctan2(rt[2, 0, :], rt[2, 1, :])

    return angles
