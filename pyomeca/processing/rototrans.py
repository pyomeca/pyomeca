from typing import Callable, Optional

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from pyomeca import Angles

from ..processing import misc


def rototrans_from_euler_angles(
    caller: Callable,
    angles: Optional[xr.DataArray] = None,
    angle_sequence: Optional[str] = None,
    translations: Optional[xr.DataArray] = None,
):
    if angles is None:
        angles = Angles()

    if translations is None:
        translations = Angles()

    if angle_sequence is None:
        angle_sequence = ""

    # Convert special zyzz angle sequence to zyz
    if angle_sequence == "zyzz":
        angles[2, :, :] -= angles[0, :, :]
        angle_sequence = "zyz"

    # If the user asked for a pure rotation
    if angles.time.size != 0 and translations.time.size == 0:
        translations = Angles(np.zeros((3, 1, angles.time.size)))

    # If the user asked for a pure translation
    if angles.time.size == 0 and translations.time.size != 0:
        angles = Angles(np.zeros((0, 1, translations.time.size)))

    # Sanity checks
    if angles.time.size != translations.time.size:
        raise IndexError(
            "Angles and translations must have the same number of frames. "
            f"You have translation = {translations.shape} and angles = {angles.shape}"
        )
    if angles.axis.size != len(angle_sequence):
        raise IndexError(
            "Angles and angles_sequence must be the same size. "
            f"You have angles axis = {angles.axis.size} and angle_sequence length = {len(angle_sequence)}"
        )
    if angles.time.size == 0:
        return caller()

    empty_rt = np.repeat(np.eye(4)[..., np.newaxis], repeats=angles.time.size, axis=2)
    rt = empty_rt.copy()
    for i in range(angles.axis.size):
        a = angles[i, ...]
        matrix_to_prod = empty_rt.copy()
        if angle_sequence[i] == "x":
            # [[1, 0     ,  0     ],
            #  [0, cos(a), -sin(a)],
            #  [0, sin(a),  cos(a)]]
            matrix_to_prod[1, 1, :] = np.cos(a)
            matrix_to_prod[1, 2, :] = -np.sin(a)
            matrix_to_prod[2, 1, :] = np.sin(a)
            matrix_to_prod[2, 2, :] = np.cos(a)
        elif angle_sequence[i] == "y":
            # [[ cos(a), 0, sin(a)],
            #  [ 0     , 1, 0     ],
            #  [-sin(a), 0, cos(a)]]
            matrix_to_prod[0, 0, :] = np.cos(a)
            matrix_to_prod[0, 2, :] = np.sin(a)
            matrix_to_prod[2, 0, :] = -np.sin(a)
            matrix_to_prod[2, 2, :] = np.cos(a)
        elif angle_sequence[i] == "z":
            # [[cos(a), -sin(a), 0],
            #  [sin(a),  cos(a), 0],
            #  [0     ,  0     , 1]]
            matrix_to_prod[0, 0, :] = np.cos(a)
            matrix_to_prod[0, 1, :] = -np.sin(a)
            matrix_to_prod[1, 0, :] = np.sin(a)
            matrix_to_prod[1, 1, :] = np.cos(a)
        else:
            raise ValueError(
                "angle_sequence must be a permutation of axes (e.g. 'xyz', 'yzx', ...)"
            )
        rt = np.einsum("ijk,jlk->ilk", rt, matrix_to_prod)
    # Put the translations
    rt[:-1, -1:, :] = translations[:3, ...]
    return caller(rt)


def rototrans_from_markers(
    caller: Callable,
    origin: xr.DataArray,
    axis_1: xr.DataArray,
    axis_2: xr.DataArray,
    axes_name: str,
    axis_to_recalculate: str,
) -> xr.DataArray:
    if origin.channel.size != 1:
        raise ValueError(
            f"`origin` must be only one marker. You have provided {origin.channel.size} markers."
        )
    if axis_1.channel.size != 2:
        raise ValueError(
            f"`axis_1` must be two markers. You have provided {axis_1.channel.size} markers."
        )
    if axis_2.channel.size != 2:
        raise ValueError(
            f"`axis_2` must be two markers. You have provided {axis_2.channel.size} markers."
        )

    # sort the axes name - If we inverse axes_names, inverse axes as well
    sorted_axes_name = "".join(sorted(axes_name))
    if axes_name != sorted_axes_name:
        axes_name = sorted_axes_name
        axis_1, axis_2 = axis_2, axis_1

    # compute vectors from markers
    vector_1 = axis_1[:3, 1, :] - axis_1[:3, 0, :]
    vector_2 = axis_2[:3, 1, :] - axis_2[:3, 0, :]

    if origin.time.size != vector_1.time.size or origin.time.size != vector_2.time.size:
        raise ValueError("Number of frame(s) for origin and axes must be the same")

    error_msg = "Axes names should be 2 values of `x`, `y` and `z` permutations"

    if axes_name[0] == "x":
        x = vector_1
        if axes_name[1] == "y":
            y = vector_2
            z = np.cross(x, y, axis=0)
        elif axes_name[1] == "z":
            z = vector_2
            y = np.cross(z, x, axis=0)
        else:
            raise ValueError(error_msg)
    elif axes_name[0] == "y":
        y = vector_1
        if axes_name[1] == "z":
            z = vector_2
            x = np.cross(y, z, axis=0)
        else:
            raise ValueError(error_msg)
    else:
        raise ValueError(error_msg)

    if axis_to_recalculate == "x":
        x = np.cross(y, z, axis=0)
    elif axis_to_recalculate == "y":
        y = np.cross(z, x, axis=0)
    elif axis_to_recalculate == "z":
        z = np.cross(x, y, axis=0)
    else:
        raise ValueError("`axis_to_recalculate must be `x`, `y` or `z`")

    rt = np.zeros((4, 4, origin.time.size))
    rt[3, 3, :] = 1
    rt = caller(rt)
    rt[:3, 0, :] = x / np.linalg.norm(x, axis=0)
    rt[:3, 1, :] = y / np.linalg.norm(y, axis=0)
    rt[:3, 2, :] = z / np.linalg.norm(z, axis=0)
    rt.meca.translation = origin
    return rt


def rototrans_from_transposed_rototrans(
    caller: Callable, rt: xr.DataArray
) -> xr.DataArray:
    rt_t = np.zeros((4, 4, rt.time.size))
    rt_t[3, 3, :] = 1
    rt_t = caller(rt_t)

    # the rotation part is just the transposed of the rotation
    rt_t.meca.rotation = rt.meca.rotation.transpose("col", "row", "time")

    # the translation part is "- rt_t * translation"
    rt_t.meca.translation = np.einsum(
        "ijk,jlk->ilk", -rt_t.meca.rotation, rt.meca.translation
    )

    return rt_t


def rototrans_from_averaged_rototrans(
    caller: Callable, rt: xr.DataArray
) -> xr.DataArray:
    # arbitrary angle sequence
    seq = "xyz"

    target = rt.mean(dim="time").expand_dims("time", axis=-1)

    angles = Angles(np.ndarray((3, 1, 1)))

    def objective_function(x):
        angles[:3, 0, 0] = x
        rt = caller.from_euler_angles(angles, seq)
        return np.ravel(rt.meca.rotation - target.meca.rotation)

    initia_guess = Angles.from_rototrans(target, seq).squeeze()
    angles[:3, 0, 0] = least_squares(objective_function, initia_guess).x
    return caller.from_euler_angles(angles, seq, translations=target.meca.translation)


def rotation_getter(array: xr.DataArray) -> xr.DataArray:
    misc.has_correct_name(array, "rototrans")
    return array[:3, :3, :]


def rotation_setter(array: xr.DataArray, value: xr.DataArray) -> None:
    misc.has_correct_name(array, "rototrans")
    array[:3, :3, :] = value[:3, :, :]


def translation_getter(array: xr.DataArray) -> xr.DataArray:
    misc.has_correct_name(array, "rototrans")
    return array[:3, 3:4, :]


def translation_setter(array: xr.DataArray, value: xr.DataArray) -> None:
    misc.has_correct_name(array, "rototrans")
    array[:3, 3:4, :] = value[:3, :, :]
