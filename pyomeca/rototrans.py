import numpy as np
from scipy.optimize import least_squares

from pyomeca import FrameDependentNpArray, FrameDependentNpArrayCollection, Markers3d


class RotoTrans(FrameDependentNpArray):
    def __new__(
        cls,
        rt=np.eye(4),
        angles=FrameDependentNpArray(),
        angle_sequence="",
        translations=FrameDependentNpArray(),
        *args,
        **kwargs
    ):
        """

        Parameters
        ----------
        rt : FrameDependentNpArray (4x4xF)
            Rototranslation matrix sorted in 4x4xF, default is the matrix that don't rotate nor translate the system, is
            ineffective if angles is provided
        angles : FrameDependentNpArray
            Euler angles of the rototranslation, angles parameter is ineffective if angles_sequence if not defined, but
            will override rt
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of 3 axes (e.g. "xyz", "yzx", ...)
        translations : FrameDependentNpArray
            First 3 rows of 4th row, translation is ineffective if angles is not provided
        """

        # Determine if we construct RotoTrans from rt or angles/translations
        if angle_sequence:
            rt = cls.rt_from_euler_angles(
                angles=angles, angle_sequence=angle_sequence, translations=translations
            )

        else:
            s = rt.shape
            if s[0] != 4 or s[1] != 4:
                raise IndexError("RotoTrans must by a 4x4xF matrix")
            # Make sure last line reads [0, 0, 0, 1]
            if len(s) == 2:
                rt[3, :] = np.array([0, 0, 0, 1])
            else:
                rt[3, 0:3, :] = 0
                rt[3, 3, :] = 1

        # Finally, we must return the newly created object:
        return super(RotoTrans, cls).__new__(cls, array=rt, *args, **kwargs)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, RotoTrans):
            return

    # --- Linear algebra methods

    def get_euler_angles(self, angle_sequence):
        """

        Parameters
        ----------
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
        Returns
        -------
        angles : Markers3d
            Euler angles associated with RotoTrans
        """
        if angle_sequence != "zyzz":
            angles = FrameDependentNpArray(
                np.ndarray((len(angle_sequence), 1, self.get_num_frames()))
            )
        else:
            angles = FrameDependentNpArray(np.ndarray((3, 1, self.get_num_frames())))

        if angle_sequence == "x":
            angles[0, :, :] = np.arcsin(self[2, 1, :])
        elif angle_sequence == "y":
            angles[0, :, :] = np.arcsin(self[0, 2, :])
        elif angle_sequence == "z":
            angles[0, :, :] = np.arcsin(self[1, 0, :])
        elif angle_sequence == "xy":
            angles[0, :, :] = np.arcsin(self[2, 1, :])
            angles[1, :, :] = np.arcsin(self[0, 2, :])
        elif angle_sequence == "xz":
            angles[0, :, :] = -np.arcsin(self[1, 2, :])
            angles[1, :, :] = -np.arcsin(self[0, 1, :])
        elif angle_sequence == "yx":
            angles[0, :, :] = -np.arcsin(self[2, 0, :])
            angles[1, :, :] = -np.arcsin(self[1, 2, :])
        elif angle_sequence == "yz":
            angles[0, :, :] = np.arcsin(self[0, 2, :])
            angles[1, :, :] = np.arcsin(self[1, 0, :])
        elif angle_sequence == "zx":
            angles[0, :, :] = np.arcsin(self[1, 0, :])
            angles[1, :, :] = np.arcsin(self[2, 1, :])
        elif angle_sequence == "zy":
            angles[0, :, :] = -np.arcsin(self[0, 1, :])
            angles[1, :, :] = -np.arcsin(self[2, 0, :])
        elif angle_sequence == "xyz":
            angles[0, :, :] = np.arctan2(-self[1, 2, :], self[2, 2, :])
            angles[1, :, :] = np.arcsin(self[0, 2, :])
            angles[2, :, :] = np.arctan2(-self[0, 1, :], self[0, 0, :])
        elif angle_sequence == "xzy":
            angles[0, :, :] = np.arctan2(self[2, 1, :], self[1, 1, :])
            angles[2, :, :] = np.arctan2(self[0, 2, :], self[0, 0, :])
            angles[1, :, :] = np.arcsin(-self[0, 1, :])
        elif angle_sequence == "yxz":
            angles[1, :, :] = np.arcsin(-self[1, 2, :])
            angles[0, :, :] = np.arctan2(self[0, 2, :], self[2, 2, :])
            angles[2, :, :] = np.arctan2(self[1, 0, :], self[1, 1, :])
        elif angle_sequence == "yzx":
            angles[2, :, :] = np.arctan2(-self[1, 2, :], self[1, 1, :])
            angles[0, :, :] = np.arctan2(-self[2, 0, :], self[0, 0, :])
            angles[1, :, :] = np.arcsin(self[1, 0, :])
        elif angle_sequence == "zxy":
            angles[1, :, :] = np.arcsin(self[2, 1, :])
            angles[2, :, :] = np.arctan2(-self[2, 0, :], self[2, 2, :])
            angles[0, :, :] = np.arctan2(-self[0, 1, :], self[1, 1, :])
        elif angle_sequence == "zyx":
            angles[2, :, :] = np.arctan2(self[2, 1, :], self[2, 2, :])
            angles[1, :, :] = np.arcsin(-self[2, 0, :])
            angles[0, :, :] = np.arctan2(self[1, 0, :], self[0, 0, :])
        elif angle_sequence == "zyz":
            angles[0, :, :] = np.arctan2(self[1, 2, :], self[0, 2, :])
            angles[1, :, :] = np.arccos(self[2, 2, :])
            angles[2, :, :] = np.arctan2(self[2, 1, :], -self[2, 0, :])
        elif angle_sequence == "zxz":
            angles[0, :, :] = np.arctan2(self[0, 2, :], -self[1, 2, :])
            angles[1, :, :] = np.arccos(self[2, 2, :])
            angles[2, :, :] = np.arctan2(self[2, 0, :], self[2, 1, :])
        elif angle_sequence == "zyzz":
            angles[0, :, :] = np.arctan2(self[1, 2, :], self[0, 2, :])
            angles[1, :, :] = np.arccos(self[2, 2, :])
            angles[2, :, :] = np.arctan2(self[2, 1, :], -self[2, 0, :])

        return angles

    @staticmethod
    def rt_from_euler_angles(
        angles=FrameDependentNpArray(),
        angle_sequence="",
        translations=FrameDependentNpArray(),
    ):
        """

        Parameters
        ----------
        angles : FrameDependentNpArray
            Euler angles of the rototranslation
        angle_sequence : str
            Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
        translations : FrameDependentNpArray
            Translation part of the Rototrans matrix

        Returns
        -------
        rt : RotoTrans
            The rototranslation associated to the input parameters
        """
        # Convert special zyzz angle sequence to zyz
        if angle_sequence == "zyzz":
            angles[2, :, :] -= angles[0, :, :]
            angle_sequence = "zyz"

        # If the user asked for a pure rotation
        if angles.get_num_frames() != 0 and translations.get_num_frames() == 0:
            translations = FrameDependentNpArray(
                np.zeros((3, 1, angles.get_num_frames()))
            )

        # If the user asked for a pure translation
        if angles.get_num_frames() == 0 and translations.get_num_frames() != 0:
            angles = FrameDependentNpArray(
                np.zeros((0, 1, translations.get_num_frames()))
            )

        # Sanity checks
        if angles.get_num_frames() != translations.get_num_frames():
            raise IndexError(
                "angles and translations must have the same number of frames"
            )
        if angles.shape[0] is not len(angle_sequence):
            raise IndexError("angles and angles_sequence must be the same size")
        if angles.get_num_frames() == 0:
            return RotoTrans()

        rt_out = np.repeat(np.eye(4)[:, :, np.newaxis], angles.get_num_frames(), axis=2)
        try:
            for i in range(len(angles)):
                a = angles[i, :, :]
                matrix_to_prod = np.repeat(
                    np.eye(4)[:, :, np.newaxis], angles.get_num_frames(), axis=2
                )
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
                        "angle_sequence must be a permutation of axes (e.g. "
                        "xyz"
                        ", "
                        "yzx"
                        ", ...)"
                    )
                rt_out = np.einsum("ijk,jlk->ilk", rt_out, matrix_to_prod)
        except IndexError:
            raise ValueError(
                "angle_sequence must be a permutation of axes (e.g. "
                "xyz"
                ", "
                "yzx"
                ", ...)"
            )

        # Put the translations
        rt_out[0:3, 3:4, :] = translations[0:3, :, :]

        return RotoTrans(rt_out)

    @staticmethod
    def define_axes(origin, axis1, axis2, axes_name, axis_to_recalculate):
        """
        This function creates system of axes from axis1 and axis2
        Parameters
        ----------
        origin : Markers3d
            A Markers3d of dimension [4, 1, nb_frames] of the origin in global reference frame of the RotoTrans
            Matrix (corresponding to the translation column)
        axis1 : Marker3d
            A Markers3d of dimension [4, 2, nb_frames] that describe the axis 1,
            the first marker being the beginning of the vector, the second being the end of the vector
        axis2 : Marker3d
            A Markers3d of dimension [4, 2, nb_frames] that describe the axis 2,
            the first marker being the beginning of the vector, the second being the end of the vector
        axes_name : str
            Name of the axis1 and axis2 respectively ("xy", "yx", "xz", ...)
        axis_to_recalculate : str
            Which of the 2 axes to recalculate

        Returns
        -------
        The systems of axes described by the markers for each frames
        """
        # Do some checks on the number of markers
        if origin.shape[1] != 1:
            raise ValueError("Origin must be only one marker")
        if axis1.shape[1] != 2:
            raise ValueError("Axis1 must have 2 markers, one beginning and one end")
        if axis2.shape[1] != 2:
            raise ValueError("Axis2 must have 2 markers, one beginning and one end")

        # Sort the axes name, If we inverse axes_names, inverse axes as well
        axes_name_tp = "".join(sorted(axes_name))
        if axes_name != axes_name_tp:
            axis1_copy = axis1
            axis1 = axis2
            axis2 = axis1_copy
            axes_name = axes_name_tp

        # Compute vectors from Markers
        axis1 = axis1[0:3, 1, :] - axis1[0:3, 0, :]
        axis2 = axis2[0:3, 1, :] - axis2[0:3, 0, :]

        if len(origin.shape) == 2:
            origin = origin.reshape((origin.shape[0], origin.shape[1], 1))
        if len(axis1.shape) == 2:
            axis1 = axis1.reshape((axis1.shape[0], axis1.shape[1], 1))
        if len(axis2.shape) == 2:
            axis2 = axis2.reshape((axis2.shape[0], axis2.shape[1], 1))

        # Do some checks on number of frames
        if origin.shape[2] != axis1.shape[2] or origin.shape[2] != axis2.shape[2]:
            raise ValueError("Number of frame for origin and axes must be the same")

        error_msg = "Axes names should be 2 values of `x`, `y` and `z` permutations"
        if axes_name[0] == "x":
            x = axis1
            if axes_name[1] == "y":
                y = axis2
                z = np.cross(x, y, axis=0)
            elif axes_name[1] == "z":
                z = axis2
                y = np.cross(z, x, axis=0)
            else:
                raise ValueError(error_msg)

        elif axes_name[0] == "y":
            y = axis1
            if axes_name[1] == "z":
                z = axis2
                x = np.cross(y, z, axis=0)
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)

        # Recalculate the temporary axis
        if axis_to_recalculate == "x":
            x = np.cross(y, z, axis=0)
        elif axis_to_recalculate == "y":
            y = np.cross(z, x, axis=0)
        elif axis_to_recalculate == "z":
            z = np.cross(x, y, axis=0)
        else:
            raise ValueError("Axis to recalculate must be `x`, `y` or `z`")

        # Normalize each vector
        x = x / np.linalg.norm(x, axis=0)
        y = y / np.linalg.norm(y, axis=0)
        z = z / np.linalg.norm(z, axis=0)

        rt = RotoTrans(rt=np.zeros((4, 4, origin.shape[2])))
        rt[0:3, 0, :] = x
        rt[0:3, 1, :] = y
        rt[0:3, 2, :] = z
        rt.set_translation(origin)
        return rt

    def rotation(self):
        """
        Returns
        -------
        Rotation part of the RotoTrans
        """
        return self[0:3, 0:3, :]

    def set_rotation(self, r):
        """
        Set rotation part of the RotoTrans
        Parameters
        ----------
        r : np.array
            A 3x3xN rotation matrix
        """
        self[0:3, 0:3, :] = r

    def translation(self):
        """
        Returns
        -------
        Translation part of the RotoTrans
        """
        return self[0:3, 3:4, :]

    def set_translation(self, t):
        """
        Set translation part of the RotoTrans
        Parameters
        ----------
        t : np.array
            A 3x1xN vector
        """
        self[0:3, 3:4, :] = t[0:3, :, :]

    def transpose(self):
        """

        Returns
        -------
        Rt_t : RotoTrans
            Transposed RotoTrans matrix ([R.T -R.T*t],[0 0 0 1])
        """
        # Create a matrix with the transposed rotation part
        rt_t = RotoTrans(rt=np.zeros((4, 4, self.get_num_frames())))
        # Fill the last column and row with 0 and bottom corner with 1
        rt_t[3, 3, :] = 1

        # The rotation part is just the transposed of the rotation
        rt_t[0:3, 0:3, :] = np.transpose(self[0:3, 0:3, :], (1, 0, 2))

        # Transpose the translation part is "-rt_transposed * Translation"
        rt_t[0:3, 3, :] = np.einsum("ijk,jlk->ilk", -rt_t[0:3, 0:3, :], self[0:3, 3, :])

        # Return transposed matrix
        return rt_t

    def inverse(self):
        """

        Returns
        -------
        Inverse of the RotoTrans matrix (which is by definition the transposed matrix)
        """
        return self.transpose()

    def mean(self):
        """

        Returns
        -------
        Performs an optimization to compute the mean over the frames
        """

        # Chose an arbitrary angle sequence to convert into angle during the optimization
        seq = "xyz"

        # Compute the element-wise mean for the optimization to target
        rt_mean = super(RotoTrans, self).mean()

        # Define the objective function
        x_tp = FrameDependentNpArray(np.ndarray((3, 1, 1)))

        def obj(x):
            x_tp[0:3, 0, 0] = x.reshape(-1, 1)
            rt = RotoTrans(angles=x_tp, angle_sequence=seq)
            return (rt[0:3, 0:3] - rt_mean[0:3, 0:3]).reshape(9)

        # Initial guess of the optimization
        x0 = np.squeeze(rt_mean.get_euler_angles(seq))

        # Call the optimizer
        x_tp[0:3, 0, 0] = least_squares(obj, x0).x.reshape(-1, 1)
        return RotoTrans(
            angles=x_tp, angle_sequence=seq, translations=rt_mean[0:3, 3, :]
        )

    def norm(self):
        """
        Compute the RotoTrans Euclidian norm

        Parameters
        ----------
        Returns
        -------
        The norm
        """
        return np.linalg.norm(self[0:3, 0:3, :], axis=(0, 1), ord=2)


class RotoTransCollection(FrameDependentNpArrayCollection):
    """
    List of RotoTrans
    """

    def get_frame(self, f):
        """
        Get fth frame of the collection
        Parameters
        ----------
        f : int
            Frame to get
        Returns
        -------
        Collection of frame f
        """
        coll = RotoTransCollection()
        for element in self:
            coll.append(element.get_frame(f))
        return coll

    def get_rt(self, i):
        """
        Get a specific RotoTrans of the collection
        Parameters
        ----------
        i : int
            Index of the RotoTrans in the collection

        Returns
        -------
        All frame of RotoTrans of index i
        """
        return self[i]

    def get_num_rt(self):
        """
        Get the number of RotoTrans in the collection
        Returns
        -------
        n : int
        Number of RotoTrans in the collection
        """
        return self.get_num_segments()
