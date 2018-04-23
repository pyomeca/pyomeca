import numpy as np

from pyomeca.types.frame_dependent import FrameDependentNpArray


class GeneralizedCoordinate(FrameDependentNpArray):
    def __new__(cls, q=np.ndarray((0, 1, 0)), *args, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            nQxNxF matrix of marker positions
        """

        # Reshape if the user sent a 2d instead of 3d shape
        if len(q.shape) == 2:
            q = np.reshape(q, (q.shape[0], 1, q.shape[1]))

        if q.shape[1] != 1:
            raise IndexError('Generalized coordinates can''t have multiple columns')

        return super(GeneralizedCoordinate, cls).__new__(cls, array=q, *args, **kwargs)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, GeneralizedCoordinate):
            return
