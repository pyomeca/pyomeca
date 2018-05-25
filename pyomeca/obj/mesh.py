import numpy as np

from pyomeca.obj.frame_dependent import FrameDependentNpArrayCollection
from pyomeca.obj.markers import Markers3d


class Mesh(Markers3d):
    def __new__(cls, vertex=np.ndarray((3, 0, 0)), triangles=np.ndarray((0, 3)), *args, **kwargs):
        """
        Parameters
        ----------
        vertex : np.ndarray
            3xNxF matrix of vertex positions
        triangles : np.ndarray, list
            Nx3 indexes matrix where N is the number of triangles and the row are the vertex to connect
        names : list of string
            name of the marker that correspond to second dimension of the positions matrix
        """

        if isinstance(triangles, list):
            triangles = np.array(triangles)

        s = triangles.shape
        if s[1] != 3:
            raise NotImplementedError('Mesh only implements triangle connections')

        obj = super(Mesh, cls).__new__(cls, data=vertex, *args, **kwargs)
        obj.triangles = triangles
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, Mesh):
            return
        self.triangles = getattr(obj, 'triangles')

    # --- Get metadata methods

    def get_num_triangles(self):
        return self.triangles.shape[0]

    def get_num_vertex(self):
        """
        Returns
        -------
        The number of vertex
        """
        return super().get_num_markers()


class MeshCollection(FrameDependentNpArrayCollection):
    """
    List of Mesh
    """

    def append(self, mesh):
        return super().append(mesh)

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
        coll = MeshCollection()
        for element in self:
            coll.append(element.get_frame(f))
        return coll

    def get_mesh(self, i):
        """
        Get a specific Mesh of the collection
        Parameters
        ----------
        i : int
            Index of the Mesh in the collection

        Returns
        -------
        All frame of Mesh of index i
        """
        if i >= len(self):
            return Mesh()

        return self[i]

    # --- Get metadata methods

    def get_num_mesh(self):
        """
        Get the number of Mesh in the collection
        Returns
        -------
        n : int
        Number of Mesh in the collection
        """
        return self.get_num_segments()
