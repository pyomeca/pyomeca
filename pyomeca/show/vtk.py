# -*- coding: utf-8 -*-
"""

Visualization toolkit in pyomeca

"""

import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPalette, QColor
from vtk import vtkInteractorStyleTrackballCamera
from vtk import vtkPolyDataMapper
from vtk import vtkPolyLine
from vtk import vtkCellArray
from vtk import vtkSphereSource
from vtk import vtkActor
from vtk import vtkRenderer
from vtk import vtkLine
from vtk import vtkPoints
from vtk import vtkPolyData
from vtk import vtkUnsignedCharArray
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from pyomeca.types.mesh import Mesh, MeshCollection
from pyomeca.types.rototrans import RotoTrans, RotoTransCollection
from pyomeca.types.markers import Markers3d

first = True
if first:
    app = QtWidgets.QApplication(sys.argv)
    first = False


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None, background_color=(0, 0, 0)):
        """
        Main window
        Parameters
        ----------
        parent
            Qt parent if the main window should be embedded to a parent window
        background_color : tuple(int)
            Color of the background
        """
        QtWidgets.QMainWindow.__init__(self, parent)
        self.frame = QtWidgets.QFrame()

        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.ren = vtkRenderer()
        self.ren.SetBackground(background_color)
        self.vtkWidget.GetRenderWindow().SetSize(1000, 100)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()
        self.change_background_color(background_color)

        self.show()
        app._in_event_loop = True
        self.is_active = True
        self.should_reset_camera = False
        app.processEvents()

    def closeEvent(self, event):
        """
        Things to do when the window is closed
        """
        self.is_active = False
        app._in_event_loop = False
        super()

    def update_frame(self):
        """
        Force the repaint of the window
        """
        if self.should_reset_camera:
            self.ren.ResetCamera()
            self.should_reset_camera = False
        self.interactor.Render()
        app.processEvents()

    def change_background_color(self, color):
        """
        Dynamically change the background color of the windows
        Parameters
        ----------
        color : tuple(int)
        """
        self.ren.SetBackground(color)
        self.setPalette(QPalette(QColor(color[0] * 255, color[1] * 255, color[2] * 255)))


class Model(QtWidgets.QWidget):
    def __init__(self, parent,
                 markers_size=5, markers_color=(1, 1, 1), markers_opacity=1.0,
                 rt_size=25):
        """
        Creates a model that will holds things to plot
        Parameters
        ----------
        parent
            Parent of the Model window
        markers_size : float
            Size the markers should be drawn
        markers_color : Tuple(int)
            Color the markers should be drawn (1 is max brightness)
        markers_opacity : float
            Opacity of the markers (0.0 is completely transparent, 1.0 completely opaque)
        rt_size : int
            Length of the axes of the system of axes
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.parent_window = parent

        palette = QPalette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.markers = Markers3d()
        self.markers_size = markers_size
        self.markers_color = markers_color
        self.markers_opacity = markers_opacity
        self.markers_actors = list()

        self.all_rt = RotoTransCollection()
        self.n_rt = 0
        self.rt_size = rt_size
        self.rt_actors = list()
        self.parent_window.should_reset_camera = True

        self.all_meshes = MeshCollection()
        self.mesh_actors = list()

    def set_markers_color(self, markers_color):
        """
        Dynamically change the color of the markers
        Parameters
        ----------
        markers_color : tuple(int)
            Color the markers should be drawn (1 is max brightness)
        """
        self.markers_color = markers_color
        self.update_markers(self.markers)

    def set_markers_size(self, markers_size):
        """
        Dynamically change the size of the markers
        Parameters
        ----------
        markers_size : float
            Size the markers should be drawn
        """
        self.markers_size = markers_size
        self.update_markers(self.markers)

    def set_markers_opacity(self, markers_opacity):
        """
        Dynamically change the opacity of the markers
        Parameters
        ----------
        markers_opacity : float
            Opacity of the markers (0.0 is completely transparent, 1.0 completely opaque)
        Returns
        -------

        """
        self.markers_opacity = markers_opacity
        self.update_markers(self.markers)

    def new_marker_set(self, markers):
        """
        Define a new marker set. This function must be called each time the number of markers change
        Parameters
        ----------
        markers : Markers3d
            One frame of markers

        """
        if markers.get_num_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        self.markers = markers

        # Remove previous actors from the scene
        for actor in self.markers_actors:
            self.parent_window.ren.RemoveActor(actor)
        self.markers_actors = list()

        # Create the geometry of a point (the coordinate) points = vtk.vtkPoints()
        for i in range(markers.get_num_markers()):
            # Create a mapper
            mapper = vtkPolyDataMapper()

            # Create an actor
            self.markers_actors.append(vtkActor())
            self.markers_actors[i].SetMapper(mapper)

            self.parent_window.ren.AddActor(self.markers_actors[i])
            self.parent_window.ren.ResetCamera()

        # Update marker position
        self.update_markers(self.markers)

    def update_markers(self, markers):
        """
        Update position of the markers on the screen (but do not repaint)
        Parameters
        ----------
        markers : Markers3d
            One frame of markers

        """

        if markers.get_num_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        if markers.get_num_markers() is not self.markers.get_num_markers():
            self.new_marker_set(markers)
            return  # Prevent calling update_markers recursively
        self.markers = markers

        for i, actor in enumerate(self.markers_actors):
            # mapper = actors.GetNextActor().GetMapper()
            mapper = actor.GetMapper()
            self.markers_actors[i].GetProperty().SetColor(self.markers_color)
            self.markers_actors[i].GetProperty().SetOpacity(self.markers_opacity)
            source = vtkSphereSource()
            source.SetCenter(markers[0:3, i])
            source.SetRadius(self.markers_size)
            mapper.SetInputConnection(source.GetOutputPort())

    def new_mesh_set(self, all_meshes):
        """
        Define a new mesh set. This function must be called each time the number of meshes change
        Parameters
        ----------
        mesh : MeshCollection
            One frame of mesh
            :param all_meshes:

        """
        if isinstance(all_meshes, Mesh):
            mesh_tp = MeshCollection()
            mesh_tp.append(all_meshes)
            all_meshes = mesh_tp

        if all_meshes.get_num_frames() is not 1:
            raise IndexError("Mesh should be from one frame only")

        if not isinstance(all_meshes, MeshCollection):
            raise TypeError("Please send a list of mesh to update_mesh")
        self.all_meshes = all_meshes

        # Remove previous actors from the scene
        for actor in self.mesh_actors:
            self.parent_window.ren.RemoveActor(actor)
        self.mesh_actors = list()

        # Create the geometry of a point (the coordinate) points = vtkPoints()
        for (i, mesh) in enumerate(self.all_meshes):
            points = vtkPoints()
            for j in range(mesh.get_num_vertex()):
                points.InsertNextPoint([0, 0, 0])

            # Create an array for each triangle
            cell = vtkCellArray()
            for j in range(mesh.get_num_triangles()):  # For each triangle
                line = vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(4)
                for k in range(len(mesh.triangles[j])):  # For each index
                    line.GetPointIds().SetId(k, mesh.triangles[j, k])
                line.GetPointIds().SetId(3, mesh.triangles[j, 0])  # Close the triangle
                cell.InsertNextCell(line)
            poly_line = vtkPolyData()
            poly_line.SetPoints(points)
            poly_line.SetLines(cell)

            # Create a mapper
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly_line)

            # Create an actor
            self.mesh_actors.append(vtkActor())
            self.mesh_actors[i].SetMapper(mapper)

            self.parent_window.ren.AddActor(self.mesh_actors[i])
            self.parent_window.ren.ResetCamera()

        # Update marker position
        self.update_mesh(self.all_meshes)

    def update_mesh(self, all_meshes):
        """
        Update position of the mesh on the screen (but do not repaint)
        Parameters
        ----------
        all_meshes : MeshCollection
            One frame of mesh

        """
        if isinstance(all_meshes, Mesh):
            mesh_tp = MeshCollection()
            mesh_tp.append(all_meshes)
            all_meshes = mesh_tp

        if all_meshes.get_num_frames() is not 1:
            raise IndexError("Mesh should be from one frame only")

        for i in range(len(all_meshes)):
            if all_meshes.get_mesh(i).get_num_vertex() is not self.all_meshes.get_mesh(i).get_num_vertex():
                self.new_mesh_set(all_meshes)
                return  # Prevent calling update_markers recursively

        if not isinstance(all_meshes, MeshCollection):
            raise TypeError("Please send a list of mesh to update_mesh")

        self.all_meshes = all_meshes

        for (i, mesh) in enumerate(self.all_meshes):
            points = vtkPoints()
            for j in range(mesh.get_num_vertex()):
                points.InsertNextPoint(mesh[0:3, j])

            poly_line = self.mesh_actors[i].GetMapper().GetInput()
            poly_line.SetPoints(points)

    def new_rt_set(self, all_rt):
        """
        Define a new rt set. This function must be called each time the number of rt change
        Parameters
        ----------
        all_rt : RotoTransCollection
            One frame of all RotoTrans to draw

        """
        if isinstance(all_rt, RotoTrans):
            rt_tp = RotoTransCollection()
            rt_tp.append(all_rt[:, :])
            all_rt = rt_tp

        if not isinstance(all_rt, RotoTransCollection):
            raise TypeError("Please send a list of rt to new_rt_set")

        # Remove previous actors from the scene
        for actor in self.rt_actors:
            self.parent_window.ren.RemoveActor(actor)
        self.rt_actors = list()

        for i, rt in enumerate(all_rt):
            if rt.get_num_frames() is not 1:
                raise IndexError("RT should be from one frame only")

            # Create the polyline which will hold the actors
            lines_poly_data = vtkPolyData()

            # Create four points of a generic system of axes
            pts = vtkPoints()
            pts.InsertNextPoint([0, 0, 0])
            pts.InsertNextPoint([1, 0, 0])
            pts.InsertNextPoint([0, 1, 0])
            pts.InsertNextPoint([0, 0, 1])
            lines_poly_data.SetPoints(pts)

            # Create the first line(between Origin and P0)
            line0 = vtkLine()
            line0.GetPointIds().SetId(0, 0)
            line0.GetPointIds().SetId(1, 1)

            # Create the second line(between Origin and P1)
            line1 = vtkLine()
            line1.GetPointIds().SetId(0, 0)
            line1.GetPointIds().SetId(1, 2)

            # Create the second line(between Origin and P1)
            line2 = vtkLine()
            line2.GetPointIds().SetId(0, 0)
            line2.GetPointIds().SetId(1, 3)

            # Create a vtkCellArray container and store the lines in it
            lines = vtkCellArray()
            lines.InsertNextCell(line0)
            lines.InsertNextCell(line1)
            lines.InsertNextCell(line2)

            # Add the lines to the polydata container
            lines_poly_data.SetLines(lines)

            # Create a vtkUnsignedCharArray container and store the colors in it
            colors = vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.InsertNextTuple([255, 0, 0])
            colors.InsertNextTuple([0, 255, 0])
            colors.InsertNextTuple([0, 0, 255])
            lines_poly_data.GetCellData().SetScalars(colors)

            # Create a mapper
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(lines_poly_data)

            # Create an actor
            self.rt_actors.append(vtkActor())
            self.rt_actors[i].SetMapper(mapper)
            self.rt_actors[i].GetProperty().SetLineWidth(5)

            self.parent_window.ren.AddActor(self.rt_actors[i])
            self.parent_window.ren.ResetCamera()

        # Set rt orientations
        self.n_rt = all_rt.get_num_rt()
        self.update_rt(all_rt)

    def update_rt(self, all_rt):
        """
        Update position of the RotoTrans on the screen (but do not repaint)
        Parameters
        ----------
        all_rt : RotoTransCollection
            One frame of all RotoTrans to draw

        """
        if isinstance(all_rt, RotoTrans):
            rt_tp = RotoTransCollection()
            rt_tp.append(all_rt[:, :])
            all_rt = rt_tp

        if all_rt.get_num_rt() is not self.n_rt:
            self.new_rt_set(all_rt)
            return  # Prevent calling update_rt recursively

        if not isinstance(all_rt, RotoTransCollection):
            raise TypeError("Please send a list of rt to new_rt_set")

        self.all_rt = all_rt

        for i, rt in enumerate(self.all_rt):
            if rt.get_num_frames() is not 1:
                raise IndexError("RT should be from one frame only")

            # Update the end points of the axes and the origin
            pts = vtkPoints()
            pts.InsertNextPoint(rt.translation())
            pts.InsertNextPoint(rt.translation() + rt[0:3, 0] * self.rt_size)
            pts.InsertNextPoint(rt.translation() + rt[0:3, 1] * self.rt_size)
            pts.InsertNextPoint(rt.translation() + rt[0:3, 2] * self.rt_size)

            # Update polydata in mapper
            lines_poly_data = self.rt_actors[i].GetMapper().GetInput()
            lines_poly_data.SetPoints(pts)
