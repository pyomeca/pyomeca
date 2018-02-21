# -*- coding: utf-8 -*-
"""

Visualization toolkit in pyomeca

"""

import vtk
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPalette, QColor
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import vtkInteractorStyleTrackballCamera
from vtk import vtkPolyData
from vtk import vtkPoints
from vtk import vtkLine
from vtk import vtkCellArray
from vtk import vtkUnsignedCharArray
from pyomeca.types import Markers3d
from pyomeca.types import RotoTransCollection

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

        self.ren = vtk.vtkRenderer()
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
        self.setPalette(QPalette(QColor(color[0]*255, color[1]*255, color[2]*255)))


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
        self.rt_size = rt_size
        self.rt_actors = list()
        self.parent_window.should_reset_camera = True

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
        if markers.n_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        self.markers = markers

        # Remove previous actors from the scene
        for actor in self.markers_actors:
            self.parent_window.ren.RemoveActor(actor)
        self.markers_actors = list()

        # Create the geometry of a point (the coordinate) points = vtk.vtkPoints()
        for i in range(markers.n_markers()):
            # Create a mapper
            mapper = vtk.vtkPolyDataMapper()

            # Create an actor
            self.markers_actors.append(vtk.vtkActor())
            self.markers_actors[i].SetMapper(mapper)

            self.parent_window.ren.AddActor(self.markers_actors[i])
            self.parent_window.ren.ResetCamera()
        self.update_markers(self.markers)

    def update_markers(self, markers):
        """
        Update position of the markers on the screen (but do not repaint)
        Parameters
        ----------
        markers : Markers3d
            One frame of markers

        """
        if markers.n_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        if markers.n_markers() is not self.markers.n_markers():
            raise IndexError("Numbers of markers should be the same set by new_markers_set")
        self.markers = markers

        for i, actor in enumerate(self.markers_actors):
            # mapper = actors.GetNextActor().GetMapper()
            mapper = actor.GetMapper()
            self.markers_actors[i].GetProperty().SetColor(self.markers_color)
            self.markers_actors[i].GetProperty().SetOpacity(self.markers_opacity)
            source = vtk.vtkSphereSource()
            source.SetCenter(markers[0:3, i])
            source.SetRadius(self.markers_size)
            mapper.SetInputConnection(source.GetOutputPort())

    def new_rt_set(self, all_rt):
        """
        Define a new rt set. This function must be called each time the number of rt change
        Parameters
        ----------
        all_rt : RotoTransCollection
            One frame of all RotoTrans to draw

        """

        if not isinstance(all_rt, RotoTransCollection):
            raise TypeError("Please send a list of rt to new_rt_set")

        # Remove previous actors from the scene
        for actor in self.rt_actors:
            self.parent_window.ren.RemoveActor(actor)
        self.rt_actors = list()

        for i, rt in enumerate(all_rt):
            if rt.n_frames() is not 1:
                raise IndexError("Markers should be from one frame only")

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
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(lines_poly_data)

            # Create an actor
            self.rt_actors.append(vtk.vtkActor())
            self.rt_actors[i].SetMapper(mapper)
            self.rt_actors[i].GetProperty().SetLineWidth(5)

            self.parent_window.ren.AddActor(self.rt_actors[i])
            self.parent_window.ren.ResetCamera()

        # Set rt orientations
        self.update_rt(all_rt)

    def update_rt(self, all_rt):
        """
        Update position of the RotoTrans on the screen (but do not repaint)
        Parameters
        ----------
        all_rt : RotoTransCollection
            One frame of all RotoTrans to draw

        """
        if not isinstance(all_rt, RotoTransCollection):
            raise TypeError("Please send a list of rt to new_rt_set")

        self.all_rt = all_rt

        for i, rt in enumerate(self.all_rt):
            # Update the end points of the axes and the origin
            pts = vtkPoints()
            pts.InsertNextPoint(rt.translation())
            pts.InsertNextPoint(rt.translation() + rt[0:3, 0] * self.rt_size)
            pts.InsertNextPoint(rt.translation() + rt[0:3, 1] * self.rt_size)
            pts.InsertNextPoint(rt.translation() + rt[0:3, 2] * self.rt_size)

            # Update polydata in mapper
            lines_poly_data = self.rt_actors[i].GetMapper().GetInput()
            lines_poly_data.SetPoints(pts)

