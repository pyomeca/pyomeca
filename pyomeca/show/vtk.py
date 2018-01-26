import vtk
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPalette, QColor
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import vtkInteractorStyleTrackballCamera
from pyomeca.types import Vectors3d

first = True
if first:
    app = QtWidgets.QApplication(sys.argv)
    first = False


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None, background_color=(0, 0, 0)):
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
        self.is_active = False
        app._in_event_loop = False
        super()

    def update_frame(self):
        if self.should_reset_camera:
            self.ren.ResetCamera()
            self.should_reset_camera = False
        self.interactor.Render()
        app.processEvents()

    def change_background_color(self, color):
        self.ren.SetBackground(color)
        self.setPalette(QPalette(QColor(color[0]*255, color[1]*255, color[2]*255)))


class Model(QtWidgets.QWidget):
    def __init__(self, parent, markers_size=5, markers_color=(1, 1, 1), markers_opacity=1.0):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent_window = parent

        palette = QPalette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.markers = Vectors3d()
        self.markers_size = markers_size
        self.markers_color = markers_color
        self.markers_opacity = markers_opacity
        self.parent_window.should_reset_camera = True
        self.actors = []

    def set_markers_color(self, markers_color):
        self.markers_color = markers_color
        self.update_markers(self.markers)

    def set_markers_size(self, markers_size):
        self.markers_size = markers_size
        self.update_markers(self.markers)

    def set_markers_opacity(self, markers_opacity):
        self.markers_opacity = markers_opacity
        self.update_markers(self.markers)

    def new_marker_set(self, markers):
        if markers.number_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        self.markers = markers

        # Remove previous actors from the scene
        for actor in self.actors:
            self.parent_window.ren.RemoveActor(actor)
        self.actors = list()

        # Create the geometry of a point (the coordinate) points = vtk.vtkPoints()
        for i in range(markers.number_markers()):
            # Create a mapper
            mapper = vtk.vtkPolyDataMapper()

            # Create an actor
            self.actors.append(vtk.vtkActor())
            self.actors[i].SetMapper(mapper)

            self.parent_window.ren.AddActor(self.actors[i])
            self.parent_window.ren.ResetCamera()
        self.update_markers(self.markers)

    def update_markers(self, markers):
        if markers.number_frames() is not 1:
            raise IndexError("Markers should be from one frame only")
        if markers.number_markers() is not self.markers.number_markers():
            raise IndexError("Numbers of markers should be the same set by new_markers_set")
        self.markers = markers

        for i, actor in enumerate(self.actors):
            # mapper = actors.GetNextActor().GetMapper()
            mapper = actor.GetMapper()
            self.actors[i].GetProperty().SetColor(self.markers_color)
            self.actors[i].GetProperty().SetOpacity(self.markers_opacity)
            source = vtk.vtkSphereSource()
            source.SetCenter(markers[0:3, i])
            source.SetRadius(self.markers_size)
            mapper.SetInputConnection(source.GetOutputPort())



