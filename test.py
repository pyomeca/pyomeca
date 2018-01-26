from pyomeca.show.vtk import Model as PyoModel
from pyomeca.show.vtk import Window as PyoWindow
from pyomeca import data as PyoData
d = PyoData.load_data("test/Test.csv")

vtkWindow = PyoWindow(background_color=(.5, .5, .5))
vtkModelReal = PyoModel(vtkWindow, markers_color=(1, 0, 0), markers_size=2.5, markers_opacity=1)
vtkModelPred = PyoModel(vtkWindow, markers_color=(0, 0, 0), markers_size=5.0, markers_opacity=.5)
vtkModelReal.new_marker_set(d[:, :, 0])
vtkModelPred.new_marker_set(d[:, :, 0])

i = 0
while vtkWindow.is_active:
    vtkModelReal.update_markers(d[:, :, i])
    vtkModelPred.update_markers(d[:, :, i])
    # vtkModelReal.set_markers_color(((i % 255.)/255., (i % 255.)/255., (i % 255.)/255.))
    # vtkModelReal.set_markers_size((i % 5.)+2)
    # vtkModelPred.set_markers_opacity((i % 100.0)/100)
    vtkWindow.update_frame()
    i += 1