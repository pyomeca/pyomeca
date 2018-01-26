from pyomeca.show.vtk import Model as PyoModel
from pyomeca.show.vtk import Window as PyoWindow
from pyomeca import data as PyoData

d = PyoData.load_data("TestDataMarkers.csv", mark_idx=[0, 1, 2, 3, 4, 5])  # all markers
d2 = PyoData.load_data("TestDataMarkers.csv", mark_idx=[[0, 1, 2], [0, 4, 2]])  # mean of 1st and 4th
d3 = PyoData.load_data("TestDataMarkers.csv", mark_idx=[[0], [1], [2]])  # mean of first 3 markers

# Create a windows with a gray background
vtkWindow = PyoWindow(background_color=(.5, .5, .5))

# Add marker holders to the window
vtkModelReal = PyoModel(vtkWindow, markers_color=(1, 0, 0), markers_size=2.5, markers_opacity=1)
vtkModelPred = PyoModel(vtkWindow, markers_color=(0, 0, 0), markers_size=5.0, markers_opacity=.5)
vtkModelMid = PyoModel(vtkWindow, markers_color=(0, 0, 1), markers_size=5.0, markers_opacity=.5)

# Add markers to the marker holders
vtkModelReal.new_marker_set(d[:, :, 0])
vtkModelPred.new_marker_set(d2[:, :, 0])
vtkModelMid.new_marker_set(d3[:, :, 0])

for i in range(d.shape[2]):
    # Update markers
    vtkModelReal.update_markers(d[:, :, i])
    vtkModelPred.update_markers(d2[:, :, i])
    vtkModelMid.update_markers(d3[:, :, i])

    # Funky online update of markers characteristics
    if i > 50:
        vtkModelReal.set_markers_color(((i % 255.)/255., (i % 255.)/255., (i % 255.)/255.))
        vtkModelPred.set_markers_size((i % 150)/50 + 3)
        vtkModelMid.set_markers_opacity((i % 75)/75 + 25)

    # Update window
    vtkWindow.update_frame()
