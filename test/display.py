"""
Test and example script for animating models
"""

from pathlib import Path

import numpy as np

from pyomeca import fileio as pyoio
from pyomeca.math.matrix import define_axes
from pyomeca.show.vtk import Model as PyoModel
from pyomeca.show.vtk import Window as PyoWindow
from pyomeca.types import RotoTrans
from pyomeca.types import RotoTransCollection

# Path to data
DATA_FOLDER = Path('.') / 'data'
markers_csv = DATA_FOLDER / 'markers.csv'
markers_analogs_c3d = DATA_FOLDER / 'markers_analogs.c3d'

# Load data
# all markers
d = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                   idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prefix=':')
# mean of 1st and 4th
d2 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                    idx=[[0, 1, 2], [0, 4, 2]], prefix=':')
# mean of first 3 markers
d3 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                    idx=[[0], [1], [2]], prefix=':')

# TODO: replace this with c3d file
d4 = pyoio.read_csv(markers_csv, first_row=5, first_column=2, header=2,
                    idx=[[0], [1], [2]], prefix=':')

# Create a windows with a nice gray background
vtkWindow = PyoWindow(background_color=(.5, .5, .5))

# Add marker holders to the window
vtkModelReal = PyoModel(vtkWindow, markers_color=(1, 0, 0), markers_size=10.0, markers_opacity=1)
vtkModelPred = PyoModel(vtkWindow, markers_color=(0, 0, 0), markers_size=10.0, markers_opacity=.5)
vtkModelMid = PyoModel(vtkWindow, markers_color=(0, 0, 1), markers_size=10.0, markers_opacity=.5)
vtkModelFromC3d = PyoModel(vtkWindow, markers_color=(0, 1, 0), markers_size=10.0, markers_opacity=.5)

# Add markers to the marker holders
vtkModelReal.new_marker_set(d[:, :, 0])
vtkModelPred.new_marker_set(d2[:, :, 0])
vtkModelMid.new_marker_set(d3[:, :, 0])
vtkModelFromC3d.new_marker_set(d4[:, :, 0])

# Create some RotoTrans attached to the first model
all_rt_real = RotoTransCollection()
all_rt_real.append(RotoTrans(angles=[0, 0, 0], angle_sequence="yxz", translations=d[:, 0, 0]))
all_rt_real.append(RotoTrans(angles=[0, 0, 0], angle_sequence="yxz", translations=d[:, 0, 0]))
vtkModelReal.new_rt_set(all_rt_real)

# Create some RotoTrans attached to the second model
all_rt_pred = RotoTransCollection()
all_rt_pred.append(define_axes(d, [3, 5], [[4, 3], [4, 5]], "zx", "z", [3, 4, 5]))
vtkModelPred.new_rt_set(all_rt_pred.get_frame(0))

# Animate all this
i = 0
while vtkWindow.is_active:
    # Update markers
    vtkModelReal.update_markers(d[:, :, i])
    vtkModelPred.update_markers(d2[:, :, i])
    vtkModelMid.update_markers(d3[:, :, i])
    vtkModelFromC3d.update_markers(d4[:, :, i])

    # Funky online update of markers characteristics
    if i > 150:
        vtkModelReal.set_markers_color(((i % 255.) / 255., (i % 255.) / 255., (i % 255.) / 255.))
        vtkModelPred.set_markers_size((i % 150) / 50 + 3)
        vtkModelMid.set_markers_opacity((i % 75) / 75 + 25)

    # Rotate one system of axes
    all_rt_real[0] = RotoTrans(angles=[i / d.number_frames() * np.pi * 2, 0, 0],
                               angle_sequence="yxz", translations=d[:, 0, 0])
    vtkModelReal.update_rt(all_rt_real)

    # Update another system of axes
    vtkModelPred.update_rt(all_rt_pred.get_frame(i))

    # Update window
    vtkWindow.update_frame()
    i = (i + 1) % d.number_frames()
