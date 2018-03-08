import numpy as np
from pyomeca.thirdparty import S2MLib
from pyomeca.types import GeneralizedCoordinate
from pyomeca.show.vtk import Model as PyoModel
from pyomeca.show.vtk import Window as PyoWindow

# Load
m = S2MLib.new("data/pyomecaman.s2mMod")

# Dynamically get the number of markers
nb_markers = S2MLib.nb_markers(m)
print("Number of markers is " + str(nb_markers))

# Dynamically get the number of generalized coordinates
nb_q = S2MLib.nb_q(m)
print("Number of Q is " + str(nb_q))

# Generate some fake data for nb_frames
nb_frames = 200
Q = GeneralizedCoordinate(np.ndarray((nb_q, nb_frames)))
Q[:, :, :] = 0
Q[0, :, :] = np.arange(0, -1, (-1-0)/nb_frames)
Q[-1, :, :] = np.arange(0, -1, (-1-0)/nb_frames)

# Get the markers from these fake generalized coordinates
T = S2MLib.get_markers(m, Q)

# Create a windows with a nice gray background
vtkWindow = PyoWindow(background_color=(.5, .5, .5))
vtkModelReal = PyoModel(vtkWindow, markers_color=(0, 0, 0), markers_size=0.01, markers_opacity=1)

# Show and loop data
i = 0
while vtkWindow.is_active:
    vtkModelReal.update_markers(T.get_frame(i))
    vtkWindow.update_frame()
    i = (i+1) % nb_frames
