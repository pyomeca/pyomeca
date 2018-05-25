import numpy as np

from pyomeca.thirdparty import biorbd
from pyomeca import GeneralizedCoordinate
from pyomeca import VtkModel, VtkWindow

# Load
m = biorbd.new("../tests/data/pyomecaman.s2mMod")

# Dynamically get the number of markers
nb_markers = biorbd.nb_markers(m)
print("Number of markers is " + str(nb_markers))

# Dynamically get the number of generalized coordinates
nb_q = biorbd.nb_q(m)
print("Number of Q is " + str(nb_q))

# Generate some fake data for nb_frames
nb_frames = 100
q_simulated = GeneralizedCoordinate(np.ndarray((nb_q, nb_frames)))
q_simulated[:, :, :] = 0  # Put everything to zero
q_simulated[0, :, :] = np.linspace(0, -1, nb_frames)  # Give it some motion
q_simulated[6, :, :] = np.linspace(0, 3.1416, nb_frames)  # And again

# Get the markers from these fake generalized coordinates
T_simulated = biorbd.get_markers(m, q_simulated)

# Reconstruct the kinematics from simulated marker
q_init = q_simulated[:, :, 0]  # Use first position as the initial guess
q_recons, qdot_recons, qddot_recons = biorbd.kalman_kinematics_reconstruction(m, T_simulated)

# Reconstruct marker positions using the Q reconstructed
T_recons = biorbd.get_markers(m, q_recons)

# Create a windows with a nice gray background
window = VtkWindow(background_color=(.5, .5, .5))
h_simulated_T = VtkModel(window, markers_color=(1, 0, 0), markers_size=0.015, markers_opacity=1)
h_reconstructed_T = VtkModel(window, markers_color=(0, 0, 0), markers_size=0.03, markers_opacity=.3)

# Show and loop data
i = 0
while window.is_active:
    h_simulated_T.update_markers(T_simulated.get_frame(i))
    h_reconstructed_T.update_markers(T_recons.get_frame(i))
    window.update_frame()
    i = (i + 1) % nb_frames
