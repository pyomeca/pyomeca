from biorbd import pyorbdl

import numpy as np

from pyomeca.obj.generalized_coordinates import GeneralizedCoordinate
from pyomeca.obj.markers import Markers3d


def new(path):
    """
    Opens and allocates memory for a model. Please note that the model must be closed by the user to prevent memory
    leaks
    Parameters
    ----------
    path : basestring
        Path of the model
    Returns
    -------
    handler : int64
        Pointer of the model in the RAM
    """
    return pyorbdl.new(path)


def nb_markers(model):
    """
    Get the number of markers in the model
    Parameters
    ----------
    model : int64
        Handler of the model in the RAM returned by the 'new' method
    Returns
    -------
    nb_markers : int
        Number of markers in the model
    """
    return pyorbdl.nb_markers(model)


def nb_q(model):
    """
    Get the number of generalized coordinates in the model
    Parameters
    ----------
    model : int64
        Handler of the model in the RAM returned by the 'new' method
    Returns
    -------
    nb_markers : int
        Number of generalized coordinates in the model
    """
    return pyorbdl.nb_q(model)


def get_markers(model, q):
    """
    Get the markers at specific generalized coordinates
    Parameters
    ----------
    model : int64
        Handler of the model in the RAM returned by the 'new' method
    q : GeneralizedCoordinate
        Generalized coordinates to evaluate the marker positions
    Returns
    -------
    markers : Markers3d
        Position of the markers evaluated at q
    """
    return Markers3d(pyorbdl.get_markers(model, q))


def kalman_kinematics_reconstruction(model, markers, qinit=GeneralizedCoordinate(),
                                     acquisition_frequency=100, noise_factor=1e-10, prediction_factor=1e-5):
    """
    Performs an Extend Kalman Filter (EKF) algorithm to reconstruct kinematics from markers. The EKF performs internally
    integration of the kinematics assuming acceleration = 0, then correct this prediction according to the error.
    Parameters
    ----------
    model : int64
        Handler of the model in the RAM returned by the 'new' method
    markers : Markers3d
        Positions of marker to perform EKF on
    qinit : GeneralizedCoordinate
        Generalized coordinate for one frame giving the initial guess for the reconstruction
    acquisition_frequency : int
        Acquisition frequency of the motion capture system
    noise_factor : int
        Factor of confidence on the measured data
    prediction_factor : int
        Factor of confidence of the prediction.
    Returns
    -------
        q : GeneralizedCoordinate
            Predicted generalized coordinates for the set of given markers
        qdot : GeneralizedCoordinate
            Predicted generalized velocities for the set of given markers
        qddot : GeneralizedCoordinate
            Predicted generalized accelerations for the set of given markers
    """
    if qinit.size == 0:
        qinit = GeneralizedCoordinate(q=np.ndarray((nb_q(model), 1, 1)))

    (q, qdot, qddot) = pyorbdl.kalman_kinematics_reconstruction(model, markers, qinit,
                                                                acquisition_frequency, noise_factor, prediction_factor)
    q = GeneralizedCoordinate(q)
    qdot = GeneralizedCoordinate(qdot)
    qddot = GeneralizedCoordinate(qddot)
    return q, qdot, qddot
