from pyomeca.thirdparty.S2MLib import pyorbdl
from pyomeca.types import Markers3d
from pyomeca.types import GeneralizedCoordinate
import numpy as np


def new(path):
    return pyorbdl.new(path)


def nb_markers(model):
    return pyorbdl.nb_markers(model)


def nb_q(model):
    return pyorbdl.nb_q(model)


def get_markers(model, q):
    return Markers3d(pyorbdl.get_markers(model, q))


def kalman_kinematics_reconstruction(model, markers, qinit = GeneralizedCoordinate(),
                                     acquisition_frequency=100, noise_factor=1e-10, prediction_factor=1e-5):
    if qinit.size == 0:
        qinit = GeneralizedCoordinate(q=np.ndarray((nb_q(model), 1, 1)))

    (q, qdot, qddot) = pyorbdl.kalman_kinematics_reconstruction(model, markers, qinit,
                                                                acquisition_frequency, noise_factor, prediction_factor)
    q = GeneralizedCoordinate(q)
    qdot = GeneralizedCoordinate(qdot)
    qddot = GeneralizedCoordinate(qddot)
    return (q, qdot, qddot)
