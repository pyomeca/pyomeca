from pyomeca.thirdparty.S2MLib import pyorbdl
from pyomeca.types import Markers3d


def new(path):
    return pyorbdl.new(path)


def nb_markers(model):
    return pyorbdl.nb_markers(model)


def nb_q(model):
    return pyorbdl.nb_q(model)


def get_markers(model, q):
    return Markers3d(pyorbdl.get_markers(model, q))
