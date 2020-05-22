import pytest

from pyomeca import Analogs, Markers
from ._constants import (
    ANALOGS_STO,
    ANALOGS_MOT,
    ANALOGS_CSV,
    MARKERS_TRC,
    EXPECTED_VALUES,
)
from .utils import is_expected_array


def test_read_sto():
    is_expected_array(Analogs.from_sto(ANALOGS_STO), **EXPECTED_VALUES[62])
    with pytest.raises(IndexError):
        Analogs.from_sto(ANALOGS_CSV)


def test_read_mot():
    is_expected_array(Analogs.from_mot(ANALOGS_MOT), **EXPECTED_VALUES[63])


def test_read_trc():
    is_expected_array(Markers.from_trc(MARKERS_TRC), **EXPECTED_VALUES[64])
