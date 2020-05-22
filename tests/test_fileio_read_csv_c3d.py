import numpy as np
import pytest

from pyomeca import Analogs, Markers
from tests._constants import (
    MARKERS_ANALOGS_C3D,
    ANALOGS_CSV,
    MARKERS_CSV,
    MARKERS_CSV_WITHOUT_HEADER,
    ANALOGS_XLSX,
    MARKERS_XLSX,
    EXPECTED_VALUES,
)
from tests.utils import is_expected_array

_extensions = ["c3d", "csv"]
_analogs_cases = [
    {"usecols": None, **EXPECTED_VALUES[10]},
    {"usecols": ["EMG1", "EMG10", "EMG11", "EMG12"], **EXPECTED_VALUES[11]},
    {"usecols": [1, 3, 5, 7], **EXPECTED_VALUES[12]},
    {"usecols": ["EMG1"], **EXPECTED_VALUES[13]},
    {"usecols": [2], **EXPECTED_VALUES[14]},
]
analogs_csv_kwargs = dict(filename=ANALOGS_CSV, header=3, first_row=5, first_column=2)
markers_csv_kwargs = dict(
    filename=MARKERS_CSV, header=2, first_row=5, first_column=2, prefix_delimiter=":"
)

_markers_cases = [
    {"usecols": None, **EXPECTED_VALUES[15]},
    {"usecols": ["CLAV_post", "PSISl", "STERr", "CLAV_post"], **EXPECTED_VALUES[16]},
    {"usecols": [1, 3, 5, 7], **EXPECTED_VALUES[17]},
    {"usecols": ["CLAV_post"], **EXPECTED_VALUES[18]},
    {"usecols": [2], **EXPECTED_VALUES[19]},
]


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    [(d.values()) for d in _analogs_cases],
)
@pytest.mark.parametrize("extension", _extensions)
def test_read_analogs(
    usecols,
    shape_val,
    first_last_val,
    mean_val,
    median_val,
    sum_val,
    nans_val,
    extension,
):
    if extension == "csv":
        data = Analogs.from_csv(**analogs_csv_kwargs, usecols=usecols)
        decimal = 0
    elif extension == "c3d":
        data = Analogs.from_c3d(
            MARKERS_ANALOGS_C3D, prefix_delimiter=".", usecols=usecols
        )
        decimal = 4
    else:
        raise ValueError("wrong extension provided")

    if usecols and isinstance(usecols[0], str):
        np.testing.assert_array_equal(x=data.channel, y=usecols)

    is_expected_array(
        data,
        shape_val,
        first_last_val,
        mean_val,
        median_val,
        sum_val,
        nans_val,
        decimal=decimal,
    )


@pytest.mark.parametrize("usecols", [[20.0]])
@pytest.mark.parametrize("extension", _extensions)
def test_read_catch_error(
    usecols, extension,
):
    with pytest.raises(IndexError):
        Markers.from_csv(MARKERS_CSV)

    if extension == "csv":
        with pytest.raises(ValueError):
            Analogs.from_csv(**analogs_csv_kwargs, usecols=usecols)
        with pytest.raises(ValueError):
            Markers.from_csv(**markers_csv_kwargs, usecols=usecols)
    elif extension == "c3d":
        with pytest.raises(ValueError):
            Analogs.from_c3d(MARKERS_ANALOGS_C3D, usecols=usecols)

    reader = getattr(Markers, f"from_{extension}")
    with pytest.raises(ValueError):
        reader(MARKERS_ANALOGS_C3D, usecols=usecols)


def test_csv_last_column_to_remove():
    last_column_to_remove = 5
    ref = Analogs.from_csv(**analogs_csv_kwargs).channel[:-last_column_to_remove]
    without_last_columns = Analogs.from_csv(
        **analogs_csv_kwargs, last_column_to_remove=last_column_to_remove
    ).channel
    np.testing.assert_array_equal(x=ref, y=without_last_columns)


def test_csv_edge_cases():
    time_with_rate = Analogs.from_csv(**analogs_csv_kwargs, attrs={"rate": 2000}).time
    assert time_with_rate[-1] == 5.7995

    time_column_with_id = Analogs.from_csv(**analogs_csv_kwargs, time_column="Frame")

    time_column_with_name = Analogs.from_csv(**analogs_csv_kwargs, time_column=0)

    np.testing.assert_array_equal(time_column_with_id.time, time_column_with_name.time)
    np.testing.assert_array_equal(time_column_with_id, time_column_with_name)

    with pytest.raises(ValueError):
        Analogs.from_csv(**analogs_csv_kwargs, time_column=[20.0])

    assert Analogs.from_csv(
        **analogs_csv_kwargs, prefix_delimiter="G", suffix_delimiter="M"
    ).shape == (38, 11600)


def test_csv_without_header():
    is_expected_array(
        Analogs.from_csv(ANALOGS_CSV, first_row=5, first_column=2),
        **EXPECTED_VALUES[59],
    )

    is_expected_array(
        Markers.from_csv(MARKERS_CSV_WITHOUT_HEADER, first_column=2),
        **EXPECTED_VALUES[58],
    )


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    [(d.values()) for d in _markers_cases],
)
@pytest.mark.parametrize("extension", _extensions)
def test_read_markers(
    usecols,
    shape_val,
    first_last_val,
    mean_val,
    median_val,
    sum_val,
    nans_val,
    extension,
):
    if extension == "csv":
        data = Markers.from_csv(**markers_csv_kwargs, usecols=usecols)
        decimal = 0
    elif extension == "c3d":
        data = Markers.from_c3d(
            MARKERS_ANALOGS_C3D, prefix_delimiter=":", usecols=usecols
        )
        decimal = 4
    else:
        raise ValueError("wrong extension provided")

    if usecols and isinstance(usecols[0], str):
        np.testing.assert_array_equal(x=data.channel, y=usecols)

    is_expected_array(
        data,
        shape_val,
        first_last_val,
        mean_val,
        median_val,
        sum_val,
        nans_val,
        decimal=decimal,
    )


def test_read_xlsx():
    is_expected_array(
        Markers.from_excel(**{**markers_csv_kwargs, **dict(filename=MARKERS_XLSX)}),
        **EXPECTED_VALUES[60],
    )
    is_expected_array(
        Analogs.from_excel(**{**analogs_csv_kwargs, **dict(filename=ANALOGS_XLSX)}),
        **EXPECTED_VALUES[61],
    )
