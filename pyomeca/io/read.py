from pathlib import Path
from typing import Callable, List, Optional, Union

import ezc3d
import numpy as np
import pandas as pd
import xarray as xr

from .utils import col_spliter, find_end_header_in_opensim_file


def read_c3d(
    caller: Callable,
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    prefix_delimiter: Optional[str] = None,
    suffix_delimiter: Optional[str] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    group = "ANALOG" if caller.__name__ == "Analogs" else "POINT"

    reader = ezc3d.c3d(f"{filename}").c3d_swig
    columns = [
        col_spliter(label, prefix_delimiter, suffix_delimiter)
        for label in reader.parameters()
        .group(group)
        .parameter("LABELS")
        .valuesAsString()
    ]

    get_data_function = getattr(reader, f"get_{group.lower()}s")

    if usecols:
        if isinstance(usecols[0], str):
            idx = [columns.index(channel) for channel in usecols]
        elif isinstance(usecols[0], int):
            idx = usecols
        else:
            raise ValueError(
                "usecols should be None, list of string or list of int."
                f"You provided {type(usecols)}"
            )
        data = get_data_function()[:, idx, :]
        channels = [columns[i] for i in idx]
    else:
        data = get_data_function()
        channels = columns

    data_by_frame = 1 if group == "POINT" else reader.header().nbAnalogByFrame()

    attrs = attrs if attrs else {}
    attrs["first_frame"] = reader.header().firstFrame() * data_by_frame
    attrs["last_frame"] = reader.header().lastFrame() * data_by_frame
    attrs["rate"] = reader.header().frameRate() * data_by_frame
    attrs["units"] = (
        reader.parameters().group(group).parameter("UNITS").valuesAsString()[0]
    )

    time = np.linspace(
        start=0, stop=data.shape[-1] / attrs["rate"], num=data.shape[-1], endpoint=False
    )
    return caller(
        data[0, ...] if group == "ANALOG" else data, channels, time, attrs=attrs
    )


def read_csv_or_excel(
    caller: Callable,
    extension: str,
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    header: Optional[int] = None,
    first_row: int = 0,
    first_column: Optional[Union[str, int]] = None,
    time_column: Optional[Union[str, int]] = None,
    trailing_columns: Optional[Union[str, int]] = None,
    prefix_delimiter: Optional[str] = None,
    suffix_delimiter: Optional[str] = None,
    skip_rows: Optional[List[int]] = None,
    pandas_kwargs: Optional[dict] = None,
    attrs: Optional[dict] = None,
    sheet_name: Union[int, str] = 0,
):
    if skip_rows is None:
        skip_rows = np.arange(header + 1, first_row) if header else np.arange(first_row)

    if pandas_kwargs is None:
        pandas_kwargs = {}

    if extension == "csv":
        data = pd.read_csv(filename, header=header, skiprows=skip_rows, **pandas_kwargs)
    else:
        data = pd.read_excel(
            filename,
            sheet_name=sheet_name,
            header=header,
            skiprows=skip_rows,
            engine="openpyxl",
            **pandas_kwargs,
        )

    if time_column is not None:
        if isinstance(time_column, int):
            time = data.iloc[:, time_column]
            data = data.drop(data.columns[time_column], axis=1)
        elif isinstance(time_column, str):
            time = data[time_column]
            data = data.drop(time_column, axis=1)
        else:
            raise ValueError(
                f"time_column should be str or int. It is {type(time_column)}"
            )
    else:
        time = None

    if first_column:
        data = data.drop(data.columns[:first_column], axis=1)

    if trailing_columns:
        data = data.drop(data.columns[-trailing_columns:], axis=1)

    channels, idx = caller._get_requested_channels_from_pandas(
        data.columns, header, usecols, prefix_delimiter, suffix_delimiter
    )
    data = caller._reshape_flat_array(data.values[:, idx] if idx else data.values)

    attrs = attrs if attrs else {}
    if "rate" in attrs and time is None:
        time = np.arange(
            start=0, stop=data.shape[-1] / attrs["rate"], step=1 / attrs["rate"]
        )
    return caller(data, channels, time, attrs=attrs)


def read_sto_or_mot(
    caller: Callable,
    filename: Union[str, Path],
    end_header: Optional[int] = None,
    **kwargs,
):
    if end_header is None:
        end_header = find_end_header_in_opensim_file(filename)

    data = caller.from_csv(
        filename,
        header=end_header + 1,
        first_column=0,
        time_column=0,
        **kwargs,
    )
    data.attrs["rate"] = (1 / (data.time[1] - data.time[0])).round().item()
    return data


def read_trc(caller: Callable, filename: Union[str, Path], **kwargs):
    data = caller.from_csv(
        filename,
        header=3,
        first_row=6,
        first_column=1,
        time_column=1,
        **kwargs,
    )
    data.attrs["rate"] = (1 / (data.time[1] - data.time[0])).round().item()
    return data
