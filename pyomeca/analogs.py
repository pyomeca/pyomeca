from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .io import read, utils


class Analogs:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray]] = None,
        channels: Optional[list] = None,
        time: Optional[Union[np.array, list, pd.Series]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Analogs DataArray with `channel` and `time` dimensions
         used for generic signals such as EMGs, force signals or any other analog signals.
         ![analogs](/images/objects/analogs.svg)

        Arguments:
            data: Array to be passed to xarray.DataArray
            channels: Channel names
            time: Time vector in seconds associated with the `data` parameter
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To instantiate an `Analogs` with 4 channels and 100 frames filled with some random data:

            ```python
            import numpy as np
            from pyomeca import Analogs

            n_channels = 4
            n_frames = 100
            data = np.random.random(size=(n_channels, n_frames))
            analogs = Analogs(data)
            ```

            You can add the channel names:

            ```python
            names = ["A", "B", "C", "D"]
            analogs = Analogs(data, channels=names)
            ```

            And an associate time vector:

            ```python
            rate = 100  # Hz
            time = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
            analogs = Analogs(data, channels=names, time=time)
            ```

        !!! note
            Calling `Analogs()` generate an empty array.
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0))
        if channels is not None:
            coords["channel"] = channels
        if time is not None:
            coords["time"] = time
        return xr.DataArray(
            data=data,
            dims=("channel", "time"),
            coords=coords,
            name="analogs",
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (10, 100), **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk.

        Arguments:
            distribution: Distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random Analogs `xarray.DataArray` sampled from a given distribution

        !!! example
            To instantiate an `Analogs` with some random data sampled from a normal distribution:

            ```python
            from pyomeca import Analogs

            n_channels = 10
            n_frames = 100
            size = n_channels, n_frames
            analogs = Analogs.from_random_data(size=size)
            ```

            You can choose any distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions):

            ```python
            analogs = Analogs.from_random_data(distribution="uniform", size=size, low=1, high=10)
            ```
        """
        return Analogs(getattr(np.random, distribution)(size=size, **kwargs).cumsum(-1))

    @classmethod
    def from_csv(
        cls,
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
    ) -> xr.DataArray:
        """
        Analogs DataArray from a csv file.

        Arguments:
            filename: Any valid string path
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz']
            header: Row of the header (0-indexed)
            first_row: First row of the data (0-indexed)
            first_column: First column of the data (0-indexed)
            time_column: Location of the time column. If None, indices are associated
            trailing_columns: If for some reason the csv reads extra columns, how many should be ignored
            prefix_delimiter: Delimiter that split each column name by its prefix (we keep only the column name)
            suffix_delimiter: Delimiter that split each column name by its suffix (we keep only the column name)
            skip_rows: Line numbers to skip (0-indexed)
            pandas_kwargs: Keyword arguments to be passed to `pandas.read_csv`
            attrs: attrs to be passed to `xr.DataArray`. If attrs['rate'] is provided, compute the time accordingly

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this csv file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/analogs.csv),
            type:

            ```python
            from pyomeca import Analogs

            data_path = "./tests/data/analogs.csv"
            analogs = Analogs.from_csv(data_path, header=3, first_row=5, first_column=2)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["IM EMG1", "IM EMG2", "IM EMG3"]
            analogs = Analogs.from_csv(
                data_path, header=3, first_row=5, first_column=2, usecols=channels
            )
            ```

            Or by position:

            ```python
            channels = [5, 6, 7]
            analogs = Analogs.from_csv(
                data_path, header=3, first_row=5, first_column=2, usecols=channels
            )
            ```

            Sometimes the channel name is delimited by a suffix or prefix.
            To access the prefix, you can specify `prefix_delimiter` and `suffix_delimiter` for the suffix.
            For example, if the name is `"IM EMG1"` and you specify `suffix_delimiter=" "`, you will select "IM".
            Similarly, if you specify `prefix_delimiter=" ":

            ```python
            channels = ["EMG1", "EMG2", "EMG3"]
            analogs = Analogs.from_csv(
                data_path,
                header=3,
                first_row=5,
                first_column=2,
                usecols=channels,
                suffix_delimiter=" ",
            )
            ```

            It is also possible to specify a column containing the time vector:

            ```python
            analogs = Analogs.from_csv(
                data_path, header=3, first_row=5, first_column=1, time_column=0
            )
            ```
        """
        return read.read_csv_or_excel(
            cls,
            "csv",
            filename,
            usecols,
            header,
            first_row,
            first_column,
            time_column,
            trailing_columns,
            prefix_delimiter,
            suffix_delimiter,
            skip_rows,
            pandas_kwargs,
            attrs,
        )

    @classmethod
    def from_excel(
        cls,
        filename: Union[str, Path],
        sheet_name: Union[str, int] = 0,
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
    ) -> xr.DataArray:
        """
        Analogs DataArray from a excel file.

        Arguments:
            filename: Any valid string path
            sheet_name: Strings are used for sheet names. Integers are used in zero-indexed sheet positions
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz']
            header: Row of the header (0-indexed)
            first_row: First row of the data (0-indexed)
            first_column: First column of the data (0-indexed)
            time_column: Location of the time column. If None, indices are associated
            trailing_columns: If for some reason the csv reads extra columns, how many should be ignored
            prefix_delimiter: Delimiter that split each column name by its prefix (we keep only the column name)
            suffix_delimiter: Delimiter that split each column name by its suffix (we keep only the column name)
            skip_rows: Line numbers to skip (0-indexed)
            pandas_kwargs: Keyword arguments to be passed to `pandas.read_excel`
            attrs: attrs to be passed to `xr.DataArray`. If attrs['rate'] is provided, compute the time accordingly

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this excel file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/analogs.xlsx),
            type:

            ```python
            from pyomeca import Analogs

            data_path = "./tests/data/analogs.xlsx"
            analogs = Analogs.from_excel(data_path, header=3, first_row=5, first_column=2)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["A"]
            analogs = Analogs.from_excel(
                data_path, header=3, first_row=5, first_column=2, usecols=channels
            )
            ```

            Or by position:

            ```python
            channels = [1]
            analogs = Analogs.from_excel(
                data_path, header=3, first_row=5, first_column=2, usecols=channels
            )
            ```

            It is also possible to specify a column containing the time vector:

            ```python
            analogs = Analogs.from_excel(
                data_path, header=3, first_row=5, first_column=1, time_column=0
            )
            ```

        """
        return read.read_csv_or_excel(
            cls,
            "excel",
            filename,
            usecols,
            header,
            first_row,
            first_column,
            time_column,
            trailing_columns,
            prefix_delimiter,
            suffix_delimiter,
            skip_rows,
            pandas_kwargs,
            attrs,
            sheet_name,
        )

    @classmethod
    def from_sto(
        cls, filename: Union[str, Path], end_header: Optional[bool] = None, **kwargs
    ) -> xr.DataArray:
        """
        Analogs DataArray from a sto file.

        Arguments:
            filename: Any valid string path
            end_header: Index where `endheader` appears (0 indexed).
                If not provided, the index is automatically determined
            kwargs: Keyword arguments to be passed to `from_csv`

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this sto file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/inverse_dyn.sto),
            type:

            ```python
            from pyomeca import Analogs

            data_path = "./tests/data/inverse_dyn.sto"
            analogs = Analogs.from_sto(data_path)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["shoulder_plane_moment", "shoulder_ele_moment"]
            analogs = Analogs.from_sto(data_path, usecols=channels)
            ```

            Or by position:

            ```python
            channels = [3, 4]
            analogs = Analogs.from_sto(data_path, usecols=channels)
            ```
        """
        return read.read_sto_or_mot(cls, filename, end_header, **kwargs)

    @classmethod
    def from_mot(
        cls, filename: Union[str, Path], end_header: Optional[bool] = None, **kwargs
    ) -> xr.DataArray:
        """
        Analogs DataArray from a mot file.

        Arguments:
            filename: Any valid string path
            end_header: Index where `endheader` appears (0 indexed). If not provided, the index is automatically determined.
            kwargs: Keyword arguments to be passed to `from_csv`

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this mot file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/inverse_kin.mot),
            type:

            ```python
            from pyomeca import Analogs

            data_path = "./tests/data/inverse_kin.mot"
            analogs = Analogs.from_mot(data_path)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["elbow_flexion", "pro_sup"]
            analogs = Analogs.from_mot(data_path, usecols=channels)
            ```

            Or by position:

            ```python
            channels = [3, 4]
            analogs = Analogs.from_mot(data_path, usecols=channels)
            ```
        """
        return read.read_sto_or_mot(cls, filename, end_header, **kwargs)

    @classmethod
    def from_c3d(
        cls,
        filename: Union[str, Path],
        usecols: Optional[List[Union[str, int]]] = None,
        prefix_delimiter: Optional[str] = None,
        suffix_delimiter: Optional[str] = None,
        attrs: Optional[dict] = None,
    ) -> xr.DataArray:
        """
        Analogs DataArray from a c3d file.

        Arguments:
            filename: Any valid string path
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
            prefix_delimiter: Delimiter that split each column name by its prefix (we keep only the column name)
            suffix_delimiter: Delimiter that split each column name by its suffix (we keep only the column name)
            attrs: attrs to be passed to xr.DataArray

        Returns:
            Analogs `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this c3d file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/markers_analogs.c3d),
            type:

            ```python
            from pyomeca import Analogs

            data_path = "./tests/data/markers_analogs.c3d"
            analogs = Analogs.from_c3d(data_path)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in:

            ```python
            channels = ["Voltage.1", "Voltage.2", "Voltage.3"]
            analogs = Analogs.from_c3d(data_path, usecols=channels)
            ```

            Sometimes the channel name is delimited by a suffix or prefix.
            To access the prefix, you can specify `prefix_delimiter` and `suffix_delimiter` for the suffix.
            For example, if the name is `"Voltage.1"` and you specify `suffix_delimiter="."`, you will select "Voltage".
            Similarly, if you specify `prefix_delimiter=".":

            ```python
            channels = ["1", "2", "3"]
            analogs = Analogs.from_c3d(data_path, usecols=channels, prefix_delimiter=".")
            ```
        """
        return read.read_c3d(
            cls, filename, usecols, prefix_delimiter, suffix_delimiter, attrs
        )

    @staticmethod
    def _reshape_flat_array(array: Union[np.array, np.ndarray]) -> xr.DataArray:
        """
        Takes a tabular numpy array (frames x N) and return a (N x frames) numpy array

        Arguments:
            array: A tabular array (frames x N) with N = 3 x marker

        Returns:
            Reshaped Analogs `xarray.DataArray`
        """
        return array.T

    @staticmethod
    def _get_requested_channels_from_pandas(
        columns, header, usecols, prefix_delimiter: str, suffix_delimiter: str
    ) -> Tuple[Optional[list], Optional[list]]:
        if usecols:
            idx, channels = [], []
            if isinstance(usecols[0], int):
                for i in usecols:
                    idx.append(i)
                    channels.append(
                        utils.col_spliter(
                            columns[i], prefix_delimiter, suffix_delimiter
                        )
                    )
            elif isinstance(usecols[0], str):
                columns_split = [
                    utils.col_spliter(col, prefix_delimiter, suffix_delimiter)
                    for col in columns
                ]
                for col in usecols:
                    idx.append(columns_split.index(col))
                    channels.append(col)
            else:
                raise ValueError(
                    "usecols should be None, list of string or list of int."
                    f"You provided {type(usecols)}"
                )
            return channels, idx

        if header is None:
            return None, None

        channels = [
            utils.col_spliter(col, prefix_delimiter, suffix_delimiter)
            for col in columns
        ]
        return channels, None
