from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .io import read, utils
from .processing.markers import markers_from_rototrans


class Markers:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray]] = None,
        channels: Optional[list] = None,
        time: Optional[Union[np.array, list, pd.Series]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Markers DataArray with `axis`, `channel` and `time` dimensions used for skin marker positions.
         ![markers](/images/objects/markers.svg)

        Arguments:
            data: Array to be passed to xarray.DataArray
            channels: Channel names
            time: Time vector in seconds associated with the `data` parameter
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Markers `xarray.DataArray` with the specified data and coordinates

        !!! example
            To instantiate a `Markers` with 4 channels and 100 frames filled with some random data:

            ```python
            import numpy as np
            from pyomeca import Markers

            n_axis = 3
            n_channels = 4
            n_frames = 100
            data = np.random.random(size=(n_axis, n_channels, n_frames))
            markers = Markers(data)
            ```

            You can add the channel names:

            ```python
            names = ["A", "B", "C", "D"]
            markers = Markers(data, channels=names)
            ```

            And an associate time vector:

            ```python
            rate = 100  # Hz
            time = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
            markers = Markers(data, channels=names, time=time)
            ```

        !!! note
            Calling `Markers()` generate an empty array.
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0, 0))
        else:
            coords["axis"] = ["x", "y", "z", "ones"]
        if data.shape[0] == 3:
            data = np.insert(data, obj=3, values=1, axis=0)
        if channels:
            coords["channel"] = channels
        if time is not None:
            coords["time"] = time
        return xr.DataArray(
            data=data,
            dims=("axis", "channel", "time"),
            coords=coords,
            name="markers",
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (3, 10, 100), **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk.

        Arguments:
            distribution: Distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random markers `xarray.DataArray` sampled from a given distribution

        !!! example
            To instantiate a `Markers` with some random data sampled from a normal distribution:

            ```python
            from pyomeca import Markers

            n_axis = 3
            n_channels = 10
            n_frames = 100
            size = n_axis, n_channels, n_frames
            markers = Markers.from_random_data(size=size)
            ```

            You can choose any distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions):

            ```python
            markers = Markers.from_random_data(distribution="uniform", size=size, low=1, high=10)
            ```
        """
        return Markers(getattr(np.random, distribution)(size=size, **kwargs).cumsum(-1))

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
        Markers DataArray from a csv file.

        Arguments:
            filename: Any valid string path
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
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
            Markers `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this csv file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/markers.csv),
            type:

            ```python
            from pyomeca import Markers

            data_path = "./tests/data/markers.csv"
            markers = Markers.from_csv(data_path, header=2, first_row=5, first_column=2)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["Daphnee:ASISr", "Daphnee:ASISl", "Daphnee:PSISr"]
            markers = Markers.from_csv(
                data_path, header=2, first_row=5, first_column=2, usecols=channels
            )
            ```

            Or by position:

            ```python
            channels = [5, 6, 7]
            markers = Markers.from_csv(
                data_path, header=2, first_row=5, first_column=2, usecols=channels
            )
            ```

            Sometimes the channel name is delimited by a suffix or prefix.
            To access the prefix, you can specify `prefix_delimiter` and `suffix_delimiter` for the suffix.
            For example, if the name is `"Daphnee:ASISr"` and you specify `suffix_delimiter=":"`, you will select "Daphnee".
            Similarly, if you specify `prefix_delimiter=":":

            ```python
            channels = ["ASISr", "ASISl", "PSISr"]
            markers = Markers.from_csv(
                data_path,
                header=2,
                first_row=5,
                first_column=2,
                usecols=channels,
                prefix_delimiter=":",
            )
            ```

            It is also possible to specify a column containing the time vector:

            ```python
            markers = Markers.from_csv(
                data_path, header=2, first_row=5, first_column=1, time_column=0
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
        Markers DataArray from an Excel file.

        Arguments:
            filename: Any valid string path
            sheet_name: Strings are used for sheet names. Integers are used in zero-indexed sheet positions
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
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
            Markers `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this excel file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/markers.xlsx),
            type:

            ```python
            from pyomeca import Markers

            data_path = "./tests/data/markers.xlsx"
            markers = Markers.from_excel(data_path, header=2, first_row=5, first_column=2)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["boite:gauche_ext"]
            markers = Markers.from_excel(
                data_path, header=2, first_row=5, first_column=2, usecols=channels
            )
            ```

            Or by position:

            ```python
            channels = [1]
            markers = Markers.from_excel(
                data_path, header=2, first_row=5, first_column=2, usecols=channels
            )
            ```

            Sometimes the channel name is delimited by a suffix or prefix.
            To access the prefix, you can specify `prefix_delimiter` and `suffix_delimiter` for the suffix.
            For example, if the name is `"boite:gauche_ext"` and you specify `suffix_delimiter=":"`, you will select "boite".
            Similarly, if you specify `prefix_delimiter=":":

            ```python
            channels = ["gauche_ext"]
            markers = Markers.from_excel(
                data_path,
                header=2,
                first_row=5,
                first_column=2,
                usecols=channels,
                prefix_delimiter=":",
            )
            ```

            It is also possible to specify a column containing the time vector:

            ```python
            markers = Markers.from_excel(
                data_path, header=2, first_row=5, first_column=1, time_column=0
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
    def from_c3d(
        cls,
        filename: Union[str, Path],
        usecols: Optional[List[Union[str, int]]] = None,
        prefix_delimiter: Optional[str] = None,
        suffix_delimiter: Optional[str] = None,
        attrs: Optional[dict] = None,
    ) -> xr.DataArray:
        """
        Markers DataArray from a c3d file.

        Arguments:
            filename: Any valid string path
            usecols: All elements must either be positional or strings that correspond to column names.
                For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
            prefix_delimiter: Delimiter that split each column name by its prefix (we keep only the column name)
            suffix_delimiter: Delimiter that split each column name by its suffix (we keep only the column name)
            attrs: attrs to be passed to xr.DataArray

        Returns:
            Markers `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this c3d file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/markers_analogs.c3d),
            type:

            ```python
            from pyomeca import Markers

            data_path = "./tests/data/markers_analogs.c3d"
            markers = Markers.from_c3d(data_path)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in:

            ```python
            channels = ["Daphnee:ASISl", "Daphnee:PSISr", "Daphnee:PSISl"]
            markers = Markers.from_c3d(data_path, usecols=channels)
            ```

            Sometimes the channel name is delimited by a suffix or prefix.
            To access the prefix, you can specify `prefix_delimiter` and `suffix_delimiter` for the suffix.
            For example, if the name is `""Daphnee:ASISl"` and you specify `suffix_delimiter=":"`, you will select "Daphnee".
            Similarly, if you specify `prefix_delimiter=":":

            ```python
            channels = ["ASISl", "PSISr", "PSISl"]
            markers = Markers.from_c3d(data_path, prefix_delimiter=":")
            ```
        """
        return read.read_c3d(
            cls, filename, usecols, prefix_delimiter, suffix_delimiter, attrs
        )

    @classmethod
    def from_trc(cls, filename: Union[str, Path], **kwargs) -> xr.DataArray:
        """
        Markers DataArray from a trc file.

        Arguments:
            filename: Any valid string path
            kwargs: Keyword arguments to be passed to `from_csv`

        Returns:
            Markers `xarray.DataArray` with the specified data and coordinates

        !!! example
            To read [this trc file](https://github.com/pyomeca/pyomeca/blob/master/tests/data/markers.trc),
            type:

            ```python
            from pyomeca import Markers

            data_path = "./tests/data/markers.trc"
            markers = Markers.from_trc(data_path)
            ```

            If you know the channel names, you can retrieve only the ones you are interested in by specifying strings:

            ```python
            channels = ["STER", "STERl"]
            markers = Markers.from_trc(data_path, usecols=channels)
            ```

            Or by position:

            ```python
            channels = [3, 4]
            markers = Markers.from_trc(data_path, usecols=channels)
            ```
        """
        return read.read_trc(cls, filename, **kwargs)

    @classmethod
    def from_rototrans(cls, markers: xr.DataArray, rt: xr.DataArray) -> xr.DataArray:
        """
        Rotates markers data from a rototrans matrix.

        Arguments:
            markers: markers array to rotate
            rt: Rototrans to rotate about

        Returns:
            A rotated `xarray.DataArray`

        !!! example
            To rotate a random markers set from random angles:

            ```python
            from pyomeca import Angles, Rototrans, Markers

            n_frames = 100
            n_markers = 10

            angles = Angles.from_random_data(size=(3, 1, n_frames))
            rt = Rototrans.from_euler_angles(angles, "xyz")
            markers = Markers.from_random_data(size=(3, n_markers, n_frames))

            rotated_markers = Markers.from_rototrans(markers, rt)
            ```
        """
        return markers_from_rototrans(markers, rt)

    @staticmethod
    def _reshape_flat_array(array: Union[np.array, np.ndarray]) -> xr.DataArray:
        if array.shape[1] % 3 != 0:
            raise IndexError(
                "Array second dimension should be divisible by 3. "
                f"You provided an array with this shape {array.shape}"
            )
        return array.T.reshape((3, int(array.shape[1] / 3), array.shape[0]), order="F")

    @staticmethod
    def _get_requested_channels_from_pandas(
        columns, header, usecols, prefix_delimiter: str, suffix_delimiter: str
    ) -> Tuple[Optional[list], Optional[list]]:
        if usecols:
            idx, channels = [], []
            if isinstance(usecols[0], int):
                for i in usecols:
                    real_idx = i * 3
                    idx.extend([real_idx, real_idx + 1, real_idx + 2])
                    channels.append(
                        utils.col_spliter(
                            columns[real_idx], prefix_delimiter, suffix_delimiter
                        )
                    )
            elif isinstance(usecols[0], str):
                columns_split = [
                    utils.col_spliter(col, prefix_delimiter, suffix_delimiter)
                    for col in columns
                ]
                for col in usecols:
                    i = columns_split.index(col)
                    idx.extend([i, i + 1, i + 2])
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
            if "Unnamed" not in col
        ]
        return channels, None
