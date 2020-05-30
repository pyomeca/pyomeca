from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr
from scipy.io import savemat


def to_wide_dataframe(array: xr.DataArray) -> pd.DataFrame:
    if array.ndim > 2:
        df = array.to_series().unstack().T
        df.columns = ["_".join(col).strip() for col in df.columns]
        return df
    return array.to_series().unstack().T


def write_csv(
    array: xr.DataArray, filename: Union[str, Path], wide: Optional[bool] = True
):
    if wide:
        array.meca.to_wide_dataframe().to_csv(filename)
    else:
        array.to_dataframe().to_csv(filename)


def write_matlab(array: xr.DataArray, filename: Union[str, Path]):
    savemat(filename, array.to_dict())
