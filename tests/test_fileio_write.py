from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import loadmat

from tests._constants import ANALOGS_DATA, MARKERS_DATA


@pytest.mark.parametrize("data", [ANALOGS_DATA, MARKERS_DATA[:, :-1, :]])
@pytest.mark.parametrize("wide", [True, False])
def test_write_csv(data: xr.DataArray, wide: bool):
    temp_filename = Path(".") / "temp.csv"
    data.meca.to_csv(filename=temp_filename, wide=wide)

    if wide:
        if data.ndim > 2:
            return
        newly_created_file = (
            pd.read_csv(temp_filename, index_col="time")
            .stack()
            .to_xarray()
            .rename({"level_1": "channel"})
            .T
        )
    else:
        newly_created_file = pd.read_csv(temp_filename, index_col=data.dims)[
            data.name
        ].to_xarray()
    data, newly_created_file = xr.align(data, newly_created_file)
    np.testing.assert_array_almost_equal(data, newly_created_file, decimal=1)
    temp_filename.unlink()


@pytest.mark.parametrize("data", [ANALOGS_DATA, MARKERS_DATA])
def test_write_matlab(data: xr.DataArray):
    temp_filename = Path(".") / "temp.mat"
    data.meca.to_matlab(filename=temp_filename)
    newly_created_file = loadmat(temp_filename)["data"]
    np.testing.assert_array_equal(data, newly_created_file)
    temp_filename.unlink()
