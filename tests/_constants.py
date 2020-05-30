from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to data
from pyomeca import Analogs, Markers

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.style.use("seaborn-ticks")  # bmh, ggplot, seaborn-whitegrid

np.random.seed(42)  # to make the random sampling consistent across runs

if "tests" in f"{Path('.').absolute()}":
    DATA_FOLDER = Path("data")
else:
    DATA_FOLDER = Path("tests") / "data"

MARKERS_CSV = DATA_FOLDER / "markers.csv"
MARKERS_ANALOGS_C3D = DATA_FOLDER / "markers_analogs.c3d"
ANALOGS_CSV = DATA_FOLDER / "analogs.csv"
MARKERS_CSV_WITHOUT_HEADER = DATA_FOLDER / "markers_without_header.csv"
MARKERS_XLSX = DATA_FOLDER / "markers.xlsx"
MARKERS_TRC = DATA_FOLDER / "markers.trc"
ANALOGS_XLSX = DATA_FOLDER / "analogs.xlsx"
ANALOGS_STO = DATA_FOLDER / "inverse_dyn.sto"
ANALOGS_MOT = DATA_FOLDER / "inverse_kin.mot"
EXPECTED_VALUES_CSV = DATA_FOLDER / "is_expected_array_val.csv"

MARKERS_DATA = Markers.from_c3d(
    MARKERS_ANALOGS_C3D,
    usecols=["CLAV_post", "PSISl", "STERr", "CLAV_post"],
    prefix_delimiter=":",
)
ANALOGS_DATA = Analogs.from_c3d(
    MARKERS_ANALOGS_C3D,
    usecols=["EMG1", "EMG10", "EMG11", "EMG12"],
    prefix_delimiter=".",
)

EXPECTED_VALUES = pd.read_csv(
    EXPECTED_VALUES_CSV,
    index_col=[0],
    converters={"shape_val": eval, "first_last_val": eval},
).to_dict("index")
