import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mlb_dfs.data import load_pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
DK = ROOT / "sample_data" / "dk_salaries_sample.csv"
PROJ = ROOT / "sample_data" / "projections_sample.csv"
DFF_DK = ROOT / "sample_data" / "dk_salaries_20260824.csv"
DFF = ROOT / "sample_data" / "dff_cheatsheet_sample.csv"


@pytest.fixture(scope="session")
def pool():
    frame, _ = load_pool(DK, PROJ, verbose=False)
    return frame


@pytest.fixture(scope="session")
def paths():
    return {"dk": DK, "proj": PROJ}


@pytest.fixture(scope="session")
def dff_paths():
    """A real-shaped Daily Fantasy Fuel cheatsheet + its matching DK export."""
    return {"dk": DFF_DK, "proj": DFF}


@pytest.fixture(scope="session")
def dff_pool(dff_paths):
    frame, _ = load_pool(dff_paths["dk"], dff_paths["proj"], verbose=False)
    return frame
