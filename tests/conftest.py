import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mlb_dfs.data import load_pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
DK = ROOT / "sample_data" / "dk_salaries_sample.csv"
PROJ = ROOT / "sample_data" / "projections_sample.csv"


@pytest.fixture(scope="session")
def pool():
    frame, _ = load_pool(DK, PROJ, verbose=False)
    return frame


@pytest.fixture(scope="session")
def paths():
    return {"dk": DK, "proj": PROJ}
