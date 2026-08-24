"""End-to-end runs through the command line entry point."""

import pandas as pd
import pytest

from dfs_optimizer import main
from mlb_dfs.rules import UPLOAD_SLOT_ORDER


def run(tmp_path, paths, *extra):
    out = tmp_path / "lineups.csv"
    code = main([
        "--dk-salaries", str(paths["dk"]),
        "--projections", str(paths["proj"]),
        "--output", str(out),
        *extra,
    ])
    assert code == 0
    run.last_path = out
    return pd.read_csv(out)


def test_output_is_dk_upload_ready(tmp_path, paths):
    """
    DK's bulk importer expects exactly the 10 roster columns, each cell
    "Name (ID)". Extra columns belong in the summary file, not here.
    """
    df = run(tmp_path, paths, "--mode", "cash")
    # Check the bytes on disk: DK wants literal duplicate P/OF headers, which
    # pandas silently renames to P.1/OF.1 when reading back.
    header = run.last_path.read_text().splitlines()[0]
    assert header == ",".join(UPLOAD_SLOT_ORDER)
    assert len(df) == 1
    for cell in df.iloc[0]:
        assert cell.endswith(")") and "(" in cell


def test_multiple_lineups_and_side_reports(tmp_path, paths):
    summary = tmp_path / "s.csv"
    exposure = tmp_path / "e.csv"
    df = run(tmp_path, paths, "--mode", "gpp", "--num-lineups", "5",
             "--stack-size", "4", "--min-unique", "3", "--seed", "1",
             "--summary-csv", str(summary), "--exposure-csv", str(exposure))
    assert len(df) == 5

    s = pd.read_csv(summary)
    assert len(s) == 5
    assert (s["Salary"] <= 50000).all()
    assert (s["Games"] >= 2).all()

    e = pd.read_csv(exposure)
    assert (e["Exposure%"] <= 100).all()
    assert e["Lineups"].sum() == 50  # 5 lineups x 10 players


def test_rank_by_simulation(tmp_path, paths):
    summary = tmp_path / "s.csv"
    df = run(tmp_path, paths, "--mode", "gpp", "--num-lineups", "4",
             "--stack-size", "4", "--rank-by", "p90", "--sims", "2000",
             "--oversample", "3", "--seed", "2", "--summary-csv", str(summary))
    assert len(df) == 4
    s = pd.read_csv(summary)
    for col in ["sim_mean", "sim_p10", "sim_p90"]:
        assert col in s.columns
    assert (s["sim_p10"] < s["sim_p90"]).all()


def test_lock_and_exclude_via_cli(tmp_path, paths):
    df = run(tmp_path, paths, "--lock", "Kevin Silva", "--exclude", "Will Nunez")
    cells = " ".join(str(c) for c in df.iloc[0])
    assert "Kevin Silva" in cells
    assert "Will Nunez" not in cells


def test_infeasible_exits_nonzero(tmp_path, paths, capsys):
    code = main([
        "--dk-salaries", str(paths["dk"]),
        "--projections", str(paths["proj"]),
        "--output", str(tmp_path / "x.csv"),
        "--min-salary-used", "60000",
    ])
    assert code == 1
    assert "No feasible lineup" in capsys.readouterr().err


def test_max_batting_order_without_order_column_warns(tmp_path, paths, capsys):
    run(tmp_path, paths, "--max-batting-order", "5")
    assert "no Order column" in capsys.readouterr().err
