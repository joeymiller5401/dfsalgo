"""Loading, name matching, and the failure modes that used to be silent."""

import pandas as pd
import pytest

from mlb_dfs.data import load_pool, normalize_name


@pytest.mark.parametrize("raw,expected", [
    ("Jose Ramirez", "jose ramirez"),
    ("José Ramírez", "jose ramirez"),      # accents must not break matching
    ("Ronald Acuna Jr.", "ronald acuna"),
    ("Vladimir Guerrero Jr", "vladimir guerrero"),
    ("Michael A. Taylor", "michael a taylor"),
    ("  Aaron Judge  ", "aaron judge"),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_accented_projection_names_still_match(tmp_path, paths):
    """A projections source that writes accents must still merge onto DK."""
    dk = pd.read_csv(paths["dk"])
    proj = pd.read_csv(paths["proj"])
    proj.loc[proj["Name"] == "Jose Cruz", "Name"] = "José Cruz"
    p = tmp_path / "p.csv"
    proj.to_csv(p, index=False)
    pool, report = load_pool(paths["dk"], p, verbose=False)
    assert "Jose Cruz" in set(pool["name"])
    assert report.unmatched_names == []


def test_pitchers_survive_real_dk_position_column(tmp_path, paths):
    """
    Regression: real DK exports use SP/RP in Position. Filtering to Classic
    slots without aliasing dropped every pitcher and made the slate infeasible.
    """
    dk = pd.read_csv(paths["dk"])
    dk["Position"] = dk["Position"].replace({"P": "SP"})
    dk["Roster Position"] = dk["Roster Position"].replace({"P": "SP"})
    path = tmp_path / "dk.csv"
    dk.to_csv(path, index=False)
    pool, _ = load_pool(path, paths["proj"], verbose=False)
    assert int(pool["is_pitcher"].sum()) == 20


def _duplicate_name_slate(tmp_path, paths, with_team):
    """Two different players, same name - the classic MLB merge trap."""
    dk = pd.read_csv(paths["dk"])
    proj = pd.read_csv(paths["proj"])
    # Adam Hall: NYY, proj 7.3 | Jose Ortiz: TOR, proj 7.6
    for frame in (dk, proj):
        frame.loc[frame["Name"] == "Adam Hall", "Name"] = "Luis Garcia"
        frame.loc[frame["Name"] == "Jose Ortiz", "Name"] = "Luis Garcia"
    if with_team:
        proj["Team"] = [
            "NYY" if (n == "Luis Garcia" and p == 7.3)
            else "TOR" if (n == "Luis Garcia" and p == 7.6)
            else None
            for n, p in zip(proj["Name"], proj["Proj"])
        ]
    dk_path, proj_path = tmp_path / "dk.csv", tmp_path / "proj.csv"
    dk.to_csv(dk_path, index=False)
    proj.to_csv(proj_path, index=False)
    return dk_path, proj_path


def test_duplicate_names_resolved_by_team(tmp_path, paths):
    dk_path, proj_path = _duplicate_name_slate(tmp_path, paths, with_team=True)
    pool, _ = load_pool(dk_path, proj_path, verbose=False)
    rows = pool[pool["name"] == "Luis Garcia"].set_index("team")
    assert len(rows) == 2
    # Each must keep its OWN projection, not its namesake's.
    assert rows.loc["NYY", "proj"] == pytest.approx(7.3)
    assert rows.loc["TOR", "proj"] == pytest.approx(7.6)


def test_duplicate_names_without_team_are_dropped_not_guessed(tmp_path, paths):
    """
    Without a Team column the two players can't be told apart. Dropping them
    with a warning beats silently giving one player the other's projection.
    """
    dk_path, proj_path = _duplicate_name_slate(tmp_path, paths, with_team=False)
    pool, report = load_pool(dk_path, proj_path, verbose=False)
    assert pool[pool["name"] == "Luis Garcia"].empty
    assert len(report.ambiguous_names) == 2


def test_unmatched_players_are_reported(tmp_path, paths):
    proj = pd.read_csv(paths["proj"])
    proj = proj[proj["Name"] != "Aaron Judge"]
    proj = proj.iloc[:-3]
    path = tmp_path / "p.csv"
    proj.to_csv(path, index=False)
    _, report = load_pool(paths["dk"], path, verbose=False)
    assert len(report.unmatched_names) == 3


def test_showdown_export_is_rejected(tmp_path, paths):
    dk = pd.read_csv(paths["dk"])
    dk["Roster Position"] = "CPT/UTIL"
    path = tmp_path / "dk.csv"
    dk.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Showdown"):
        load_pool(path, paths["proj"], verbose=False)


def test_opponent_and_game_are_parsed(pool):
    nyy = pool[pool["team"] == "NYY"].iloc[0]
    assert nyy["opponent"] == "BOS"
    assert nyy["game_id"] == "NYY@BOS"
    assert pool["game_id"].nunique() == 5


def test_ceiling_floor_bracket_projection(pool):
    assert (pool["ceiling"] >= pool["proj"]).all()
    assert (pool["floor"] <= pool["proj"]).all()
