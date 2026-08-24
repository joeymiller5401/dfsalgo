"""
Daily Fantasy Fuel cheatsheet support.

DFF names almost every column differently from the generic format, supplies no
ceiling/floor, and ships an ownership column that is entirely blank. Each of
those degrades the optimizer quietly if it isn't handled.
"""

import pandas as pd
import pytest

from mlb_dfs.data import filter_availability, load_pool
from mlb_dfs.optimizer import Constraints, generate_lineups

from test_optimizer import assert_dk_legal


def test_split_first_last_name_columns(dff_pool):
    """DFF has first_name/last_name, not Name."""
    assert "Zack Wheeler" in set(dff_pool["name"])
    assert "Pete Crow-Armstrong" in set(dff_pool["name"])


def test_ppg_projection_is_found(dff_pool):
    wheeler = dff_pool[dff_pool["name"] == "Zack Wheeler"].iloc[0]
    assert wheeler["proj"] == pytest.approx(19.6)


def test_everything_matched(dff_paths):
    _, report = load_pool(dff_paths["dk"], dff_paths["proj"], verbose=False)
    assert report.unmatched_names == []
    assert report.matched == 167


def test_empty_ownership_column_is_reported(dff_paths):
    """
    DFF's ownership_projection is blank. Silently treating it as zero makes
    --ownership-leverage a no-op with no explanation.
    """
    pool, report = load_pool(dff_paths["dk"], dff_paths["proj"], verbose=False)
    assert (pool["ownership"] == 0).all()
    assert any("ownership" in note for note in report.notes)


def test_ceiling_and_floor_are_derived(dff_paths):
    pool, report = load_pool(dff_paths["dk"], dff_paths["proj"], verbose=False)
    assert any("ceiling/floor derived" in note for note in report.notes)
    assert (pool["ceiling"] > pool["proj"]).all()
    assert (pool["floor"] < pool["proj"]).all()
    # Pitchers are less volatile than hitters, so their band is tighter.
    p = pool[pool["is_pitcher"]]
    h = pool[~pool["is_pitcher"]]
    p_width = ((p["ceiling"] - p["floor"]) / p["proj"]).mean()
    h_width = ((h["ceiling"] - h["floor"]) / h["proj"]).mean()
    assert p_width < h_width


def test_modes_diverge_without_supplied_ceiling_floor(dff_pool):
    """
    Regression: with ceiling and floor both falling back to the projection,
    gpp/cash/balanced all produced the identical lineup.
    """
    def picks(mode):
        lineup = generate_lineups(dff_pool, Constraints(mode=mode), 1, seed=1)[0]
        return {dff_pool.at[i, "name"] for i, _ in lineup}
    assert picks("gpp") != picks("balanced")
    assert picks("cash") != picks("balanced")


def test_volatile_form_widens_the_distribution(dff_pool):
    """Joc Pederson (L5 0.8 / L10 0.6 / szn 6.1) should get a lower floor
    relative to his projection than steady Steven Kwan (7.2 / 7.7 / 6.2)."""
    joc = dff_pool[dff_pool["name"] == "Joc Pederson"].iloc[0]
    kwan = dff_pool[dff_pool["name"] == "Steven Kwan"].iloc[0]
    assert joc["floor"] / joc["proj"] < kwan["floor"] / kwan["proj"]


def test_confirmed_order_filter(dff_pool):
    filtered = filter_availability(dff_pool, require_confirmed_order=True, verbose=False)
    assert len(filtered) == len(dff_pool) - 27
    hitters = filtered[~filtered["is_pitcher"]]
    assert hitters["batting_order"].notna().all()


def test_out_players_dropped_dtd_kept(dff_paths, tmp_path):
    proj = pd.read_csv(dff_paths["proj"])
    proj.loc[proj["last_name"] == "Buxton", "injury_status"] = "OUT"
    path = tmp_path / "p.csv"
    proj.to_csv(path, index=False)
    pool, _ = load_pool(dff_paths["dk"], path, verbose=False)
    filtered = filter_availability(pool, exclude_injured=True, verbose=False)
    names = set(filtered["name"])
    assert "Byron Buxton" not in names
    assert "Jake Burger" in names          # DTD - risky, but playable


def test_non_starting_pitchers_dropped(dff_paths, tmp_path):
    """A reliever rostered in a P slot is a disaster; DFF flags real starters."""
    proj = pd.read_csv(dff_paths["proj"])
    proj.loc[proj["last_name"] == "Gausman", "starting_pitcher"] = None
    path = tmp_path / "p.csv"
    proj.to_csv(path, index=False)
    pool, _ = load_pool(dff_paths["dk"], path, verbose=False)
    filtered = filter_availability(pool, require_starting_pitcher=True, verbose=False)
    assert "Kevin Gausman" not in set(filtered["name"])
    assert "Zack Wheeler" in set(filtered["name"])


def test_stack_min_implied_gate(dff_pool):
    """Don't stack an offense Vegas expects to be quiet."""
    c = Constraints(mode="gpp", stack_size=4, stack_min_implied=4.5)
    for lineup in generate_lineups(dff_pool, c, 3, seed=2):
        assert_dk_legal(dff_pool, lineup)
        hitters = [i for i, _ in lineup if not bool(dff_pool.at[i, "is_pitcher"])]
        from collections import Counter
        team, count = Counter(dff_pool.at[i, "team"] for i in hitters).most_common(1)[0]
        assert count >= 4
        implied = dff_pool[dff_pool["team"] == team]["implied_team_score"].iloc[0]
        assert implied >= 4.5


def test_lineups_from_dff_are_dk_legal(dff_pool):
    c = Constraints(mode="gpp", stack_size=4, secondary_stack_size=3, randomness=0.1)
    lineups = generate_lineups(dff_pool, c, 5, seed=3)
    assert len(lineups) == 5
    for lineup in lineups:
        assert_dk_legal(dff_pool, lineup)


def test_multi_position_eligibility_from_dk(dff_pool):
    """DFF lists one position; DK's multi-eligibility must win."""
    ramirez = dff_pool[dff_pool["name"] == "Jose Ramirez"].iloc[0]
    assert set(ramirez["positions"]) == {"3B", "SS"}


# --- DK-only fallback -------------------------------------------------------
# Running with no projections file at all, on DK's AvgPointsPerGame.

def test_dk_only_mode_builds_a_legal_lineup(dff_paths):
    pool, report = load_pool(dff_paths["dk"], None, verbose=False)
    assert len(pool) == 167
    lineup = generate_lineups(pool, Constraints(mode="balanced"), 1, seed=1)[0]
    assert_dk_legal(pool, lineup)


def test_dk_only_mode_warns_loudly(dff_paths):
    _, report = load_pool(dff_paths["dk"], None, verbose=False)
    assert any("NO PROJECTIONS FILE" in note for note in report.notes)
    assert any("today's lineup" in note for note in report.notes)


def test_dk_only_mode_cannot_know_who_is_playing(dff_paths):
    """
    The concrete cost of skipping a projections file: season averages happily
    roster players who are not in today's lineup.
    """
    dk_only, _ = load_pool(dff_paths["dk"], None, verbose=False)
    with_proj, _ = load_pool(dff_paths["dk"], dff_paths["proj"], verbose=False)
    playing = set(filter_availability(with_proj, require_confirmed_order=True,
                                      verbose=False)["name"])
    # Nothing in the DK-only pool marks the 27 benched players as unavailable.
    benched = set(with_proj["name"]) - playing
    assert len(benched) == 27
    assert benched & set(dk_only["name"]) == benched


def test_dk_only_mode_still_derives_a_spread(dff_paths):
    pool, report = load_pool(dff_paths["dk"], None, verbose=False)
    assert (pool["ceiling"] > pool["proj"]).all()
    assert any("ceiling/floor derived" in note for note in report.notes)
