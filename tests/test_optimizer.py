"""Every lineup must be one DK will actually accept."""

from collections import Counter

import pytest

from mlb_dfs.optimizer import Constraints, generate_lineups
from mlb_dfs.rules import (
    MAX_HITTERS_PER_TEAM,
    MIN_DISTINCT_GAMES,
    ROSTER_SLOTS,
    SALARY_CAP,
    TOTAL_PLAYERS,
)


def assert_dk_legal(pool, lineup):
    """Assert a lineup satisfies every DK MLB Classic roster rule."""
    idxs = [i for i, _ in lineup]

    assert len(idxs) == TOTAL_PLAYERS
    assert len(set(idxs)) == TOTAL_PLAYERS, "a player was rostered twice"

    slots = Counter(slot for _, slot in lineup)
    assert slots == Counter(ROSTER_SLOTS), f"bad slot distribution: {slots}"

    for i, slot in lineup:
        assert slot in pool.at[i, "positions"], (
            f"{pool.at[i, 'name']} is not eligible at {slot}"
        )

    salary = sum(int(pool.at[i, "salary"]) for i in idxs)
    assert salary <= SALARY_CAP, f"over the cap: {salary}"

    hitters = [i for i in idxs if not bool(pool.at[i, "is_pitcher"])]
    per_team = Counter(pool.at[i, "team"] for i in hitters)
    assert max(per_team.values()) <= MAX_HITTERS_PER_TEAM

    games = set(pool.at[i, "game_id"] for i in idxs)
    assert len(games) >= MIN_DISTINCT_GAMES, "DK requires at least 2 games"

    # No hitter facing one of your own pitchers.
    pitcher_opponents = {pool.at[i, "opponent"] for i in idxs
                         if bool(pool.at[i, "is_pitcher"])}
    for i in hitters:
        assert pool.at[i, "team"] not in pitcher_opponents, (
            f"{pool.at[i, 'name']} bats against your own pitcher"
        )


@pytest.mark.parametrize("mode", ["cash", "gpp", "balanced"])
def test_single_lineup_is_legal(pool, mode):
    lineups = generate_lineups(pool, Constraints(mode=mode), 1, seed=1)
    assert len(lineups) == 1
    assert_dk_legal(pool, lineups[0])


def test_portfolio_is_legal_and_diverse(pool):
    c = Constraints(mode="gpp", min_unique_players=4, randomness=0.1)
    lineups = generate_lineups(pool, c, 10, seed=2)
    assert len(lineups) == 10
    for lineup in lineups:
        assert_dk_legal(pool, lineup)
    sets = [set(i for i, _ in l) for l in lineups]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert len(sets[a] & sets[b]) <= TOTAL_PLAYERS - 4


def test_objective_beats_alternatives(pool):
    """The balanced lineup should have the highest projection of the modes."""
    def total(mode):
        lineup = generate_lineups(pool, Constraints(mode=mode), 1, seed=3)[0]
        return sum(float(pool.at[i, "proj"]) for i, _ in lineup)
    assert total("balanced") >= total("cash")
    assert total("balanced") >= total("gpp")


def test_cash_mode_maximizes_floor(pool):
    def total_floor(mode):
        lineup = generate_lineups(pool, Constraints(mode=mode), 1, seed=3)[0]
        return sum(float(pool.at[i, "floor"]) for i, _ in lineup)
    assert total_floor("cash") >= total_floor("gpp")


def test_auto_stack_picks_a_team(pool):
    c = Constraints(mode="gpp", stack_size=5)
    for lineup in generate_lineups(pool, c, 3, seed=4):
        assert_dk_legal(pool, lineup)
        hitters = [i for i, _ in lineup if not bool(pool.at[i, "is_pitcher"])]
        counts = Counter(pool.at[i, "team"] for i in hitters)
        assert max(counts.values()) >= 5


def test_forced_stack_team_is_honored(pool):
    c = Constraints(mode="gpp", stack_team="NYY", stack_size=4)
    for lineup in generate_lineups(pool, c, 3, seed=5):
        assert_dk_legal(pool, lineup)
        hitters = [i for i, _ in lineup if not bool(pool.at[i, "is_pitcher"])]
        assert sum(1 for i in hitters if pool.at[i, "team"] == "NYY") >= 4


def test_secondary_stack_is_a_different_team(pool):
    c = Constraints(mode="gpp", stack_size=4, secondary_stack_size=3)
    for lineup in generate_lineups(pool, c, 3, seed=6):
        assert_dk_legal(pool, lineup)
        hitters = [i for i, _ in lineup if not bool(pool.at[i, "is_pitcher"])]
        counts = Counter(pool.at[i, "team"] for i in hitters).most_common(2)
        assert counts[0][1] >= 4
        assert counts[1][1] >= 3
        assert counts[0][0] != counts[1][0]


def test_locks_and_excludes(pool):
    locked, banned = "Kevin Silva", "Will Nunez"
    c = Constraints(mode="balanced", lock_names=[locked], exclude_names=[banned])
    for lineup in generate_lineups(pool, c, 3, seed=7):
        assert_dk_legal(pool, lineup)
        names = {pool.at[i, "name"] for i, _ in lineup}
        assert locked in names
        assert banned not in names


def test_max_per_team_override(pool):
    c = Constraints(mode="gpp", max_per_team=2)
    for lineup in generate_lineups(pool, c, 3, seed=8):
        hitters = [i for i, _ in lineup if not bool(pool.at[i, "is_pitcher"])]
        assert max(Counter(pool.at[i, "team"] for i in hitters).values()) <= 2


def test_exposure_cap_is_respected(pool):
    c = Constraints(mode="gpp", max_exposure=0.5, min_unique_players=3, randomness=0.1)
    lineups = generate_lineups(pool, c, 10, seed=9)
    counts = Counter()
    for lineup in lineups:
        counts.update(i for i, _ in lineup)
    assert max(counts.values()) <= 5


def test_min_salary_floor(pool):
    c = Constraints(mode="balanced", min_salary_used=49500)
    lineup = generate_lineups(pool, c, 1, seed=10)[0]
    assert 49500 <= sum(int(pool.at[i, "salary"]) for i, _ in lineup) <= SALARY_CAP


def test_opposing_hitter_allowed_when_flag_set(pool):
    """Turning the rule off must actually relax the model, not error."""
    c = Constraints(mode="gpp", no_opposing_pitcher_hitter=False)
    lineups = generate_lineups(pool, c, 1, seed=11)
    assert len(lineups) == 1


def test_infeasible_returns_empty(pool):
    """An impossible salary floor should fail cleanly, not raise."""
    c = Constraints(mode="balanced", min_salary_used=SALARY_CAP + 5000)
    assert generate_lineups(pool, c, 1, seed=12, quiet=True) == []
