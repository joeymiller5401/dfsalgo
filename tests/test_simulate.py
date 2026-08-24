"""The correlated simulation is what justifies stacking - verify the structure."""

import numpy as np
import pytest

from mlb_dfs.optimizer import Constraints, generate_lineups
from mlb_dfs.simulate import (
    lineup_metrics,
    rank_lineups,
    score_lineups,
    select_portfolio,
    simulate_players,
)

SIMS = 20000
SEED = 42


@pytest.fixture(scope="module")
def draws(pool):
    return simulate_players(pool, n_sims=SIMS, seed=SEED)


def _row(pool, name):
    return list(pool["name"]).index(name)


def _corr(draws, pool, a, b):
    return float(np.corrcoef(draws[_row(pool, a)], draws[_row(pool, b)])[0, 1])


def test_shape_and_alignment(pool, draws):
    assert draws.shape == (len(pool), SIMS)


def test_marginal_means_track_projections(pool, draws):
    """Simulated means must reproduce the projections they were built from."""
    error = np.abs(draws.mean(axis=1) - pool["proj"].to_numpy(dtype=float))
    assert error.max() < 0.5


def test_teammates_are_positively_correlated(pool, draws):
    assert _corr(draws, pool, "Adam Hall", "Tony Pratt") > 0.15


def test_pitcher_and_opposing_hitter_are_negatively_correlated(pool, draws):
    # Kevin Silva (NYY) pitches against Tony Reed (BOS).
    assert _corr(draws, pool, "Kevin Silva", "Tony Reed") < -0.1


def test_opposing_hitters_are_mildly_positive(pool, draws):
    """High-total games lift both offenses - this is why bring-backs work."""
    r = _corr(draws, pool, "Adam Hall", "Tony Reed")
    assert 0.0 < r < 0.2


def test_players_in_different_games_are_independent(pool, draws):
    assert abs(_corr(draws, pool, "Adam Hall", "Ryan Grant")) < 0.05


def test_hitters_never_negative_pitchers_can_be(pool, draws):
    is_p = pool["is_pitcher"].to_numpy(dtype=bool)
    assert draws[~is_p].min() >= 0.0
    assert draws[is_p].min() < 0.0


def test_distribution_is_right_skewed(pool, draws):
    """Fantasy scoring has a long right tail; the median sits below the mean."""
    row = _row(pool, "Tony Pratt")
    assert np.median(draws[row]) < np.mean(draws[row])


def test_stacked_lineups_have_more_variance(pool):
    """
    The core claim behind stacking: correlated lineups have fatter tails than
    uncorrelated ones with a comparable projection.
    """
    draws = simulate_players(pool, n_sims=SIMS, seed=SEED)
    stacked = generate_lineups(pool, Constraints(mode="gpp", stack_size=5), 1, seed=1)
    spread = generate_lineups(pool, Constraints(mode="gpp", max_per_team=1), 1, seed=1)
    s_scores = score_lineups(draws, stacked, pool)[0]
    p_scores = score_lineups(draws, spread, pool)[0]
    assert s_scores.std() > p_scores.std()
    # ... and a higher ceiling, which is the whole point in a tournament.
    assert np.percentile(s_scores, 99) > np.percentile(p_scores, 99)


def test_metrics_are_ordered(pool, draws):
    lineups = generate_lineups(pool, Constraints(mode="gpp"), 1, seed=2)
    m = lineup_metrics(score_lineups(draws, lineups, pool)[0])
    assert m["sim_p10"] < m["sim_median"] < m["sim_p90"] < m["sim_p99"]


def test_rank_objectives_differ(pool, draws):
    lineups = generate_lineups(pool, Constraints(mode="gpp", randomness=0.2), 8, seed=3)
    scores = score_lineups(draws, lineups, pool)
    assert not np.array_equal(
        np.argsort(-rank_lineups(scores, "p90")),
        np.argsort(-rank_lineups(scores, "p10")),
    )


def test_select_portfolio_respects_exposure_and_uniqueness(pool, draws):
    from collections import Counter

    lineups = generate_lineups(pool, Constraints(mode="gpp", randomness=0.25), 20, seed=4)
    scores = score_lineups(draws, lineups, pool)
    keep, relaxed = select_portfolio(lineups, scores, 6, min_unique=3,
                                     objective="p90", max_exposure=0.5)
    chosen = [lineups[k] for k in keep]
    assert len(chosen) == 6
    counts = Counter()
    for lineup in chosen:
        counts.update(i for i, _ in lineup)
    # The cap holds unless the selector had to backfill - and backfilling is
    # always reported rather than silently exceeding what the user asked for.
    if relaxed == 0:
        assert max(counts.values()) <= 3
    sets = [set(i for i, _ in l) for l in chosen]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert len(sets[a] & sets[b]) <= 7


def test_exposure_cap_holds_when_candidates_are_plentiful(pool, draws):
    from collections import Counter

    lineups = generate_lineups(pool, Constraints(mode="gpp", randomness=0.35), 40, seed=5)
    scores = score_lineups(draws, lineups, pool)
    keep, relaxed = select_portfolio(lineups, scores, 5, min_unique=3,
                                     objective="p90", max_exposure=0.6)
    assert relaxed == 0
    counts = Counter()
    for k in keep:
        counts.update(i for i, _ in lineups[k])
    assert max(counts.values()) <= 3
