"""
Correlated Monte Carlo simulation of MLB DFS outcomes.

A single "ceiling" number can't express why stacking works. Stacking wins
because teammates' scores are *correlated*: when a team puts up 9 runs, four
of your hitters cash together. This module simulates that dependence directly,
so lineups can be judged on their whole outcome distribution.

Model
-----
Gaussian copula with shifted-lognormal marginals:

* Draw correlated standard normals with a block structure (below).
* Map each player's normal onto a lognormal matched to their projected mean
  and spread. The mapping is monotonic, so the rank correlation survives
  exactly, and the marginal stays right-skewed like real fantasy scoring.

Correlation only exists *within a game*, so the matrix is block diagonal -
each game is factorized on its own, which is both faster and better
conditioned than one slate-wide matrix.
"""

import numpy as np

# Within-game correlations, expressed on the copula (i.e. rank correlation).
# Because the lognormal transform is monotonic but non-linear, realized
# *Pearson* correlation comes out somewhat lower than these numbers, while
# Spearman lands right on them. Tune by feel; the signs matter most.
RHO_TEAMMATE_HITTERS = 0.35     # same lineup, same rally
RHO_OPPOSING_HITTERS = 0.10     # high-total games lift both sides ("bring-back")
RHO_PITCHER_VS_OPP_HITTER = -0.30   # your pitcher dealing means their bats don't
RHO_PITCHER_OWN_HITTERS = 0.08  # run support -> wins, and wins score points
RHO_OPPOSING_PITCHERS = -0.15

# Marginal shape defaults when a projections file has no Ceiling/Floor.
DEFAULT_CV_HITTER = 0.85
DEFAULT_CV_PITCHER = 0.45

# Lognormals are positive; shifting lets pitchers post negative scores.
HITTER_SHIFT = 0.0
PITCHER_SHIFT = -5.0

# For a normal, the p85-p15 span is ~2.07 sigma. Used to back a standard
# deviation out of a ceiling/floor pair.
_P85_P15_SPAN = 2.07


def _marginal_params(pool):
    """Per-player (shift, mu, sigma) for the shifted-lognormal marginal."""
    proj = pool["proj"].to_numpy(dtype=float)
    ceiling = pool["ceiling"].to_numpy(dtype=float)
    floor = pool["floor"].to_numpy(dtype=float)
    is_p = pool["is_pitcher"].to_numpy(dtype=bool)

    std = (ceiling - floor) / _P85_P15_SPAN
    # No usable spread (missing ceiling/floor) -> fall back to a default CV.
    cv = np.where(is_p, DEFAULT_CV_PITCHER, DEFAULT_CV_HITTER)
    fallback = np.abs(proj) * cv
    std = np.where(std > 1e-6, std, fallback)
    std = np.maximum(std, 1e-3)

    shift = np.where(is_p, PITCHER_SHIFT, HITTER_SHIFT)
    centered = proj - shift
    # Guard against a non-positive mean after shifting.
    centered = np.maximum(centered, 1e-3)

    var_ratio = 1.0 + (std ** 2) / (centered ** 2)
    sigma = np.sqrt(np.log(var_ratio))
    mu = np.log(centered) - 0.5 * sigma ** 2
    return shift, mu, sigma


def _nearest_psd(matrix):
    """Clip negative eigenvalues so the block can be factorized."""
    sym = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-8, None)
    fixed = (vecs * vals) @ vecs.T
    # Rescale back to unit diagonal so it stays a correlation matrix.
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _game_correlation(teams, opponents, is_pitcher):
    """Correlation matrix for the players in a single game."""
    n = len(teams)
    same_team = teams[:, None] == teams[None, :]
    # a's team is b's opponent
    opposed = (teams[:, None] == opponents[None, :]) & (teams[:, None] != "")

    pit = is_pitcher[:, None]
    pit_t = is_pitcher[None, :]
    hh = ~pit & ~pit_t
    pp = pit & pit_t
    ph = (pit & ~pit_t) | (~pit & pit_t)

    c = np.zeros((n, n))
    c[hh & same_team] = RHO_TEAMMATE_HITTERS
    c[hh & opposed] = RHO_OPPOSING_HITTERS
    c[ph & same_team] = RHO_PITCHER_OWN_HITTERS
    c[ph & opposed] = RHO_PITCHER_VS_OPP_HITTER
    c[pp & opposed] = RHO_OPPOSING_PITCHERS
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 1.0)
    return _nearest_psd(c)


def simulate_players(pool, n_sims=10000, seed=None):
    """
    Simulate every player's fantasy score.

    Returns an array of shape (len(pool), n_sims), row-aligned to `pool`.
    """
    rng = np.random.default_rng(seed)
    n = len(pool)
    z = np.empty((n, n_sims))

    teams = pool["team"].to_numpy(dtype=object).astype(str)
    opponents = pool["opponent"].to_numpy(dtype=object).astype(str)
    is_p = pool["is_pitcher"].to_numpy(dtype=bool)
    games = pool["game_id"].to_numpy(dtype=object).astype(str)

    # Correlation is block diagonal by game: factorize each game separately.
    positions = np.arange(n)
    for game in np.unique(games):
        rows = positions[games == game]
        if len(rows) == 1:
            z[rows[0]] = rng.standard_normal(n_sims)
            continue
        corr = _game_correlation(teams[rows], opponents[rows], is_p[rows])
        chol = np.linalg.cholesky(corr)
        z[rows] = chol @ rng.standard_normal((len(rows), n_sims))

    shift, mu, sigma = _marginal_params(pool)
    return shift[:, None] + np.exp(mu[:, None] + sigma[:, None] * z)


def score_lineups(draws, lineups, pool):
    """
    Total each lineup across every simulation.

    Returns (n_lineups, n_sims). `lineups` holds (player_index, slot) pairs
    using the pool's index labels.
    """
    label_to_row = {label: row for row, label in enumerate(pool.index)}
    out = np.empty((len(lineups), draws.shape[1]))
    for n, lineup in enumerate(lineups):
        rows = [label_to_row[i] for i, _ in lineup]
        out[n] = draws[rows].sum(axis=0)
    return out


def lineup_metrics(scores):
    """
    Summarize one lineup's simulated distribution.

    Deliberately no "max": the maximum of N draws grows with N, so it says more
    about --sims than about the lineup. p99 is the stable version of the same
    question ("how good is the good case?").
    """
    return {
        "sim_mean": float(np.mean(scores)),
        "sim_p10": float(np.percentile(scores, 10)),
        "sim_median": float(np.percentile(scores, 50)),
        "sim_p90": float(np.percentile(scores, 90)),
        "sim_p99": float(np.percentile(scores, 99)),
    }


def rank_lineups(lineup_scores, objective="p90"):
    """
    Score each lineup for portfolio selection.

    "p90"  - upside, for tournaments
    "mean" - expected points, for cash
    "p10"  - downside protection, for the safest possible build
    """
    if objective == "mean":
        return lineup_scores.mean(axis=1)
    if objective == "p10":
        return np.percentile(lineup_scores, 10, axis=1)
    return np.percentile(lineup_scores, 90, axis=1)


def select_portfolio(lineups, lineup_scores, num_lineups, min_unique,
                     objective="p90", max_exposure=1.0):
    """
    Greedily keep the best-simulating lineups that stay diverse.

    Candidates are ranked by the simulated objective, then taken in order,
    skipping any that overlap an already-kept lineup too heavily or that would
    push a player past the exposure cap.

    The cap has to be re-checked here, not just during generation: selecting a
    subset of candidates changes every player's exposure.

    Returns (kept_indices, n_relaxed) where n_relaxed counts lineups that could
    only be filled by ignoring the exposure cap.
    """
    from collections import Counter

    ranks = rank_lineups(lineup_scores, objective)
    order = [int(pos) for pos in np.argsort(-ranks)]
    max_overlap = 10 - min_unique
    cap = None
    if max_exposure < 1.0:
        cap = max(1, int(round(max_exposure * num_lineups)))

    kept, kept_sets, counts = [], [], Counter()

    def try_add(pos, enforce_cap):
        players = set(i for i, _ in lineups[pos])
        if any(len(players & prev) > max_overlap for prev in kept_sets):
            return False
        if enforce_cap and cap is not None:
            if any(counts[i] >= cap for i in players):
                return False
        kept.append(pos)
        kept_sets.append(players)
        counts.update(players)
        return True

    for pos in order:
        if len(kept) >= num_lineups:
            break
        try_add(pos, enforce_cap=True)

    # A strict cap can leave the portfolio short on a thin slate. Backfill
    # rather than silently return fewer lineups - but report how many slots
    # needed the cap relaxed, so an exceeded exposure is never a surprise.
    relaxed = 0
    if len(kept) < num_lineups:
        already = set(kept)
        for pos in order:
            if len(kept) >= num_lineups:
                break
            if pos in already:
                continue
            if try_add(pos, enforce_cap=False):
                relaxed += 1

    return kept, relaxed
