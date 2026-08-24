# DraftKings MLB Lineup Optimizer

An integer-linear-programming optimizer for DraftKings MLB **Classic** contests,
with correlated team stacking and Monte Carlo lineup scoring.

It doesn't guess. It solves for the best valid lineup under the salary cap and
roster rules, then — optionally — simulates thousands of correlated game
outcomes to judge which of those lineups actually has the upside you want.

## Roster rules it enforces

Every lineup it produces is one DraftKings will accept on upload:

- 10 players: **P, P, C, 1B, 2B, 3B, SS, OF, OF, OF**
- $50,000 salary cap
- Max **5 hitters** from any one team
- Players from at least **2 different games**
- Multi-position eligibility (`1B/3B`, `SS/3B`, …)
- No hitter batting against your own pitcher (toggleable)

## Install

```bash
pip install -r requirements.txt
```

## The two files you need

**1. Your DK salary export** — on DraftKings, open the contest's lineup page and
click "Export to CSV." This is the only source of **player IDs** (required for
bulk upload), **multi-position eligibility**, and the authoritative salaries.

**2. A projections file** — what your DK export cannot tell you: who is
actually in today's lineup, and how today's matchup changes things.

These are not interchangeable. The DK export's `AvgPointsPerGame` is a *season
average*, and optimizing on it will roster players who aren't playing — a
guaranteed zero. The optimizer specifically hunts cheap players with good
numbers, which is exactly where benched players sit. On the included sample
slate, building from `AvgPointsPerGame` alone puts **2 benched players in a
10-man lineup.**

### Projection file formats

Column names are matched by alias, so most sources work with no preprocessing.
A [Daily Fantasy Fuel](https://www.dailyfantasyfuel.com) MLB cheatsheet works
as-is — see `sample_data/dff_cheatsheet_sample.csv`.

| What it means | Accepted column names |
|---|---|
| Player name | `Name`, `Player`, or `first_name` + `last_name` |
| Projection **(required)** | `Proj`, `Projection`, `FPTS`, `Points`, `ppg_projection` |
| Ceiling / floor | `Ceiling`, `Floor` |
| Ownership | `Ownership`, `Own%`, `ownership_projection` |
| Team | `Team`, `TeamAbbrev` |
| Batting order | `Order`, `Batting Order`, `confirmed_order` |
| Injury | `injury_status`, `Status` |
| Confirmed starter | `starting_pitcher` |
| Vegas implied runs | `implied_team_score`, `team_total` |
| Recent form | `L5_fppg_avg`, `L10_fppg_avg`, `szn_fppg_avg` |

Only a name and a projection are required. Everything else unlocks more.

> **Include `Team` if you can.** MLB slates routinely carry two players with the
> same name. Without a team to disambiguate, one player's projection gets
> attached to the other. This tool refuses to guess — it drops them and tells
> you — but a `Team` column just fixes it.

### If your source has no ceiling/floor

Many cheatsheets (DFF included) give a point projection and nothing else. Left
alone, that makes `--mode gpp` and `--mode cash` **silently identical to
`balanced`**, since both objectives fall back to the same number.

So the loader derives them: the 85th/15th percentile of the same shifted
lognormal the simulator uses, with spread from a role-typical coefficient of
variation, widened for players whose recent-form averages disagree with each
other. It tells you when it does this.

That form signal is a rough proxy — an L5 average is noisier than a season
average by construction, so some of the dispersion is sampling noise rather
than real volatility. It ranks players sensibly; don't read it as precise. If
your source supplies real ceiling and floor numbers, those always win.

## Quick start

```bash
# One safe cash lineup (maximize floor)
python dfs_optimizer.py \
  --dk-salaries sample_data/dk_salaries_20260824.csv \
  --projections sample_data/dff_cheatsheet_sample.csv \
  --mode cash --output cash_lineup.csv

# 20 GPP lineups: solver-chosen 4-man stack + 3-man bring-back, confirmed
# lineups only, ranked by simulated upside
python dfs_optimizer.py \
  --dk-salaries sample_data/dk_salaries_20260824.csv \
  --projections sample_data/dff_cheatsheet_sample.csv \
  --mode gpp --num-lineups 20 \
  --stack-size 4 --secondary-stack-size 3 --stack-min-implied 4.5 \
  --require-confirmed-order \
  --rank-by p90 --max-exposure 0.5 --min-unique 4 \
  --output gpp_lineups.csv --summary-csv summary.csv
```

`--output` is written in DK's exact bulk-upload format — the ten roster columns,
each cell `Name (ID)` — so it can be fed straight into DK's "Bulk Upload" entry
screen. Stats live in `--summary-csv` and `--exposure-csv`, because extra
columns can trip DK's importer.

## Who is actually playing

The highest-value thing a projections file gives you. All of these are no-ops if
your source doesn't carry the relevant column.

| Flag | Effect |
|---|---|
| *(default)* | Drops players listed `OUT`/`IL`; keeps and flags `DTD` |
| `--require-confirmed-order` | Only use hitters with a confirmed batting order |
| *(default)* | Drops pitchers not listed as today's starter |
| `--include-injured` | Keep `OUT`/`IL` players anyway |
| `--allow-non-starting-pitchers` | Keep unconfirmed pitchers |
| `--max-batting-order 6` | Cut 7-8-9 hitters |

## Running without a projections file

```bash
python dfs_optimizer.py --dk-salaries DKSalaries.csv --mode balanced
```

Falls back to DK's `AvgPointsPerGame`. This is a **smoke test, not a strategy** —
it cannot know who is playing, ignores matchup entirely, and every entrant in the
contest has the same numbers. It prints a warning saying so.

## Modes

| Mode | Maximizes | Default max hitters/team | Best for |
|---|---|---|---|
| `cash` | **floor** | 4 | 50/50s, double-ups — minimize bust risk |
| `gpp` | **ceiling** | 5 | Tournaments — chase upside |
| `balanced` | **projection** | 5 | General use, single lineup |

## Stacking

Stacking is the highest-leverage decision in MLB DFS: teammates' scores are
correlated, so when a team puts up nine runs, four of your hitters cash at once.

```bash
--stack-size 4                    # 4 hitters from one team; solver picks which
--stack-size 4 --stack-team CHC   # ...or force the team yourself
--secondary-stack-size 3          # plus 3 from a second team (the bring-back)
--stack-min-implied 4.5           # only stack offenses Vegas likes
```

Leaving `--stack-team` off is usually better: the solver evaluates every team's
stack against salary and the rest of the pool, which is a decision it's better
at than you are. `--stack-min-implied` applies to both halves of a double stack.

## Building a portfolio

Twenty near-identical lineups is twenty ways to lose the same way.

| Flag | Effect |
|---|---|
| `--min-unique 4` | Any two lineups differ by at least 4 players |
| `--max-exposure 0.5` | No player appears in more than 50% of lineups |
| `--randomness 0.15` | Jitter projections per lineup, so #2..#N aren't just "next best" |
| `--ownership-leverage 0.05` | Discount chalk to find differentiated builds |
| `--lock` / `--exclude` | Force players in or out by name |

## Simulation

`--simulate` (or any `--rank-by`) scores lineups against thousands of correlated
game outcomes rather than a single ceiling number.

The model is a **Gaussian copula with shifted-lognormal marginals**:

- Each player's marginal is matched to their projected mean and spread, and
  stays right-skewed like real scoring.
- Correlations apply **within a game**, which is where they exist:

| Relationship | Correlation | Why |
|---|---|---|
| Two hitters, same team | **+0.35** | Same rally, same big inning |
| Hitter vs. opposing hitter | **+0.10** | High-total games lift both sides |
| Pitcher vs. opposing hitter | **−0.30** | Your ace dealing means their bats don't |
| Pitcher vs. own hitters | **+0.08** | Run support produces wins |
| Two opposing pitchers | **−0.15** | Only one of them is having a good day |

Those constants are **rank** correlations (the copula's). Realized Pearson
correlation lands a bit lower, because the lognormal transform is monotonic but
not linear. Tune by feel — the signs matter far more than the third decimal.

`--rank-by` builds more candidates than you asked for, simulates them all, and
keeps the best:

```bash
--rank-by p90    # upside — tournaments
--rank-by mean   # expected points
--rank-by p10    # downside protection — the safest build
```

Reported per lineup: `sim_mean`, `sim_p10`, `sim_median`, `sim_p90`, `sim_p99`.
There's deliberately no "max" — the maximum of N draws grows with N, so it
describes your `--sims` setting more than your lineup.

### What the simulation does *not* model

It has no model of **the field**. It tells you how your lineup scores, not how
often it beats 150,000 other entries — and in a large GPP those differ, because
a great lineup that 8% of the field also entered pays much less than a slightly
worse unique one. `--ownership-leverage` is a blunt proxy; real field modeling
is the next thing worth building (see below).

The extreme tail is also approximate: real DFS scoring is a lumpy mixture
(0-fers, then home runs), not a smooth lognormal. The middle of the distribution
is far more trustworthy than the last percentile.

## Full options

```bash
python dfs_optimizer.py --help
```

## Tests

```bash
python -m pytest
```

The suite asserts that every generated lineup satisfies each DK roster rule, and
covers the data-loading traps that cause silent, wrong output — real DK exports
listing pitchers as `SP`/`RP`, accented names, same-name players, blank
ownership columns, and missing ceiling/floor.

## Performance

A full 15-game slate (~510 players) solves in roughly 1.5–2s per lineup. Note
that `--rank-by` with `--oversample 3` solves 3× the lineups you asked for, so
20 sim-ranked lineups is ~60 solves.

## Extending it

- **Field-ownership modeling** — replace `--ownership-leverage` with a real GPP
  equity model: simulate the field's lineups too, and optimize for expected
  payout rather than expected points. This is the biggest remaining gap.
- **Batting-order correlation** — consecutive hitters (3-4-5) correlate more
  tightly than a 2-hole and an 8-hole. The order data is already loaded.
- **Park and weather factors** — scale team run environments before simulating.
- **Late swap** — re-run with locked players for games already started and a
  trimmed pool for the rest.
