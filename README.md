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

## What you need to provide

**1. Your DK salary export.** On DraftKings, open the contest's lineup page and
click "Export to CSV."

**2. Your projections CSV**, one row per player:

```
Name,Proj,Ceiling,Floor,Ownership,Team,Order
Aaron Judge,11.2,24.1,3.4,28.5,NYY,2
```

Only `Name` and `Proj` are required. The rest unlock more:

| Column | Unlocks |
|---|---|
| `Ceiling`, `Floor` | GPP/cash objectives, and the spread used by the simulator |
| `Ownership` | `--ownership-leverage` for contrarian builds |
| `Team` | **Correctly separating two players with the same name** |
| `Order` | `--max-batting-order` to cut 8- and 9-hole hitters |

> **Include `Team` if you can.** MLB slates routinely carry two players with the
> same name. Without a team to disambiguate, one player's projection gets
> attached to the other. This tool refuses to guess — it drops them and tells
> you — but a `Team` column just fixes it.

Sample files are in `sample_data/` so you can try it immediately.

## Quick start

```bash
# One safe cash lineup (maximize floor)
python dfs_optimizer.py \
  --dk-salaries sample_data/dk_salaries_sample.csv \
  --projections sample_data/projections_sample.csv \
  --mode cash --output cash_lineup.csv

# 20 GPP lineups: solver-chosen 4-man stack + 3-man bring-back,
# ranked by simulated upside, nobody in more than half the lineups
python dfs_optimizer.py \
  --dk-salaries sample_data/dk_salaries_sample.csv \
  --projections sample_data/projections_sample.csv \
  --mode gpp --num-lineups 20 \
  --stack-size 4 --secondary-stack-size 3 \
  --rank-by p90 --max-exposure 0.5 --min-unique 4 \
  --output gpp_lineups.csv --summary-csv summary.csv
```

`--output` is written in DK's exact bulk-upload format — the ten roster columns,
each cell `Name (ID)` — so it can be fed straight into DK's "Bulk Upload" entry
screen. Stats live in `--summary-csv` and `--exposure-csv`, because extra
columns can trip DK's importer.

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
--stack-size 4 --stack-team NYY   # ...or force the team yourself
--secondary-stack-size 3          # plus 3 from a second team (the bring-back)
```

Leaving `--stack-team` off is usually better: the solver evaluates every team's
stack against salary and the rest of the pool, which is a decision it's better
at than you are.

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

- Each player's marginal is matched to their projected mean and to the spread
  implied by their `Ceiling`/`Floor`, and stays right-skewed like real scoring.
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
listing pitchers as `SP`/`RP`, accented names, and same-name players.

## Performance

A full 15-game slate (~510 players) solves in roughly 1.5–2s per lineup. Note
that `--rank-by` with `--oversample 3` solves 3× the lineups you asked for, so
20 sim-ranked lineups is ~60 solves.

## Extending it

- **Field-ownership modeling** — replace `--ownership-leverage` with a real GPP
  equity model: simulate the field's lineups too, and optimize for expected
  payout rather than expected points. This is the biggest remaining gap.
- **Batting-order correlation** — consecutive hitters (3-4-5) correlate more
  tightly than a 2-hole and an 8-hole. The `Order` column is loaded already.
- **Park and weather factors** — scale team run environments before simulating.
- **Late swap** — re-run with locked players for games already started and a
  trimmed pool for the rest.
