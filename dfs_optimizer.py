#!/usr/bin/env python3
"""
DraftKings MLB Classic Lineup Optimizer
=======================================

Builds DK MLB Classic lineups (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF - $50,000 cap)
with integer linear programming, correlated stacking, and Monte Carlo scoring.

INPUTS
------
1. Your DK salary export ("Export to CSV" on the contest lineup page).
2. Your projections CSV, merged onto DK by player name:
       Name, Proj [, Ceiling, Floor, Ownership, Team, Order]
   Only Name and Proj are required. Include Team whenever two players on the
   slate share a name - it's the only way to tell them apart.

EXAMPLES
--------
    # One safe cash lineup
    python dfs_optimizer.py --dk-salaries DKSalaries.csv \
        --projections proj.csv --mode cash

    # 20 GPP lineups, solver-chosen 4-man stack + 3-man bring-back,
    # ranked by simulated upside
    python dfs_optimizer.py --dk-salaries DKSalaries.csv \
        --projections proj.csv --mode gpp --num-lineups 20 \
        --stack-size 4 --secondary-stack-size 3 \
        --rank-by p90 --max-exposure 0.5 --output lineups.csv

Run with --help for all options.
"""

import argparse
import sys

import pandas as pd

from mlb_dfs.data import filter_availability, load_pool
from mlb_dfs.optimizer import Constraints, generate_lineups
from mlb_dfs.output import exposures, summarize, to_upload_dataframe
from mlb_dfs.simulate import lineup_metrics, score_lineups, select_portfolio, simulate_players


def build_parser():
    ap = argparse.ArgumentParser(
        description="DraftKings MLB Classic lineup optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dk-salaries", required=True, help="Path to DK salary export CSV")
    ap.add_argument("--projections", default=None,
                    help="Path to your projections CSV. If omitted, falls back to DK's "
                         "AvgPointsPerGame - a season average, not a projection.")
    ap.add_argument("--mode", choices=["cash", "gpp", "balanced"], default="balanced",
                    help="cash=max floor, gpp=max ceiling, balanced=max projection")
    ap.add_argument("--num-lineups", type=int, default=1)
    ap.add_argument("--output", default="lineups.csv",
                    help="DK-ready upload CSV (exactly the 10 roster columns)")

    roster = ap.add_argument_group("roster constraints")
    roster.add_argument("--max-per-team", type=int, default=None,
                        help="Max hitters from one team (DK caps this at 5)")
    roster.add_argument("--min-salary-used", type=int, default=0)
    roster.add_argument("--allow-opposing-pitcher-hitter", action="store_true",
                        help="Permit rostering a hitter facing your own pitcher")
    roster.add_argument("--lock", default="", help="Comma-separated names to force in")
    roster.add_argument("--exclude", default="", help="Comma-separated names to ban")
    roster.add_argument("--max-batting-order", type=int, default=None,
                        help="Drop hitters batting below this spot (needs an Order column)")

    avail = ap.add_argument_group("who is actually playing")
    avail.add_argument("--include-injured", action="store_true",
                       help="Keep players listed OUT/IL (they are dropped by default)")
    avail.add_argument("--require-confirmed-order", action="store_true",
                       help="Only use hitters with a confirmed batting order")
    avail.add_argument("--allow-non-starting-pitchers", action="store_true",
                       help="Keep pitchers not listed as today's starter")

    stacking = ap.add_argument_group("stacking")
    stacking.add_argument("--stack-size", type=int, default=0,
                          help="Hitters required from the primary stack team")
    stacking.add_argument("--stack-team", default=None,
                          help="Force the stack onto one team; omit to let the solver choose")
    stacking.add_argument("--secondary-stack-size", type=int, default=0,
                          help="Hitters required from a second team (the bring-back)")
    stacking.add_argument("--stack-min-implied", type=float, default=None,
                          help="Only stack teams with at least this Vegas implied run "
                               "total (needs an implied_team_score column)")

    portfolio = ap.add_argument_group("portfolio shape")
    portfolio.add_argument("--min-unique", type=int, default=3,
                           help="Min differing players between any two lineups")
    portfolio.add_argument("--max-exposure", type=float, default=1.0,
                           help="Max share of lineups any one player may appear in (0-1)")
    portfolio.add_argument("--ownership-leverage", type=float, default=0.0,
                           help="Subtract leverage*ownership from the objective to fade chalk")
    portfolio.add_argument("--randomness", type=float, default=0.0,
                           help="Jitter projections per lineup (e.g. 0.15) to diversify")

    sim = ap.add_argument_group("simulation")
    sim.add_argument("--simulate", action="store_true",
                     help="Report each lineup's simulated outcome distribution")
    sim.add_argument("--sims", type=int, default=10000, help="Monte Carlo iterations")
    sim.add_argument("--rank-by", choices=["mean", "p90", "p10"], default=None,
                     help="Build extra candidates, then keep the best by this metric")
    sim.add_argument("--oversample", type=float, default=3.0,
                     help="Candidate multiple to generate when using --rank-by")
    sim.add_argument("--seed", type=int, default=None, help="Seed for reproducible runs")

    reports = ap.add_argument_group("extra reports")
    reports.add_argument("--summary-csv", default=None, help="Write the per-lineup summary")
    reports.add_argument("--exposure-csv", default=None, help="Write player exposures")
    return ap


def apply_batting_order_filter(pool, max_order):
    """Drop weak-spot hitters. Pitchers and unknown orders are always kept."""
    if max_order is None:
        return pool
    if "batting_order" not in pool.columns or pool["batting_order"].isna().all():
        print("[warn] --max-batting-order ignored: projections have no Order column.",
              file=sys.stderr)
        return pool
    order = pd.to_numeric(pool["batting_order"], errors="coerce")
    keep = pool["is_pitcher"] | order.isna() | (order <= max_order)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[info] --max-batting-order {max_order}: dropped {dropped} hitter(s).",
              file=sys.stderr)
    return pool[keep].reset_index(drop=True)


def main(argv=None):
    args = build_parser().parse_args(argv)

    pool, _report = load_pool(args.dk_salaries, args.projections)
    pool = filter_availability(
        pool,
        exclude_injured=not args.include_injured,
        require_confirmed_order=args.require_confirmed_order,
        require_starting_pitcher=not args.allow_non_starting_pitchers,
    )
    pool = apply_batting_order_filter(pool, args.max_batting_order)
    if pool.empty:
        print("[error] No players left after loading. Check that your projection names "
              "match the DK export.", file=sys.stderr)
        return 1

    constraints = Constraints(
        mode=args.mode,
        max_per_team=args.max_per_team,
        min_salary_used=args.min_salary_used,
        no_opposing_pitcher_hitter=not args.allow_opposing_pitcher_hitter,
        lock_names=[n.strip() for n in args.lock.split(",") if n.strip()],
        exclude_names=[n.strip() for n in args.exclude.split(",") if n.strip()],
        stack_team=args.stack_team,
        stack_size=args.stack_size,
        secondary_stack_size=args.secondary_stack_size,
        ownership_leverage=args.ownership_leverage,
        min_unique_players=args.min_unique,
        max_exposure=args.max_exposure,
        randomness=args.randomness,
        stack_min_implied=args.stack_min_implied,
    )

    want = args.num_lineups
    if args.rank_by:
        # Generate a deeper candidate pool, then keep the best-simulating slice.
        target = max(want, int(round(want * max(1.0, args.oversample))))
        if not constraints.randomness:
            # Without jitter every extra candidate is just the next-best ILP
            # solution, which makes for a thin, near-identical candidate set.
            constraints.randomness = 0.15
    else:
        target = want

    lineups = generate_lineups(pool, constraints, target, seed=args.seed)
    if not lineups:
        print("[error] No feasible lineup found. Loosen the salary floor, stack size, "
              "locks/excludes, or exposure cap.", file=sys.stderr)
        return 1

    sim_metrics = None
    if args.rank_by or args.simulate:
        draws = simulate_players(pool, n_sims=args.sims, seed=args.seed)
        scores = score_lineups(draws, lineups, pool)
        if args.rank_by:
            keep, relaxed = select_portfolio(lineups, scores, want, args.min_unique,
                                             args.rank_by, args.max_exposure)
            if relaxed:
                print(f"[warn] {relaxed} lineup(s) exceed --max-exposure "
                      f"{args.max_exposure}: not enough diverse candidates. "
                      f"Raise --oversample or --randomness to fix.", file=sys.stderr)
            lineups = [lineups[k] for k in keep]
            scores = scores[keep]
            print(f"[info] kept {len(lineups)} of {target} candidates by simulated "
                  f"{args.rank_by}.", file=sys.stderr)
        sim_metrics = [lineup_metrics(scores[n]) for n in range(len(lineups))]

    upload = to_upload_dataframe(pool, lineups)
    upload.to_csv(args.output, index=False)

    summary = summarize(pool, lineups, sim_metrics)
    exposure = exposures(pool, lineups)
    if args.summary_csv:
        summary.to_csv(args.summary_csv, index=False)
    if args.exposure_csv:
        exposure.to_csv(args.exposure_csv, index=False)

    print(f"[info] wrote {len(upload)} lineup(s) to {args.output} (DK bulk-upload ready)")
    print()
    print(summary.to_string(index=False))
    if len(lineups) > 1:
        print()
        print("Top exposures:")
        print(exposure.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
