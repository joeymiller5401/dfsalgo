"""
Integer-programming lineup builder for DK MLB Classic.

Every (player, roster slot) pair is a binary variable. CBC maximizes the
mode's objective subject to DK's roster rules plus whatever stacking,
exposure and diversity constraints you ask for.
"""

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from .rules import (
    MAX_HITTERS,
    MAX_HITTERS_PER_TEAM,
    MIN_DISTINCT_GAMES,
    ROSTER_SLOTS,
    SALARY_CAP,
    TOTAL_PLAYERS,
    canonical_team,
)

def _solver():
    """
    Pick a CBC solver that works across PuLP versions.

    PuLP 3.x warns that PULP_CBC_CMD goes away in 4.0 in favour of COIN_CMD,
    so try the modern name first and fall back to the bundled one.
    """
    for name in ("COIN_CMD", "PULP_CBC_CMD"):
        factory = getattr(pulp, name, None)
        if factory is None:
            continue
        try:
            solver = factory(msg=0)
            if solver.available():
                return solver
        except Exception:
            continue
    raise RuntimeError(
        "No CBC solver available. Install one with: pip install 'pulp[cbc]'"
    )


MODE_DEFAULTS = {
    # cash games: protect the floor, cap exposure to any one offense
    "cash": {"objective_col": "floor", "max_per_team": 4},
    # tournaments: chase the ceiling, allow a full 5-man stack
    "gpp": {"objective_col": "ceiling", "max_per_team": MAX_HITTERS_PER_TEAM},
    "balanced": {"objective_col": "proj", "max_per_team": MAX_HITTERS_PER_TEAM},
}


@dataclass
class Constraints:
    mode: str = "balanced"
    max_per_team: int = None            # None -> mode default
    min_salary_used: int = 0
    no_opposing_pitcher_hitter: bool = True
    lock_names: list = field(default_factory=list)
    exclude_names: list = field(default_factory=list)

    # Stacking. stack_team=None with stack_size>0 lets the solver pick the team.
    stack_team: str = None
    stack_size: int = 0
    secondary_stack_size: int = 0

    ownership_leverage: float = 0.0     # subtract leverage * ownership
    min_unique_players: int = 3         # diversity between generated lineups
    max_exposure: float = 1.0           # max share of lineups any one player fills
    randomness: float = 0.0             # jitter projections to diversify a portfolio
    stack_min_implied: float = None     # only stack teams with this Vegas run total
    enforce_dk_rules: bool = True       # 2-game minimum, 5-hitter team cap


class LineupBuilder:
    """Precomputes the slate structure once, then solves repeatedly."""

    def __init__(self, pool, constraints: Constraints, rng=None):
        self.pool = pool
        self.c = constraints
        self.rng = rng or np.random.default_rng()
        self.solver = _solver()

        self.idx = list(pool.index)
        self.positions = {i: list(pool.at[i, "positions"]) for i in self.idx}
        self.salary = {i: int(pool.at[i, "salary"]) for i in self.idx}
        self.team = {i: pool.at[i, "team"] for i in self.idx}
        self.game = {i: pool.at[i, "game_id"] for i in self.idx}
        self.is_pitcher = {i: bool(pool.at[i, "is_pitcher"]) for i in self.idx}
        self.name = {i: pool.at[i, "name"] for i in self.idx}

        self.hitters = [i for i in self.idx if not self.is_pitcher[i]]
        self.pitchers = [i for i in self.idx if self.is_pitcher[i]]

        self.team_hitters = {}
        for i in self.hitters:
            self.team_hitters.setdefault(self.team[i], []).append(i)

        self.game_players = {}
        for i in self.idx:
            self.game_players.setdefault(self.game[i], []).append(i)

        # Vegas implied run total per team, when the projections supply one.
        self.team_implied = {}
        if "implied_team_score" in pool.columns:
            for i in self.hitters:
                value = pool.at[i, "implied_team_score"]
                if pd.notna(value):
                    self.team_implied[self.team[i]] = float(value)

        self.name_to_idxs = {}
        for i in self.idx:
            self.name_to_idxs.setdefault(self.name[i], []).append(i)

        mode_cfg = MODE_DEFAULTS[self.c.mode]
        self.obj_col = mode_cfg["objective_col"]
        cap = self.c.max_per_team if self.c.max_per_team is not None else mode_cfg["max_per_team"]
        if self.c.enforce_dk_rules:
            cap = min(cap, MAX_HITTERS_PER_TEAM)
        self.max_per_team = cap

        self.base_value = {
            i: float(pool.at[i, self.obj_col])
            - self.c.ownership_leverage * float(pool.at[i, "ownership"])
            for i in self.idx
        }

    # -- objective ---------------------------------------------------------
    def _values(self):
        """Per-solve player values, optionally jittered to diversify a portfolio."""
        if not self.c.randomness:
            return self.base_value
        noise = self.rng.normal(1.0, self.c.randomness, size=len(self.idx))
        return {i: self.base_value[i] * max(0.0, n) for i, n in zip(self.idx, noise)}

    # -- model -------------------------------------------------------------
    def solve(self, previous_lineups=None, banned=frozenset()):
        """
        Solve for one lineup.

        previous_lineups: list of sets of player indices already used, for the
                          min-unique diversity constraint.
        banned:           player indices frozen out (exposure caps).
        Returns a list of (player_index, slot) or None if infeasible.
        """
        previous_lineups = previous_lineups or []
        c = self.c
        prob = pulp.LpProblem("dk_mlb_classic", pulp.LpMaximize)

        x = {}
        for i in self.idx:
            if i in banned:
                continue
            for slot in self.positions[i]:
                if slot in ROSTER_SLOTS:
                    x[(i, slot)] = pulp.LpVariable(f"x_{i}_{slot}", cat="Binary")

        # "used[i]" is an expression, not a variable - keeps the model small.
        used = {}
        for i in self.idx:
            slots = [s for s in self.positions[i] if (i, s) in x]
            if slots:
                used[i] = pulp.lpSum(x[(i, s)] for s in slots)

        active = list(used)
        if not active:
            return None

        values = self._values()
        prob += pulp.lpSum(values[i] * used[i] for i in active)

        # Each player at most once, across every slot they're eligible for.
        for i in active:
            prob += used[i] <= 1, f"once_{i}"

        # Fill every roster slot exactly.
        for slot, count in ROSTER_SLOTS.items():
            terms = [x[(i, slot)] for i in self.idx if (i, slot) in x]
            if len(terms) < count:
                return None
            prob += pulp.lpSum(terms) == count, f"slot_{slot}"

        # Salary cap.
        salary_expr = pulp.lpSum(self.salary[i] * used[i] for i in active)
        prob += salary_expr <= SALARY_CAP, "cap"
        if c.min_salary_used:
            prob += salary_expr >= c.min_salary_used, "min_salary"

        # DK: at most N hitters from one team.
        for team, members in self.team_hitters.items():
            terms = [used[i] for i in members if i in used]
            if terms:
                prob += pulp.lpSum(terms) <= self.max_per_team, f"teamcap_{team}"

        # DK: players from at least 2 different games.
        if c.enforce_dk_rules and len(self.game_players) >= MIN_DISTINCT_GAMES:
            g = {}
            for game, members in self.game_players.items():
                terms = [used[i] for i in members if i in used]
                if not terms:
                    continue
                g[game] = pulp.LpVariable(f"g_{abs(hash(game))}", cat="Binary")
                prob += pulp.lpSum(terms) <= TOTAL_PLAYERS * g[game], f"game_{game}"
            if len(g) >= MIN_DISTINCT_GAMES:
                prob += pulp.lpSum(g.values()) >= MIN_DISTINCT_GAMES, "min_games"

        # Never roster a hitter facing your own pitcher. One aggregated
        # constraint per pitcher instead of one per (pitcher, hitter) pair.
        if c.no_opposing_pitcher_hitter:
            for p in self.pitchers:
                if p not in used:
                    continue
                opp = self.pool.at[p, "opponent"]
                if not opp:
                    continue
                opp_terms = [used[h] for h in self.team_hitters.get(opp, []) if h in used]
                if opp_terms:
                    prob += (pulp.lpSum(opp_terms) <= MAX_HITTERS * (1 - used[p]),
                             f"nopvh_{p}")

        # Locks / excludes.
        for nm in c.lock_names:
            targets = [i for i in self.name_to_idxs.get(nm, []) if i in used]
            if not targets:
                print(f"[warn] --lock '{nm}' is not in the player pool; ignored.",
                      file=sys.stderr)
                continue
            prob += pulp.lpSum(used[i] for i in targets) == 1, f"lock_{nm}"
        for nm in c.exclude_names:
            for i in self.name_to_idxs.get(nm, []):
                if i in used:
                    prob += used[i] == 0, f"excl_{nm}_{i}"

        # Stacking.
        self._add_stacks(prob, used)

        # Diversity against lineups already built.
        max_overlap = TOTAL_PLAYERS - c.min_unique_players
        for n, prev in enumerate(previous_lineups):
            terms = [used[i] for i in prev if i in used]
            if terms:
                prob += pulp.lpSum(terms) <= max_overlap, f"uniq_{n}"

        status = prob.solve(self.solver)
        if pulp.LpStatus[status] != "Optimal":
            return None

        return [(i, slot) for (i, slot) in x if x[(i, slot)].value() == 1]

    def _add_stacks(self, prob, used):
        """Force a primary (and optional secondary) team stack."""
        c = self.c
        if c.stack_size <= 0:
            return

        eligible = {}
        for team, members in self.team_hitters.items():
            terms = [used[i] for i in members if i in used]
            if len(terms) >= c.stack_size:
                eligible[team] = terms

        # Vegas gate: don't stack an offense the market expects to be quiet.
        if c.stack_min_implied is not None and self.team_implied:
            gated = {t: terms for t, terms in eligible.items()
                     if self.team_implied.get(t, float("nan")) >= c.stack_min_implied}
            if gated:
                eligible = gated
            else:
                print(f"[warn] no team meets --stack-min-implied "
                      f"{c.stack_min_implied}; gate not applied.", file=sys.stderr)

        if c.stack_team:
            want = canonical_team(c.stack_team)
            if want not in eligible:
                print(f"[warn] --stack-team {c.stack_team}: fewer than {c.stack_size} "
                      f"hitters selectable for this lineup (pool size, or players "
                      f"frozen out by --max-exposure); stack not applied.",
                      file=sys.stderr)
                return
            eligible = {want: eligible[want]}

        if not eligible:
            print(f"[warn] no team has {c.stack_size} selectable hitters for this "
                  f"lineup; stack not applied.", file=sys.stderr)
            return

        # y[t] = 1 when team t is the primary stack. Exactly one team is.
        y = {t: pulp.LpVariable(f"stk_{t}", cat="Binary") for t in eligible}
        prob += pulp.lpSum(y.values()) == 1, "one_primary_stack"
        for t, terms in eligible.items():
            prob += pulp.lpSum(terms) >= c.stack_size * y[t], f"stack_{t}"

        if c.secondary_stack_size <= 0:
            return

        sec_eligible = {}
        for t, members in self.team_hitters.items():
            terms = [used[i] for i in members if i in used]
            if len(terms) >= c.secondary_stack_size:
                sec_eligible[t] = terms

        # The Vegas gate covers the secondary stack too - "only stack teams
        # above X" reads the same way for both halves of a double stack.
        if c.stack_min_implied is not None and self.team_implied:
            gated = {t: terms for t, terms in sec_eligible.items()
                     if self.team_implied.get(t, float("nan")) >= c.stack_min_implied}
            if len(gated) >= 2:
                sec_eligible = gated
        if len(sec_eligible) < 2:
            print("[warn] not enough teams for a secondary stack; skipped.", file=sys.stderr)
            return

        z = {t: pulp.LpVariable(f"sec_{t}", cat="Binary") for t in sec_eligible}
        prob += pulp.lpSum(z.values()) == 1, "one_secondary_stack"
        for t, terms in sec_eligible.items():
            prob += pulp.lpSum(terms) >= c.secondary_stack_size * z[t], f"secstack_{t}"
            if t in y:
                # A team can't be both the primary and the secondary stack.
                prob += y[t] + z[t] <= 1, f"distinct_stack_{t}"


def generate_lineups(pool, constraints: Constraints, num_lineups, seed=None, quiet=False):
    """Build a portfolio of lineups honoring diversity and exposure caps."""
    builder = LineupBuilder(pool, constraints, rng=np.random.default_rng(seed))
    lineups = []
    used_sets = []
    counts = {}
    cap = None
    if constraints.max_exposure < 1.0:
        cap = max(1, int(round(constraints.max_exposure * num_lineups)))

    for n in range(num_lineups):
        banned = frozenset()
        if cap is not None:
            banned = frozenset(i for i, ct in counts.items() if ct >= cap)
        chosen = builder.solve(previous_lineups=used_sets, banned=banned)
        if chosen is None and banned:
            # Exposure caps can over-constrain a thin slate; relax and retry once.
            chosen = builder.solve(previous_lineups=used_sets)
            if chosen is not None and not quiet:
                print(f"[warn] lineup {n + 1}: exposure cap relaxed to stay feasible.",
                      file=sys.stderr)
        if chosen is None:
            if not quiet:
                print(f"[info] stopped after {len(lineups)} lineup(s): no further "
                      f"lineups satisfy the diversity/stack constraints.", file=sys.stderr)
            break
        players = set(i for i, _ in chosen)
        used_sets.append(players)
        lineups.append(chosen)
        for i in players:
            counts[i] = counts.get(i, 0) + 1

    return lineups
