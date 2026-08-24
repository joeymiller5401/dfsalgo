"""
Load a DK salary export plus a projections file into a single player pool.

Two jobs, both places the original script failed silently:

1. Reading DK's own quirks (SP/RP vs P, Showdown files, "Game Info" parsing).
2. Matching projections onto DK's player list without corrupting rows when two
   players share a name.

Projection sources disagree on almost every column name, so the loader matches
on aliases rather than exact headers. Daily Fantasy Fuel's cheatsheet
(`first_name`/`last_name`/`ppg_projection`/`confirmed_order`/...) works without
any preprocessing.
"""

import re
import sys
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .distributions import Z_P15, Z_P85, lognormal_quantile, role_defaults
from .rules import MIN_DISTINCT_GAMES, SHOWDOWN_MARKERS, canonical_team, normalize_positions

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

# Statuses that mean the player will not take the field. DTD ("day to day") is
# deliberately absent - those players often play, so they're flagged, not cut.
OUT_STATUSES = {"O", "OUT", "IL", "IL10", "IL15", "IL60", "NA", "SUSP", "DL"}

# How strongly recent-form dispersion widens a derived distribution.
FORM_VOLATILITY_WEIGHT = 0.5


def normalize_name(name):
    """
    Build a match key that survives formatting differences between sources:
    accents, punctuation, and generational suffixes.

        "José Ramírez"      -> "jose ramirez"
        "Ronald Acuna Jr."  -> "ronald acuna"
        "Travis d'Arnaud"   -> "travis darnaud"
    """
    if name is None:
        return ""
    text = str(name).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[.,'`]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = [p for p in text.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def find_col(df, *candidates):
    """Case-insensitive column lookup; returns the first match or None."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _player_names(df):
    """
    Pull a full name out of a projections file.

    Handles a single "Name"/"Player" column, or a split first/last pair as used
    by Daily Fantasy Fuel and several other cheatsheet exports.
    """
    name_col = find_col(df, "Name", "Player", "player_name", "full_name")
    if name_col is not None:
        return df[name_col].astype(str).str.strip()

    first = find_col(df, "first_name", "First Name", "First")
    last = find_col(df, "last_name", "Last Name", "Last")
    if first is not None and last is not None:
        return (df[first].astype(str).str.strip() + " "
                + df[last].astype(str).str.strip()).str.strip()
    return None


@dataclass
class LoadReport:
    """What happened during load, so failures are visible instead of silent."""
    dk_rows: int = 0
    projection_rows: int = 0
    matched: int = 0
    unmatched_names: list = field(default_factory=list)
    no_valid_position: list = field(default_factory=list)
    ambiguous_names: list = field(default_factory=list)
    missing_projection_value: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def render(self):
        lines = [
            f"DK players: {self.dk_rows} | projections: {self.projection_rows} "
            f"| usable pool: {self.matched}"
        ]

        def _bullet(label, names, hint=""):
            if not names:
                return
            preview = ", ".join(str(n) for n in names[:5])
            more = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
            lines.append(f"  - {len(names)} {label}: {preview}{more}{hint}")

        _bullet("dropped, no matching projection", self.unmatched_names)
        _bullet("dropped, no Classic roster slot", self.no_valid_position)
        _bullet("dropped, blank/invalid projection", self.missing_projection_value)
        _bullet("dropped, ambiguous name", self.ambiguous_names,
                hint="  -> add a 'Team' column to your projections to fix")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def _parse_game_info(game_info, team_canon):
    """
    DK's "Game Info" looks like "NYY@BOS 07:05PM ET" (AWAY@HOME).

    Returns (opponent, game_id). game_id is a stable matchup key, used for
    DK's 2-game rule and for correlating players in the same game.
    """
    text = str(game_info or "").strip()
    match = re.match(r"^\s*([A-Za-z]{2,4})\s*@\s*([A-Za-z]{2,4})", text)
    if not match:
        return "", ""
    away = canonical_team(match.group(1))
    home = canonical_team(match.group(2))
    game_id = f"{away}@{home}"
    if team_canon == away:
        return home, game_id
    if team_canon == home:
        return away, game_id
    return "", game_id


def load_dk_salaries(path):
    """Read a DK Classic salary export into normalized columns."""
    dk = pd.read_csv(path)

    name_col = find_col(dk, "Name")
    salary_col = find_col(dk, "Salary")
    team_col = find_col(dk, "TeamAbbrev", "Team")
    id_col = find_col(dk, "ID", "Id")
    game_col = find_col(dk, "Game Info", "GameInfo")
    avg_col = find_col(dk, "AvgPointsPerGame", "Avg Points Per Game", "AvgPPG")

    # Prefer "Roster Position" - it is the slot DK will actually accept.
    # "Position" holds SP/RP, which are not roster slots.
    roster_pos_col = find_col(dk, "Roster Position")
    position_col = find_col(dk, "Position")
    pos_col = roster_pos_col or position_col

    missing = [label for label, col in
               [("Name", name_col), ("Position", pos_col),
                ("Salary", salary_col), ("TeamAbbrev", team_col)] if col is None]
    if missing:
        raise ValueError(
            f"DK salary file is missing expected column(s): {missing}. "
            f"Found: {list(dk.columns)}"
        )

    raw_positions = dk[pos_col].astype(str)
    if roster_pos_col and position_col:
        tokens = set()
        for value in raw_positions:
            tokens.update(t.strip().upper() for t in str(value).replace("/", ",").split(","))
        if tokens & SHOWDOWN_MARKERS:
            raise ValueError(
                "This looks like a DK *Showdown* export (roster positions CPT/UTIL). "
                "This optimizer builds Classic lineups - export the Classic contest instead."
            )

    pool = pd.DataFrame({
        "name": dk[name_col].astype(str).str.strip(),
        "team_raw": dk[team_col].astype(str).str.strip().str.upper(),
        "salary": pd.to_numeric(dk[salary_col], errors="coerce"),
    })
    pool["team"] = pool["team_raw"].map(canonical_team)
    pool["positions"] = [normalize_positions(v) for v in raw_positions]
    pool["dk_id"] = dk[id_col].astype(str) if id_col else [str(i) for i in range(len(dk))]
    pool["dk_avg"] = pd.to_numeric(dk[avg_col], errors="coerce") if avg_col else np.nan

    if game_col:
        parsed = [_parse_game_info(gi, tm) for gi, tm in zip(dk[game_col], pool["team"])]
    else:
        parsed = [("", "") for _ in range(len(pool))]
    pool["opponent"] = [p[0] for p in parsed]
    pool["game_id"] = [p[1] for p in parsed]
    # Without Game Info we can't tell games apart; fall back to team so the
    # 2-game rule degrades to "2 different teams" rather than breaking.
    pool.loc[pool["game_id"] == "", "game_id"] = pool["team"]

    pool["name_key"] = pool["name"].map(normalize_name)
    pool["is_pitcher"] = pool["positions"].apply(lambda ps: "P" in ps)
    return pool


def load_projections(path):
    """
    Read a projections file into normalized columns.

    Only a name and a projection are required. Everything else is optional and
    unlocks extra behavior; column names are matched by alias, so most
    cheatsheet exports work as-is.
    """
    proj = pd.read_csv(path)

    names = _player_names(proj)
    if names is None:
        raise ValueError(
            "Projections file needs a 'Name' column (or 'first_name' + "
            f"'last_name') matching DK player names. Found: {list(proj.columns)}"
        )

    value_col = find_col(proj, "Proj", "Projection", "FPTS", "Points", "Fpts",
                         "ppg_projection", "PPG", "proj_points", "projection_points")
    if value_col is None:
        raise ValueError(
            "Projections file needs a projection column - one of Proj, Projection, "
            f"FPTS, Points, or ppg_projection. Found: {list(proj.columns)}"
        )

    ceil_col = find_col(proj, "Ceiling", "Ceil", "ceiling_projection")
    floor_col = find_col(proj, "Floor", "floor_projection")
    own_col = find_col(proj, "Ownership", "Own", "Own%", "pOwn", "ownership_projection")
    team_col = find_col(proj, "Team", "TeamAbbrev", "team")
    order_col = find_col(proj, "Order", "BattingOrder", "Batting Order",
                         "Lineup Slot", "confirmed_order", "batting_order")
    status_col = find_col(proj, "injury_status", "Injury", "Status", "injury")
    starter_col = find_col(proj, "starting_pitcher", "is_starter", "Starter")
    implied_col = find_col(proj, "implied_team_score", "ImpliedTeamScore",
                           "implied_runs", "team_total")
    opp_col = find_col(proj, "opp", "Opponent", "Opp")

    out = pd.DataFrame({
        "name_key": names.map(normalize_name),
        "proj": pd.to_numeric(proj[value_col], errors="coerce"),
    })
    out["ceiling"] = pd.to_numeric(proj[ceil_col], errors="coerce") if ceil_col else np.nan
    out["floor"] = pd.to_numeric(proj[floor_col], errors="coerce") if floor_col else np.nan
    out["ownership"] = pd.to_numeric(proj[own_col], errors="coerce") if own_col else np.nan
    out["batting_order"] = pd.to_numeric(proj[order_col], errors="coerce") if order_col else np.nan
    out["proj_team"] = proj[team_col].map(canonical_team) if team_col else ""
    out["injury_status"] = (proj[status_col].astype(str).str.strip().str.upper()
                            if status_col else "")
    out["is_listed_starter"] = (
        proj[starter_col].astype(str).str.strip().str.upper().eq("YES")
        if starter_col else pd.NA
    )
    out["implied_team_score"] = (pd.to_numeric(proj[implied_col], errors="coerce")
                                 if implied_col else np.nan)
    out["proj_opp"] = proj[opp_col].map(canonical_team) if opp_col else ""

    # Recent-form averages, used to estimate volatility when a source gives no
    # ceiling/floor of its own.
    form_cols = [find_col(proj, c) for c in
                 ("L5_fppg_avg", "L10_fppg_avg", "szn_fppg_avg")]
    form_cols = [c for c in form_cols if c is not None]
    for n, col in enumerate(form_cols):
        out[f"form_{n}"] = pd.to_numeric(proj[col], errors="coerce")

    out.attrs["has_team"] = team_col is not None
    out.attrs["has_ceiling"] = ceil_col is not None
    out.attrs["has_floor"] = floor_col is not None
    out.attrs["has_ownership"] = own_col is not None
    out.attrs["has_starter_flag"] = starter_col is not None
    out.attrs["n_form_cols"] = len(form_cols)
    return out


def derive_ceiling_floor(pool, report=None):
    """
    Estimate a ceiling and floor when the projections file supplies none.

    Without this, `--mode gpp` and `--mode cash` silently collapse into
    `balanced`, because both objectives fall back to the point projection.

    The estimate is the 85th/15th percentile of the same shifted lognormal the
    simulator uses, so a derived ceiling means exactly what a simulated one
    does. Spread comes from a role-typical coefficient of variation, widened
    for players whose recent-form averages disagree with each other.

    That form signal is a rough proxy: an L5 average is noisier than a season
    average by construction, so some of the dispersion is sampling noise rather
    than true volatility. It ranks players sensibly; don't read it as precise.
    """
    proj = pool["proj"].to_numpy(dtype=float)
    is_p = pool["is_pitcher"].to_numpy(dtype=bool)
    cv, shift = role_defaults(is_p)

    form_cols = [c for c in pool.columns if c.startswith("form_")]
    if len(form_cols) >= 2:
        form = pool[form_cols].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.nanstd(form, axis=1) / np.maximum(np.nanmean(form, axis=1), 1e-6)
        spread = np.nan_to_num(spread, nan=0.0, posinf=0.0)
        cv = np.clip(cv + FORM_VOLATILITY_WEIGHT * spread, cv * 0.7, cv * 1.8)
        if report is not None:
            report.notes.append(
                f"ceiling/floor derived from projections + recent-form volatility "
                f"({len(form_cols)} form columns)"
            )
    elif report is not None:
        report.notes.append(
            "ceiling/floor derived from projections using role-typical variance "
            "(no ceiling/floor or recent-form columns found)"
        )

    std = np.abs(proj) * cv
    ceiling = lognormal_quantile(proj, std, shift, Z_P85)
    floor = lognormal_quantile(proj, std, shift, Z_P15)
    return ceiling, floor


def _merge_projections(dk, proj, report):
    """Attach projections to DK rows, refusing to guess on ambiguous names."""
    has_team = bool(proj.attrs.get("has_team"))

    dk_dupes = set(dk["name_key"][dk["name_key"].duplicated(keep=False)])
    proj_dupes = set(proj["name_key"][proj["name_key"].duplicated(keep=False)])
    contested = dk_dupes | proj_dupes

    dk = dk.reset_index(drop=True)
    dk["_row"] = range(len(dk))
    value_cols = ["proj", "ceiling", "floor", "ownership", "batting_order"]

    if has_team:
        proj = proj.drop_duplicates(subset=["name_key", "proj_team"], keep="first")
        merged = dk.merge(proj, on="name_key", how="left", suffixes=("", "_p"))
        # A contested name only counts as matched when the teams agree. Rows
        # that disagree are invalidated rather than dropped, so they surface as
        # unmatched instead of vanishing from the report.
        is_contested = merged["name_key"].isin(contested)
        merged.loc[is_contested & (merged["proj_team"] != merged["team"]), value_cols] = None
        merged["_matched"] = merged["proj"].notna()
        merged = (merged.sort_values(["_row", "_matched"], ascending=[True, False])
                        .drop_duplicates(subset=["dk_id"], keep="first")
                        .drop(columns=["_matched"]))
    else:
        unique_proj = proj.drop_duplicates(subset=["name_key"], keep="first")
        merged = dk.merge(unique_proj, on="name_key", how="left", suffixes=("", "_p"))
        # Can't tell duplicated names apart without a team column - drop them
        # rather than assign one player's projection to another.
        clash = merged[merged["name_key"].isin(contested)]
        report.ambiguous_names = sorted(f"{n} ({t})" for n, t
                                        in zip(clash["name"], clash["team"]))
        merged = merged[~merged["name_key"].isin(contested)].copy()
        merged = merged.sort_values("_row")

    return merged


def load_pool(dk_salaries_path, projections_path=None, verbose=True):
    """
    Build the player pool the optimizer runs on.

    `projections_path` is optional. Without it the loader falls back to DK's own
    AvgPointsPerGame, which is a season average, not a projection - useful for a
    smoke test, not for playing. See the warning it prints.

    Returns (pool_dataframe, LoadReport).
    """
    dk = load_dk_salaries(dk_salaries_path)
    report = LoadReport(dk_rows=len(dk))

    # Drop players with no Classic roster slot (e.g. DH-only) before matching.
    no_slot = dk[dk["positions"].apply(len) == 0]
    report.no_valid_position = sorted(no_slot["name"].tolist())
    dk = dk[dk["positions"].apply(len) > 0].copy()

    if projections_path is None:
        if dk["dk_avg"].isna().all():
            raise ValueError(
                "No --projections given and the DK export has no AvgPointsPerGame "
                "column to fall back on."
            )
        merged = dk.reset_index(drop=True)
        merged["_row"] = range(len(merged))
        merged["proj"] = merged["dk_avg"]
        for col, default in [("ceiling", np.nan), ("floor", np.nan),
                             ("ownership", np.nan), ("batting_order", np.nan),
                             ("implied_team_score", np.nan)]:
            merged[col] = default
        merged["injury_status"] = ""
        merged["is_listed_starter"] = pd.NA
        report.projection_rows = int(merged["proj"].notna().sum())
        report.notes.append(
            "NO PROJECTIONS FILE: using DK's AvgPointsPerGame (a season average). "
            "It cannot know who is in today's lineup - expect benched players."
        )
        proj_attrs = {"has_ceiling": False, "has_floor": False, "has_ownership": False}
    else:
        proj = load_projections(projections_path)
        report.projection_rows = len(proj)
        proj_attrs = dict(proj.attrs)
        merged = _merge_projections(dk, proj, report)

    unmatched = merged[merged["proj"].isna()]
    report.unmatched_names = sorted(unmatched["name"].tolist())
    merged = merged[merged["proj"].notna()].copy()

    bad_salary = merged[merged["salary"].isna()]
    report.missing_projection_value = sorted(bad_salary["name"].tolist())
    merged = merged[merged["salary"].notna()].copy()

    if merged.empty:
        report.matched = 0
        if verbose:
            print(report.render(), file=sys.stderr)
        return merged, report

    # Ownership: an all-blank column is the same as no column at all, and
    # silently makes --ownership-leverage a no-op.
    if "ownership" not in merged.columns or merged["ownership"].isna().all():
        if proj_attrs.get("has_ownership"):
            report.notes.append(
                "ownership column is present but empty - --ownership-leverage "
                "will have no effect"
            )
        merged["ownership"] = 0.0
    else:
        merged["ownership"] = merged["ownership"].fillna(0.0)

    # Ceiling/floor: derive when absent, otherwise sanity-check what was given.
    need_ceiling = ("ceiling" not in merged.columns) or merged["ceiling"].isna().all()
    need_floor = ("floor" not in merged.columns) or merged["floor"].isna().all()
    if need_ceiling or need_floor:
        ceiling, floor = derive_ceiling_floor(merged, report)
        if need_ceiling:
            merged["ceiling"] = ceiling
        if need_floor:
            merged["floor"] = floor
    merged["ceiling"] = merged["ceiling"].fillna(merged["proj"])
    merged["floor"] = merged["floor"].fillna(merged["proj"])

    # Ceiling never below the projection, floor never above it.
    merged["ceiling"] = merged[["ceiling", "proj"]].max(axis=1)
    merged["floor"] = merged[["floor", "proj"]].min(axis=1)

    for col in ("implied_team_score", "batting_order"):
        if col not in merged.columns:
            merged[col] = np.nan
    if "injury_status" not in merged.columns:
        merged["injury_status"] = ""
    merged["injury_status"] = merged["injury_status"].fillna("").astype(str)
    if "is_listed_starter" not in merged.columns:
        merged["is_listed_starter"] = pd.NA

    merged["salary"] = merged["salary"].astype(int)
    merged = merged.drop(columns=["_row"], errors="ignore").reset_index(drop=True)
    report.matched = len(merged)

    if verbose:
        print(report.render(), file=sys.stderr)
        n_games = merged["game_id"].nunique()
        if n_games < MIN_DISTINCT_GAMES:
            print(f"[warn] pool covers only {n_games} game(s); DK requires players "
                  f"from at least {MIN_DISTINCT_GAMES}.", file=sys.stderr)

    return merged, report


def filter_availability(pool, exclude_injured=True, require_confirmed_order=False,
                        require_starting_pitcher=True, verbose=True):
    """
    Drop players who won't accumulate points today.

    This is the single biggest edge a projections file provides over DK's own
    season averages: it knows who is actually in the lineup. A benched player
    scores zero, and the optimizer hunts cheap players with good numbers -
    exactly where the benched ones sit.
    """
    keep = pd.Series(True, index=pool.index)
    notes = []

    if exclude_injured and "injury_status" in pool.columns:
        status = pool["injury_status"].fillna("").astype(str).str.upper()
        out = status.isin(OUT_STATUSES)
        if out.any():
            notes.append(f"dropped {int(out.sum())} player(s) listed OUT/IL")
            keep &= ~out
        dtd = status.eq("DTD")
        if dtd.any():
            notes.append(f"{int(dtd.sum())} player(s) are day-to-day (kept - "
                         f"use --exclude to remove any you don't trust)")

    if require_starting_pitcher and "is_listed_starter" in pool.columns:
        flags = pool["is_listed_starter"]
        if flags.notna().any():
            not_starting = pool["is_pitcher"] & (flags != True)  # noqa: E712
            if not_starting.any():
                notes.append(f"dropped {int(not_starting.sum())} pitcher(s) not "
                             f"listed as today's starter")
                keep &= ~not_starting

    if require_confirmed_order and "batting_order" in pool.columns:
        unconfirmed = (~pool["is_pitcher"]) & pool["batting_order"].isna()
        if unconfirmed.any():
            notes.append(f"dropped {int(unconfirmed.sum())} hitter(s) with no "
                         f"confirmed batting order")
            keep &= ~unconfirmed

    if verbose:
        for note in notes:
            print(f"[info] {note}", file=sys.stderr)

    return pool[keep].reset_index(drop=True)
