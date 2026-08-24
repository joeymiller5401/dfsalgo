"""
Load a DK salary export + your projections into a single player pool.

The two jobs here are both places the old version failed silently:

1. Reading DK's own quirks (SP/RP vs P, Showdown files, "Game Info" parsing).
2. Matching your projections onto DK's player list without corrupting rows
   when two players share a name.
"""

import re
import sys
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from .rules import MIN_DISTINCT_GAMES, SHOWDOWN_MARKERS, canonical_team, normalize_positions

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_name(name):
    """
    Build a match key that survives the formatting differences between DK and
    every projection source: accents, punctuation, and generational suffixes.

        "José Ramírez"    -> "jose ramirez"
        "Ronald Acuna Jr." -> "ronald acuna"
    """
    if name is None:
        return ""
    text = str(name).strip()
    # Strip accents: "José" -> "Jose"
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
        return "\n".join(lines)


def _parse_game_info(game_info, team_canon):
    """
    DK's "Game Info" looks like "NYY@BOS 07:05PM ET" (AWAY@HOME).

    Returns (opponent, game_id). game_id is a stable key for the matchup so we
    can enforce DK's 2-game rule and correlate players in the same game.
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
        # If Roster Position is Showdown-flavored, fall back to Position.
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
    """Read your projections file into normalized columns."""
    proj = pd.read_csv(path)

    name_col = find_col(proj, "Name", "Player")
    if name_col is None:
        raise ValueError("Projections file needs a 'Name' column matching DK player names.")
    value_col = find_col(proj, "Proj", "Projection", "FPTS", "Points", "Fpts")
    if value_col is None:
        raise ValueError("Projections file needs a 'Proj' (or 'Projection'/'FPTS') column.")

    ceil_col = find_col(proj, "Ceiling", "Ceil")
    floor_col = find_col(proj, "Floor")
    own_col = find_col(proj, "Ownership", "Own", "Own%", "pOwn")
    team_col = find_col(proj, "Team", "TeamAbbrev")
    order_col = find_col(proj, "Order", "BattingOrder", "Batting Order", "Lineup Slot")

    out = pd.DataFrame({
        "name_key": proj[name_col].map(normalize_name),
        "proj": pd.to_numeric(proj[value_col], errors="coerce"),
    })
    out["ceiling"] = pd.to_numeric(proj[ceil_col], errors="coerce") if ceil_col else pd.NA
    out["floor"] = pd.to_numeric(proj[floor_col], errors="coerce") if floor_col else pd.NA
    out["ownership"] = pd.to_numeric(proj[own_col], errors="coerce") if own_col else 0.0
    out["batting_order"] = pd.to_numeric(proj[order_col], errors="coerce") if order_col else pd.NA
    out["proj_team"] = proj[team_col].map(canonical_team) if team_col else ""

    # Missing ceiling/floor fall back to the point estimate.
    out["ceiling"] = pd.to_numeric(out["ceiling"], errors="coerce").fillna(out["proj"])
    out["floor"] = pd.to_numeric(out["floor"], errors="coerce").fillna(out["proj"])
    out["ownership"] = pd.to_numeric(out["ownership"], errors="coerce").fillna(0.0)
    out["has_team"] = bool(team_col)
    return out


def load_pool(dk_salaries_path, projections_path, verbose=True):
    """
    Merge DK salaries with your projections into the pool the optimizer uses.

    Returns (pool_dataframe, LoadReport).
    """
    dk = load_dk_salaries(dk_salaries_path)
    proj = load_projections(projections_path)
    report = LoadReport(dk_rows=len(dk), projection_rows=len(proj))

    has_team = bool(proj["has_team"].iloc[0]) if len(proj) else False
    proj = proj.drop(columns=["has_team"])

    # Drop players with no Classic roster slot (e.g. DH-only) before matching.
    no_slot = dk[dk["positions"].apply(len) == 0]
    report.no_valid_position = sorted(no_slot["name"].tolist())
    dk = dk[dk["positions"].apply(len) > 0].copy()

    # --- Name matching -----------------------------------------------------
    # Two players can share a name (Luis Garcia, Will Smith, Josh Bell). Merging
    # on name alone produces a cross-product with swapped projections, so any
    # name that is duplicated must be disambiguated by team.
    dk_dupes = set(dk["name_key"][dk["name_key"].duplicated(keep=False)])
    proj_dupes = set(proj["name_key"][proj["name_key"].duplicated(keep=False)])
    contested = dk_dupes | proj_dupes

    dk = dk.reset_index(drop=True)
    dk["_row"] = range(len(dk))

    if has_team:
        proj = proj.drop_duplicates(subset=["name_key", "proj_team"], keep="first")
        merged = dk.merge(proj, on="name_key", how="left", suffixes=("", "_p"))
        # A contested name only counts as matched when the teams agree. Rows that
        # disagree are invalidated (not dropped) so they surface as unmatched
        # instead of vanishing from the report.
        is_contested = merged["name_key"].isin(contested)
        team_mismatch = is_contested & (merged["proj_team"] != merged["team"])
        merged.loc[team_mismatch, ["proj", "ceiling", "floor", "ownership", "batting_order"]] = None
        # One row per DK player, preferring the one that actually matched.
        merged["_matched"] = merged["proj"].notna()
        merged = (merged.sort_values(["_row", "_matched"], ascending=[True, False])
                        .drop_duplicates(subset=["dk_id"], keep="first")
                        .drop(columns=["_matched"]))
        still_bad = []
    else:
        unique_proj = proj.drop_duplicates(subset=["name_key"], keep="first")
        merged = dk.merge(unique_proj, on="name_key", how="left", suffixes=("", "_p"))
        # Can't tell duplicated names apart without a team column - drop them
        # rather than assign one player's projection to another.
        clash = merged[merged["name_key"].isin(contested)]
        still_bad = sorted(f"{n} ({t})" for n, t in zip(clash["name"], clash["team"]))
        merged = merged[~merged["name_key"].isin(contested)].copy()
        merged = merged.sort_values("_row")

    report.ambiguous_names = still_bad

    unmatched = merged[merged["proj"].isna()]
    report.unmatched_names = sorted(unmatched["name"].tolist())
    merged = merged[merged["proj"].notna()].copy()

    bad_salary = merged[merged["salary"].isna()]
    report.missing_projection_value = sorted(bad_salary["name"].tolist())
    merged = merged[merged["salary"].notna()].copy()

    # Ceiling should never sit below the projection, floor never above it -
    # a bad projections file otherwise makes the mode objectives nonsense.
    merged["ceiling"] = merged[["ceiling", "proj"]].max(axis=1)
    merged["floor"] = merged[["floor", "proj"]].min(axis=1)

    merged["salary"] = merged["salary"].astype(int)
    merged = merged.drop(columns=["_row"]).reset_index(drop=True)
    report.matched = len(merged)

    if verbose:
        print(report.render(), file=sys.stderr)

    if len(merged):
        n_games = merged["game_id"].nunique()
        if n_games < MIN_DISTINCT_GAMES:
            print(f"[warn] pool covers only {n_games} game(s); DK requires players "
                  f"from at least {MIN_DISTINCT_GAMES}.", file=sys.stderr)

    return merged, report
