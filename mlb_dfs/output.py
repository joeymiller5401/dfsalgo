"""Format lineups for DraftKings bulk upload and summarize a portfolio."""

from collections import Counter

import pandas as pd

from .rules import UPLOAD_SLOT_ORDER


def lineup_players(lineup):
    """Player indices in a lineup, ignoring slot assignment."""
    return [i for i, _ in lineup]


def order_for_upload(pool, lineup):
    """
    Arrange a lineup into DK's upload order: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF.
    Each cell is "Name (ID)", which is what DK's own export uses.
    """
    by_slot = {}
    for i, slot in lineup:
        by_slot.setdefault(slot, []).append(i)
    for slot in by_slot:
        # Stable, readable ordering within a slot: most expensive first.
        by_slot[slot].sort(key=lambda i: -int(pool.at[i, "salary"]))

    cells, taken = [], Counter()
    for slot in UPLOAD_SLOT_ORDER:
        i = by_slot[slot][taken[slot]]
        taken[slot] += 1
        cells.append(f"{pool.at[i, 'name']} ({pool.at[i, 'dk_id']})")
    return cells


def to_upload_dataframe(pool, lineups):
    """
    DK-ready frame: exactly the 10 roster columns, nothing else.

    Extra columns can trip DK's bulk importer, so stats live in the summary
    frame instead.
    """
    rows = [order_for_upload(pool, lineup) for lineup in lineups]
    return pd.DataFrame(rows, columns=list(UPLOAD_SLOT_ORDER))


def summarize(pool, lineups, sim_metrics=None):
    """One row per lineup: salary, projection, stack shape, sim distribution."""
    rows = []
    for n, lineup in enumerate(lineups):
        idxs = lineup_players(lineup)
        hitters = [i for i in idxs if not bool(pool.at[i, "is_pitcher"])]
        team_counts = Counter(pool.at[i, "team"] for i in hitters)
        top = team_counts.most_common(2)
        stack = "-".join(str(count) for _, count in top if count >= 2) or "none"
        stack_teams = "/".join(team for team, count in top if count >= 2) or "-"

        row = {
            "Lineup": n + 1,
            "Salary": int(sum(int(pool.at[i, "salary"]) for i in idxs)),
            "Proj": round(float(sum(float(pool.at[i, "proj"]) for i in idxs)), 2),
            "Stack": stack,
            "StackTeams": stack_teams,
            "Games": len(set(pool.at[i, "game_id"] for i in idxs)),
            "Own%": round(float(sum(float(pool.at[i, "ownership"]) for i in idxs)), 1),
        }
        if sim_metrics is not None:
            for key, value in sim_metrics[n].items():
                row[key] = round(value, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def exposures(pool, lineups):
    """How often each player appears across the portfolio."""
    counts = Counter()
    for lineup in lineups:
        for i in lineup_players(lineup):
            counts[i] += 1
    total = max(1, len(lineups))
    rows = [{
        "Player": pool.at[i, "name"],
        "Team": pool.at[i, "team"],
        "Pos": "/".join(pool.at[i, "positions"]),
        "Salary": int(pool.at[i, "salary"]),
        "Proj": float(pool.at[i, "proj"]),
        "Lineups": count,
        "Exposure%": round(100.0 * count / total, 1),
    } for i, count in counts.most_common()]
    return pd.DataFrame(rows)
