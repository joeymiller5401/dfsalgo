"""
DraftKings MLB *Classic* contest rules.

Everything DK enforces at upload time lives here, so the optimizer can never
produce a lineup the site will reject.
"""

# Roster: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
ROSTER_SLOTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
UPLOAD_SLOT_ORDER = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
TOTAL_PLAYERS = sum(ROSTER_SLOTS.values())  # 10

SALARY_CAP = 50000

# DK rule: no more than 5 *hitters* from any one team (pitchers don't count).
MAX_HITTERS_PER_TEAM = 5

# DK rule: a lineup must include players from at least 2 different games.
MIN_DISTINCT_GAMES = 2

# Max hitters in a lineup (10 roster spots - 2 pitchers). Used as a big-M bound.
MAX_HITTERS = TOTAL_PLAYERS - ROSTER_SLOTS["P"]  # 8

# DK writes pitchers as SP/RP in the "Position" column but "P" in the
# "Roster Position" column. Outfield variants show up in some third-party
# exports. Anything not listed here is ignored (and reported).
POSITION_ALIASES = {
    "P": "P", "SP": "P", "RP": "P", "PITCHER": "P",
    "C": "C", "CATCHER": "C",
    "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
}

# Showdown slates use these roster positions; Classic does not. Seeing them
# means the user exported the wrong contest type.
SHOWDOWN_MARKERS = {"CPT", "UTIL", "CAPTAIN"}

# Team abbreviations differ between DK and most projection sources. Both sides
# get canonicalized to the same token before merging, so WSH/WAS or CHW/CWS
# don't silently fail to match.
_TEAM_ALIAS_GROUPS = [
    ("ARI", "ARZ"),
    ("CWS", "CHW", "CHA"),
    ("CHC", "CHN"),
    ("KC", "KCR", "KCA"),
    ("SD", "SDP", "SDN"),
    ("SF", "SFG", "SFN"),
    ("TB", "TBR", "TBA"),
    ("WAS", "WSH", "WSN", "WAS"),
    ("LAD", "LA", "LAN"),
    ("LAA", "ANA", "ANG"),
    ("NYY", "NYA"),
    ("NYM", "NYN"),
    ("STL", "SLN"),
    ("OAK", "ATH", "SAC"),   # Athletics relocation aliases
    ("MIA", "FLA"),
]

TEAM_ALIASES = {}
for _group in _TEAM_ALIAS_GROUPS:
    _canon = _group[0]
    for _variant in _group:
        TEAM_ALIASES[_variant] = _canon


def canonical_team(abbrev):
    """Normalize a team abbreviation so DK and projection sources agree."""
    if abbrev is None:
        return ""
    t = str(abbrev).strip().upper()
    return TEAM_ALIASES.get(t, t)


def normalize_positions(raw):
    """
    Turn a DK position string into the Classic roster slots a player can fill.

    "SP"      -> ["P"]
    "1B/3B"   -> ["1B", "3B"]
    "OF"      -> ["OF"]

    Unknown tokens (e.g. "DH", which has no Classic slot) are dropped.
    """
    if raw is None:
        return []
    text = str(raw).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return []
    slots = []
    for token in text.replace("/", ",").replace("|", ",").split(","):
        token = token.strip()
        if not token:
            continue
        slot = POSITION_ALIASES.get(token)
        if slot and slot not in slots:
            slots.append(slot)
    return slots
