"""DK-specific parsing rules."""

import pytest

from mlb_dfs.rules import canonical_team, normalize_positions


@pytest.mark.parametrize("raw,expected", [
    ("P", ["P"]),
    ("SP", ["P"]),          # real DK exports say SP/RP, not P
    ("RP", ["P"]),
    ("1B/3B", ["1B", "3B"]),
    ("SS/3B", ["SS", "3B"]),
    ("OF", ["OF"]),
    ("LF", ["OF"]),
    ("DH", []),             # no DH slot in Classic
    ("", []),
    ("nan", []),
])
def test_normalize_positions(raw, expected):
    assert normalize_positions(raw) == expected


@pytest.mark.parametrize("variant,canon", [
    ("WSH", "WAS"), ("WSN", "WAS"),
    ("CHW", "CWS"), ("CHA", "CWS"),
    ("SDP", "SD"), ("SFG", "SF"), ("KCR", "KC"), ("TBR", "TB"),
    ("nyy", "NYY"), ("ATH", "OAK"),
    ("BOS", "BOS"),
])
def test_canonical_team(variant, canon):
    assert canonical_team(variant) == canon
