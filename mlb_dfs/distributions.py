"""
Shared distribution math.

Both the projection loader (deriving a ceiling/floor when a source doesn't
supply one) and the simulator model a player's score as a *shifted lognormal*
matched to their projected mean and spread. Keeping the math in one place
means a derived ceiling always means the same thing the simulator means by it.
"""

import numpy as np

# Typical coefficient of variation for a DK MLB score, by role. Hitters are
# far streakier than pitchers: a hitter can go 0-for-4 (0 points) or hit two
# home runs, while a starter's floor is set by innings pitched.
DEFAULT_CV_HITTER = 0.85
DEFAULT_CV_PITCHER = 0.45

# Lognormals are strictly positive. Hitters bottom out near zero, so no shift.
# Pitchers can post negative scores (earned runs), so their distribution is
# shifted down to allow it.
HITTER_SHIFT = 0.0
PITCHER_SHIFT = -5.0

# Standard normal quantiles for the ceiling/floor convention used here:
# ceiling = 85th percentile, floor = 15th percentile.
Z_P85 = 1.0364334
Z_P15 = -1.0364334

# For a normal, the p85-p15 span is ~2.07 sigma. Used to invert a supplied
# ceiling/floor pair back into a standard deviation.
P85_P15_SPAN = Z_P85 - Z_P15


def lognormal_params(mean, std, shift):
    """(mu, sigma) of the underlying normal for a shifted lognormal."""
    centered = np.maximum(np.asarray(mean, dtype=float) - shift, 1e-3)
    std = np.maximum(np.asarray(std, dtype=float), 1e-3)
    sigma = np.sqrt(np.log(1.0 + (std ** 2) / (centered ** 2)))
    mu = np.log(centered) - 0.5 * sigma ** 2
    return mu, sigma


def lognormal_quantile(mean, std, shift, z):
    """Value at standard-normal quantile `z` for a shifted lognormal."""
    mu, sigma = lognormal_params(mean, std, shift)
    return shift + np.exp(mu + sigma * z)


def role_defaults(is_pitcher):
    """Per-player (cv, shift) arrays keyed off pitcher/hitter."""
    is_pitcher = np.asarray(is_pitcher, dtype=bool)
    cv = np.where(is_pitcher, DEFAULT_CV_PITCHER, DEFAULT_CV_HITTER)
    shift = np.where(is_pitcher, PITCHER_SHIFT, HITTER_SHIFT)
    return cv, shift
