"""Time conversions. Internal time is TDB seconds past J2000 (the Horizons time scale)."""

import warnings
from functools import lru_cache

import numpy as np
from astropy.time import Time
from erfa import ErfaWarning

# UTC beyond the published leap-second table is "dubious" (future leap seconds are unknown);
# the ~1 s ambiguity is irrelevant for display, so silence the warning.
warnings.filterwarnings("ignore", category=ErfaWarning)

from core.constants import DAY, J2000_JD


def jd_to_t(jd_tdb):
    """Julian date (TDB) -> seconds past J2000 (TDB)."""
    return (np.asarray(jd_tdb, dtype=float) - J2000_JD) * DAY


def t_to_jd(t):
    return np.asarray(t, dtype=float) / DAY + J2000_JD


def iso_to_t(iso, scale="tdb"):
    """'YYYY-MM-DD[ HH:MM[:SS]]' in the given scale -> seconds past J2000 TDB."""
    return float(jd_to_t(Time(iso, scale=scale).tdb.jd))


@lru_cache(maxsize=4096)
def _t_to_utc_cached(t_rounded):
    return Time(t_to_jd(t_rounded), format="jd", scale="tdb").utc.iso[:16]


def t_to_utc_str(t):
    """Seconds past J2000 TDB -> 'YYYY-MM-DD HH:MM' UTC (cached to the minute)."""
    return _t_to_utc_cached(round(float(t) / 60.0) * 60.0)


def t_to_year(t):
    """Seconds past J2000 -> decimal year (for plot axes). J2000 is 2000-01-01 12:00."""
    return 2000.0 + (np.asarray(t, dtype=float) / DAY + 0.5) / 365.25
