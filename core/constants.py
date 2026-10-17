"""Physical constants and fixed mission data. SI units (m, s, kg) unless noted."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"

G = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
AU = 1.495978707e11  # m (IAU 2012)
DAY = 86400.0  # s
YEAR = 365.25 * DAY  # Julian year, s
G0 = 9.80665  # m/s^2, standard gravity (for Isp)
J2000_JD = 2451545.0  # JD (TDB) of the J2000 epoch
LUNAR_DISTANCE = 384_400e3  # m, mean Earth-Moon distance (display only)

# Gravitational parameters GM [m^3/s^2], JPL DE440 (Park et al. 2021, AJ 161:105).
# "_bary" entries are planet-system barycenters (planet + moons).
GM = {
    "sun": 1.32712440041279419e20,
    "mercury": 2.2031868551e13,
    "venus": 3.24858592e14,
    "earth": 3.98600435507e14,
    "moon": 4.902800118e12,
    "mars_bary": 4.2828375816e13,
    "jupiter_bary": 1.267127641e17,
    "saturn_bary": 3.79405848418e16,
    "uranus_bary": 5.7945564e15,
    "neptune_bary": 6.83652710058e15,
    # Pluto system (dwarf planet since 2006): Horizons GM(Pluto) 869.326 km^3/s^2 x (1 + Charon/Pluto 0.122)
    "pluto_bary": 869.326e9 * 1.122,
    # largest main-belt asteroids (JPL small-body perturber set SB441-N16, approx.)
    "ceres": 6.26284e10,
    "vesta": 1.72882e10,
    "pallas": 1.3665e10,
    "hygiea": 5.78e9,
}

# Body radii [m], display and J2 use
RADIUS = {
    "sun": 6.957e8,
    "earth": 6.3781366e6,
    "moon": 1.7374e6,
    "mercury": 2.4397e6,
    "venus": 6.0518e6,
    "mars_bary": 3.3895e6,
    "jupiter_bary": 6.9911e7,
    "saturn_bary": 5.8232e7,
    "uranus_bary": 2.5362e7,
    "neptune_bary": 2.4622e7,
}
EARTH_J2 = 1.08262668e-3  # EGM2008
EARTH_OMEGA = 7.2921150e-5  # rad/s, Earth rotation rate

# JPL Horizons body codes: name -> (COMMAND, CENTER). Moon is fetched geocentric. Bennu is fetched
# heliocentric: before ~2031-06 Horizons' SSB-centered Bennu states carry a ~116 km offset relative to
# the DE441 barycenter (its trajectory was integrated with DE424); Sun-relative states are consistent.
HORIZONS_BODIES = {
    "sun": ("10", "500@0"),
    "mercury": ("199", "500@0"),
    "venus": ("299", "500@0"),
    "earth": ("399", "500@0"),
    "mars_bary": ("4", "500@0"),
    "jupiter_bary": ("5", "500@0"),
    "saturn_bary": ("6", "500@0"),
    "uranus_bary": ("7", "500@0"),
    "neptune_bary": ("8", "500@0"),
    "pluto_bary": ("9", "500@0"),
    "moon": ("301", "500@399"),
    "ceres": ("1;", "500@10"),
    "vesta": ("4;", "500@10"),
    "pallas": ("2;", "500@10"),
    "hygiea": ("10;", "500@10"),
    "bennu": ("101955;", "500@10"),
}

# Ephemeris span. JPL's Bennu trajectory (sb-101955-118_long) ends 2135-Oct-01.
EPHEM_START = "2026-01-01"
EPHEM_STOP = "2185-01-01"
BENNU_EPHEM_STOP = "2135-09-30"

# --- 101955 Bennu, JPL Small-Body Database (fetched 2026-10-15) ---
BENNU = {
    "gm": 4.8904,  # m^3/s^2 (Chesley et al. 2020, JGR Planets)
    "density": 1194.0,  # kg/m^3 (Daly et al. 2020, Sci. Adv.)
    "diameter": 484.44,  # m, volume-equivalent (Daly et al. 2020)
    "extent": (505.0, 492.0, 456.7),  # m (Daly et al. 2020)
    "rot_period_h": 4.296061,  # h, synodic (Hergenrother et al. 2019, Nat. Commun.)
    "pole_ra_deg": 85.4522,  # spin pole, ICRF
    "pole_dec_deg": -60.3678,
    "albedo": 0.044,
}
BENNU["mass"] = BENNU["gm"] / G
BENNU["radius"] = BENNU["diameter"] / 2.0
BENNU["omega"] = 2.0 * np.pi / (BENNU["rot_period_h"] * 3600.0)  # rad/s

# Earth close approaches of Bennu from SBDB (TDB, distance au, v_rel km/s)
BENNU_CLOSE_APPROACHES = [
    ("2054-09-30 01:00", 0.03930, 5.09),
    ("2060-09-23 00:36", 0.00501, 6.19),
    ("2080-09-22 00:25", 0.01563, 6.39),
    ("2135-09-25 07:49", 0.00136, 6.29),
]

# DART results (Cheng et al. 2023, Nature 616:457) and impactor defaults (lecture 1, slide 11)
DART = {
    "mass_impact": 579.4,  # kg
    "speed": 6144.7,  # m/s
    "beta": 3.61,
    "beta_range": (2.2, 4.9),
}
