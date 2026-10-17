"""Minimal JPL Horizons / SBDB API client (https://ssd-api.jpl.nasa.gov/doc/horizons.html)."""

import time

import numpy as np
import requests

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
MAX_ROWS = 80_000  # Horizons caps output length; stay below it per request


def _get(url, params, retries=4, timeout=300):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def vectors(command, center, start, stop, step):
    """Fetch ICRF state vectors. Returns (jd_tdb [N], pos_km [N,3], vel_kms [N,3]).

    start/stop are 'YYYY-MM-DD[ HH:MM]' (TDB), step e.g. '1 d', '6 h', '1 h'.
    """
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'",
        "VEC_TABLE": "'2'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
        "TIME_TYPE": "'TDB'",
    }
    result = _get(HORIZONS_URL, params)["result"]
    if "$$SOE" not in result:
        raise RuntimeError(f"Horizons returned no data for {command}@{center}:\n{result[:2000]}")
    rows = result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    data = np.array([[float(f) for i, f in enumerate(row.split(",")[:8]) if i != 1] for row in rows])
    return data[:, 0], data[:, 1:4], data[:, 4:7]


def vectors_chunked(command, center, start_jd, stop_jd, step_days, step_str):
    """Like vectors() but split into requests of at most MAX_ROWS rows. Dates given as JD (TDB)."""
    from astropy.time import Time

    span = MAX_ROWS * step_days
    jds, rs, vs = [], [], []
    a = start_jd
    while a < stop_jd - 1e-9:
        b = min(a + span, stop_jd)
        s = Time(a, format="jd", scale="tdb").iso[:16]
        e = Time(b, format="jd", scale="tdb").iso[:16]
        jd, r, v = vectors(command, center, s, e, step_str)
        jds.append(jd), rs.append(r), vs.append(v)
        a = b
    jd = np.concatenate(jds)
    jd, idx = np.unique(jd, return_index=True)  # chunk boundaries overlap by one row
    return jd, np.concatenate(rs)[idx], np.concatenate(vs)[idx]


def sbdb(sstr):
    """Small-Body Database record with physical parameters and Earth close approaches."""
    return _get(
        SBDB_URL,
        {"sstr": sstr, "phys-par": "1", "ca-data": "1", "ca-body": "Earth", "full-prec": "1"},
        timeout=60,
    )
