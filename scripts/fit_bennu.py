"""Estimate Bennu's non-gravitational parameters (A2, AMRAT) against JPL's trajectory, 2026-2059.

    uv run python -m scripts.fit_bennu        -> data/bennu_nongrav_fit.json
The fit window ends before the 2060 Earth approach, which amplifies model errors downstream.
"""

import json

from core.dynamics.asteroid import FIT_FILE, fit_nongrav
from core.ephemeris import Ephemeris
from core.timeutil import iso_to_t


def main():
    eph = Ephemeris()
    res = fit_nongrav(eph, iso_to_t("2026-01-01 00:00"), iso_to_t("2059-12-31 00:00"))
    FIT_FILE.write_text(json.dumps(res, indent=2))
    print(f"A2 = {res['a2']:.5e} au/d^2 (literature {res['literature']['a2']:.4e}), "
          f"AMRAT = {res['amrat']:.4e} m^2/kg, rms {res['rms_m'] / 1e3:.3f} km -> {FIT_FILE}")


if __name__ == "__main__":
    main()
