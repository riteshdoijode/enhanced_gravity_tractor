"""Cached JPL ephemerides with cubic Hermite interpolation (position + velocity nodes).

All public methods take t = seconds past J2000 (TDB) and return SI units in the ICRF frame,
relative to the Solar System Barycenter (SSB) unless stated otherwise.
"""

import json

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from core.constants import DATA_DIR

EPHEM_FILE = DATA_DIR / "ephemeris.npz"
META_FILE = DATA_DIR / "ephemeris_meta.json"


class UniformHermite:
    """Cubic Hermite interpolation on uniformly spaced nodes: O(1) lookup, ~5x faster than scipy's
    general spline for the scalar-time calls made inside ODE right-hand sides."""

    def __init__(self, t, r, v):
        self.t0, self.h, self.n = float(t[0]), float(t[1] - t[0]), len(t)
        self.r, self.m = r, v * self.h  # tangents scaled to the unit interval

    def __call__(self, t, deriv=False):
        x = (np.asarray(t, float) - self.t0) / self.h
        i = np.clip(np.floor(x).astype(int), 0, self.n - 2)
        s = (x - i)[..., None]
        p0, p1, m0, m1 = self.r[i], self.r[i + 1], self.m[i], self.m[i + 1]
        s2 = s * s
        if not deriv:
            s3 = s2 * s
            return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (3 * s2 - 2 * s3) * p1 + (s3 - s2) * m1
        return ((6 * s2 - 6 * s) * (p0 - p1) + (3 * s2 - 4 * s + 1) * m0 + (3 * s2 - 2 * s) * m1) / self.h


class _Derivative:
    def __init__(self, interp):
        self.interp = interp

    def __call__(self, t):
        return self.interp(t, deriv=True)


class Ephemeris:
    def __init__(self, path=EPHEM_FILE, meta_path=META_FILE):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `uv run python -m scripts.fetch_data` first")
        raw = np.load(path)
        self.meta = json.loads(meta_path.read_text())
        self.bodies = list(self.meta["centers"].keys())
        self.center = self.meta["centers"]
        self._t, self._pos, self._vel = {}, {}, {}
        for b in self.bodies:
            t, r, v = raw[f"{b}_t"], raw[f"{b}_r"], raw[f"{b}_v"]
            self._t[b] = t
            dt = np.diff(t)
            if np.allclose(dt, dt[0], rtol=0, atol=1e-3):
                self._pos[b] = UniformHermite(t, r, v)
                self._vel[b] = _Derivative(self._pos[b])
            else:  # Bennu: daily nodes plus hourly windows around Earth approaches
                self._pos[b] = CubicHermiteSpline(t, r, v, axis=0)
                self._vel[b] = self._pos[b].derivative()

    def with_bennu(self, nominal):
        """A view of these ephemerides in which body "bennu" is the given trajectory (e.g. the keyhole
        scenario of core/scenario.py), barycentric, over the trajectory's own span. Other bodies are shared."""
        import copy

        other = copy.copy(self)
        other._t, other._pos, other._vel = dict(self._t), dict(self._pos), dict(self._vel)
        other.center = dict(self.center)
        lo, hi = nominal.span
        other._t["bennu"] = nominal.t
        other._pos["bennu"] = lambda t: nominal.state(np.clip(t, lo, hi))[..., :3]
        other._vel["bennu"] = lambda t: nominal.state(np.clip(t, lo, hi))[..., 3:]
        other.center["bennu"] = "ssb"
        return other

    def span(self, body):
        return float(self._t[body][0]), float(self._t[body][-1])

    def nodes(self, body):
        """Raw Horizons node times for a body (s past J2000)."""
        return self._t[body]

    def _local(self, body, t, deriv):
        return (self._vel if deriv else self._pos)[body](t)

    def pos(self, body, t):
        """Barycentric position [m] (bodies stored relative to Earth or Sun are shifted to the SSB)."""
        r = self._local(body, t, False)
        c = self.center[body]
        return r if c == "ssb" else r + self._local(c, t, False)

    def vel(self, body, t):
        v = self._local(body, t, True)
        c = self.center[body]
        return v if c == "ssb" else v + self._local(c, t, True)

    def state(self, body, t):
        return np.concatenate([self.pos(body, t), self.vel(body, t)], axis=-1)

    def rel_pos(self, body, center, t):
        """Position of `body` relative to `center` [m] (exact stored vector when available)."""
        if self.center.get(body) == center:
            return self._local(body, t, False)
        return self.pos(body, t) - self.pos(center, t)

    def rel_vel(self, body, center, t):
        if self.center.get(body) == center:
            return self._local(body, t, True)
        return self.vel(body, t) - self.vel(center, t)
