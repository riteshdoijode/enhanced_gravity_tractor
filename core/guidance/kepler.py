"""Two-body tools with universal variables (Curtis, Orbital Mechanics for Engineering Students, ch. 3 & 5).

All functions are vectorized: positions/velocities have shape (..., 3), times shape (...).

Kepler propagation (Curtis Alg. 3.4): solve for the universal anomaly chi,
    sqrt(mu) dt = r0 vr0/sqrt(mu) chi^2 C(a chi^2) + (1 - a r0) chi^3 S(a chi^2) + r0 chi,   a = 2/r0 - v0^2/mu
then r = f r0 + g v0, v = fdot r0 + gdot v0.

Lambert (Curtis Alg. 5.2): find v1, v2 so a Keplerian arc goes from r1 to r2 in dt.
    A = sin(dth) sqrt(r1 r2 / (1 - cos dth)),  y(z) = r1 + r2 + A (z S(z) - 1)/sqrt(C(z))
    sqrt(mu) dt = (y/C)^(3/2) S + A sqrt(y)
    f = 1 - y/r1, g = A sqrt(y/mu), gdot = 1 - y/r2;  v1 = (r2 - f r1)/g,  v2 = (gdot r2 - r1)/g
"""

import math

import numpy as np
from numba import njit

from core import parallel
from core.constants import GM

MU_SUN = GM["sun"]
TWO_PI_SQ = 4.0 * np.pi**2


def stumpff(z):
    """Stumpff functions C(z), S(z); series near z = 0 avoid cancellation."""
    with np.errstate(over="ignore", invalid="ignore"):  # huge hyperbolic brackets overflow to inf harmlessly
        return _stumpff(z)


def _stumpff(z):
    z = np.asarray(z, float)
    C, S = np.empty_like(z), np.empty_like(z)
    pos, neg = z > 1e-6, z < -1e-6
    small = ~(pos | neg)
    sp = np.sqrt(z[pos])
    C[pos] = (1 - np.cos(sp)) / z[pos]
    S[pos] = (sp - np.sin(sp)) / sp**3
    sn = np.sqrt(-z[neg])
    C[neg] = (np.cosh(sn) - 1) / (-z[neg])
    S[neg] = (np.sinh(sn) - sn) / sn**3
    zs = z[small]
    C[small] = 0.5 - zs / 24 + zs**2 / 720
    S[small] = 1 / 6 - zs / 120 + zs**2 / 5040
    return C, S


def propagate(r0, v0, dt, mu=MU_SUN, iterations=60, tol=1e-12):
    """Kepler propagation by dt (scalar or array broadcast against the leading dims of r0)."""
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    shape = np.broadcast_shapes(r0.shape[:-1], v0.shape[:-1], np.shape(dt))
    r0 = np.broadcast_to(r0, shape + (3,)).reshape(-1, 3)
    v0 = np.broadcast_to(v0, shape + (3,)).reshape(-1, 3)
    dt = np.broadcast_to(np.asarray(dt, float), shape).reshape(-1).copy()
    r, v = _propagate(r0, v0, dt, mu, iterations, tol)
    return r.reshape(shape + (3,)), v.reshape(shape + (3,))


def _propagate(r0, v0, dt, mu, iterations, tol):
    """Compiled kernel (numba, GIL released), split across the worker threads for large batches."""
    r0 = np.ascontiguousarray(r0, dtype=float)
    v0 = np.ascontiguousarray(v0, dtype=float)
    dt = np.ascontiguousarray(dt, dtype=float)
    r, v = np.empty_like(r0), np.empty_like(v0)

    def run(a, b):
        _kepler_batch(r0[a:b], v0[a:b], dt[a:b], float(mu), int(iterations), float(tol), r[a:b], v[a:b])

    parallel.chunked(run, len(dt), min_chunk=256)
    return r, v


@njit(cache=True, nogil=True)
def _stumpff_s(z):
    if z > 1e-6:
        sp = math.sqrt(z)
        return (1.0 - math.cos(sp)) / z, (sp - math.sin(sp)) / sp**3
    if z < -1e-6:
        sn = math.sqrt(-z)
        return (math.cosh(sn) - 1.0) / (-z), (math.sinh(sn) - sn) / sn**3
    return 0.5 - z / 24.0 + z * z / 720.0, 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


@njit(cache=True, nogil=True)
def _kepler_F(chi, alpha, r0n, vr0, smu, dt):
    z = alpha * chi * chi
    C, S = _stumpff_s(z)
    F = r0n * vr0 / smu * chi * chi * C + (1.0 - alpha * r0n) * chi**3 * S + r0n * chi - smu * dt
    dF = r0n * vr0 / smu * chi * (1.0 - z * S) + (1.0 - alpha * r0n) * chi * chi * C + r0n
    if F != F:  # NaN from overflow in a huge hyperbolic bracket: treat as "far above the root"
        F = math.inf
    return F, dF


@njit(cache=True, nogil=True)
def _kepler_batch(r0, v0, dt_in, mu, iterations, tol, r_out, v_out):
    """Scalar version of _propagate_numpy, one arc at a time (same bracketing and safeguards)."""
    smu = math.sqrt(mu)
    for k in range(dt_in.shape[0]):
        x0, y0, z0 = r0[k, 0], r0[k, 1], r0[k, 2]
        u0, w0, s0 = v0[k, 0], v0[k, 1], v0[k, 2]
        r0n = math.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
        vr0 = (x0 * u0 + y0 * w0 + z0 * s0) / r0n
        alpha = 2.0 / r0n - (u0 * u0 + w0 * w0 + s0 * s0) / mu
        dt = dt_in[k]
        ell = alpha > 1e-12
        if ell:
            dt = np.fmod(dt, 2.0 * math.pi / math.sqrt(mu * alpha**3))
            span = 2.0 * math.pi / math.sqrt(alpha)
        else:
            span = smu * abs(dt) / r0n + 1.0
        if dt >= 0:
            lo, hi = 0.0, span
        else:
            lo, hi = -span, 0.0
        for _ in range(200):  # widen the bracket (hyperbolic arcs) until it contains the root
            Flo = _kepler_F(lo, alpha, r0n, vr0, smu, dt)[0]
            Fhi = _kepler_F(hi, alpha, r0n, vr0, smu, dt)[0]
            grow_hi, grow_lo = Fhi < 0, Flo > 0
            if not (grow_hi or grow_lo):
                break
            if grow_hi:
                hi = hi + 2.0 * (hi - lo)
            if grow_lo:
                lo = lo - 2.0 * (hi - lo)
        if ell:
            chi = min(max(smu * alpha * dt, lo), hi)
        else:
            chi = 0.5 * (lo + hi)
        for _ in range(iterations):
            F, dF = _kepler_F(chi, alpha, r0n, vr0, smu, dt)
            if F < 0:
                lo = chi
            if F > 0:
                hi = chi
            newton = chi - F / dF
            new = newton if (newton > lo and newton < hi) else 0.5 * (lo + hi)
            done = abs(new - chi) <= tol * max(1.0, abs(chi))
            chi = new
            if done:
                break
        z = alpha * chi * chi
        C, S = _stumpff_s(z)
        f = 1.0 - chi * chi / r0n * C
        g = dt - chi**3 * S / smu
        rx, ry, rz = f * x0 + g * u0, f * y0 + g * w0, f * z0 + g * s0
        rn = math.sqrt(rx * rx + ry * ry + rz * rz)
        fdot = smu / (rn * r0n) * (alpha * chi**3 * S - chi)
        gdot = 1.0 - chi * chi / rn * C
        r_out[k, 0], r_out[k, 1], r_out[k, 2] = rx, ry, rz
        v_out[k, 0] = fdot * x0 + gdot * u0
        v_out[k, 1] = fdot * y0 + gdot * w0
        v_out[k, 2] = fdot * z0 + gdot * s0


def _propagate_numpy(r0, v0, dt, mu, iterations, tol):
    """Safeguarded Newton on the universal Kepler equation F(chi) = 0 (vectorized NumPy reference).

    dF/dchi = r(chi) > 0, so F is monotonic: keep a bracket [lo, hi] and fall back to bisection
    whenever a Newton step would leave it (robust for highly eccentric and hyperbolic arcs).
    """
    r0n = np.linalg.norm(r0, axis=-1)
    vr0 = np.sum(r0 * v0, axis=-1) / r0n
    alpha = 2.0 / r0n - np.sum(v0 * v0, axis=-1) / mu
    smu = np.sqrt(mu)
    ell = alpha > 1e-12  # elliptic: whole revolutions change nothing, so solve for dt mod period
    period = np.full_like(dt, np.inf)
    period[ell] = 2 * np.pi / np.sqrt(mu * alpha[ell] ** 3)
    dt = np.where(ell, np.fmod(dt, period), dt)

    def F_dF(chi):
        z = alpha * chi**2
        C, S = stumpff(z)
        with np.errstate(over="ignore", invalid="ignore"):
            F = r0n * vr0 / smu * chi**2 * C + (1 - alpha * r0n) * chi**3 * S + r0n * chi - smu * dt
            dF = r0n * vr0 / smu * chi * (1 - z * S) + (1 - alpha * r0n) * chi**2 * C + r0n
        return np.nan_to_num(F, nan=np.inf), dF

    sgn = np.where(dt >= 0, 1.0, -1.0)
    span = np.where(ell, 2 * np.pi / np.sqrt(np.where(ell, alpha, 1.0)), smu * np.abs(dt) / r0n + 1.0)
    lo, hi = np.where(sgn > 0, 0.0, -span), np.where(sgn > 0, span, 0.0)
    for _ in range(200):  # widen the bracket for hyperbolic arcs until it contains the root
        Flo, _ = F_dF(lo)
        Fhi, _ = F_dF(hi)
        grow_hi, grow_lo = Fhi < 0, Flo > 0
        if not (grow_hi.any() or grow_lo.any()):
            break
        hi = np.where(grow_hi, hi + 2 * (hi - lo), hi)
        lo = np.where(grow_lo, lo - 2 * (hi - lo), lo)
    chi = np.clip(smu * alpha * dt, lo, hi)
    chi = np.where(ell, chi, 0.5 * (lo + hi))
    for _ in range(iterations):
        F, dF = F_dF(chi)
        lo, hi = np.where(F < 0, chi, lo), np.where(F > 0, chi, hi)
        newton = chi - F / dF
        ok = (newton > lo) & (newton < hi)
        new = np.where(ok, newton, 0.5 * (lo + hi))
        done = np.abs(new - chi) <= tol * np.maximum(1.0, np.abs(chi))
        chi = new
        if np.all(done):
            break
    z = alpha * chi**2
    C, S = stumpff(z)
    f = 1 - chi**2 / r0n * C
    g = dt - chi**3 * S / smu
    r = f[..., None] * r0 + g[..., None] * v0
    rn = np.linalg.norm(r, axis=-1)
    fdot = smu / (rn * r0n) * (alpha * chi**3 * S - chi)
    gdot = 1 - chi**2 / rn * C
    v = fdot[..., None] * r0 + gdot[..., None] * v0
    return r, v


def _tof(z, r1n, r2n, A, mu):
    """Time of flight for universal variable z; y < 0 means 'too fast' (TOF -> 0), C -> 0 means TOF -> inf."""
    C, S = stumpff(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = r1n + r2n + A * (z * S - 1) / np.sqrt(C)
        bad = y < 0
        y = np.where(bad, 0.0, y)
        t = ((y / C) ** 1.5 * S + A * np.sqrt(y)) / np.sqrt(mu)
    t = np.where(bad, 0.0, np.nan_to_num(t, nan=np.inf))
    return np.where(C < 1e-12, np.inf, t), y, C, S  # z -> 4 pi^2: one full revolution, TOF -> infinity


def lambert_batch(r1, r2, dt, **kw):
    """lambert() over a flat batch (n, 3), split across the worker threads (NumPy releases the GIL on
    these large arrays: ~1.6x faster for a 38k-arc porkchop on 6 performance cores)."""
    r1, r2, dt = np.asarray(r1, float), np.asarray(r2, float), np.asarray(dt, float)
    parts = parallel.chunked(lambda a, b: lambert(r1[a:b], r2[a:b], dt[a:b], **kw), len(dt), min_chunk=2000)
    return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])


def lambert(r1, r2, dt, mu=MU_SUN, prograde_axis=(0.0, 0.0, 1.0), n_grid=160, n_bisect=64):
    """Single-revolution Lambert solution(s). Returns v1, v2 (NaN where no solution).

    The transfer direction is prograde about `prograde_axis`. The universal-variable equation is
    bracketed on a z grid (searched from the elliptic end, which picks the low-energy branch when the
    long-way time of flight has a minimum) and then refined by bisection, fully vectorized.
    """
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    shape = np.broadcast_shapes(r1.shape, r2.shape)[:-1]
    r1, r2 = np.broadcast_to(r1, shape + (3,)), np.broadcast_to(r2, shape + (3,))
    dt = np.broadcast_to(np.asarray(dt, float), shape)
    r1n, r2n = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cos_dth = np.clip(np.sum(r1 * r2, -1) / (r1n * r2n), -1.0, 1.0)
    dth = np.arccos(cos_dth)
    long_way = np.cross(r1, r2) @ np.asarray(prograde_axis, float) < 0
    dth = np.where(long_way, 2 * np.pi - dth, dth)
    A = np.sin(dth) * np.sqrt(r1n * r2n / np.maximum(1 - cos_dth, 1e-15))

    z_grid = np.concatenate([np.linspace(-4 * TWO_PI_SQ, 0.0, n_grid // 4, endpoint=False),
                             TWO_PI_SQ * (1 - np.geomspace(1.0, 1e-7, n_grid - n_grid // 4))])
    flat = dt.size
    zg = np.broadcast_to(z_grid, (flat, len(z_grid)))
    T, *_ = _tof(zg, r1n.reshape(-1, 1), r2n.reshape(-1, 1), A.reshape(-1, 1), mu)
    target = dt.reshape(-1, 1)
    above = T > target
    # walk down from the elliptic end: first grid point whose TOF drops to/below the target
    rev = above[:, ::-1]
    k_rev = np.argmin(rev, axis=1)  # index of first False in reversed order
    found = ~rev[np.arange(flat), k_rev] & rev[:, 0]
    k = len(z_grid) - 1 - k_rev  # TOF(z_k) <= dt < TOF(z_{k+1})
    k = np.clip(k, 0, len(z_grid) - 2)
    lo, hi = z_grid[k], z_grid[k + 1]
    r1f, r2f, Af, tf = r1n.ravel(), r2n.ravel(), A.ravel(), dt.ravel()
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        Tm, *_ = _tof(mid, r1f, r2f, Af, mu)
        go_up = Tm < tf
        lo, hi = np.where(go_up, mid, lo), np.where(go_up, hi, mid)
    z = 0.5 * (lo + hi)
    _, y, C, S = _tof(z, r1f, r2f, Af, mu)
    f = 1 - y / r1f
    g = Af * np.sqrt(y / mu)
    gdot = 1 - y / r2f
    r1v, r2v = r1.reshape(-1, 3), r2.reshape(-1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        v1 = (r2v - f[:, None] * r1v) / g[:, None]
        v2 = (gdot[:, None] * r2v - r1v) / g[:, None]
    bad = ~found | (y <= 0) | ~np.isfinite(g) | (np.abs(g) < 1e-9)
    v1[bad], v2[bad] = np.nan, np.nan
    return v1.reshape(shape + (3,)), v2.reshape(shape + (3,))


def ecliptic_pole_icrf():
    """Unit normal of the J2000 ecliptic in ICRF (prograde reference for heliocentric transfers)."""
    eps = np.radians(23.4392911)
    return np.array([0.0, -np.sin(eps), np.cos(eps)])


def lambert_multirev(r1, r2, dt, n_max=3, mu=MU_SUN, prograde_axis=(0.0, 0.0, 1.0), n_grid=96, n_bisect=60):
    """All prograde Lambert solutions with 0..n_max complete revolutions.

    For N >= 1 the universal variable lies in (4 pi^2 N^2, 4 pi^2 (N+1)^2); TOF(z) is U-shaped there
    (infinite at both ends) so each N gives two solutions (left/right branch) when dt exceeds its minimum.
    Returns v1, v2 with shape (2 n_max + 1, ..., 3); row 0 is the single-revolution solution, rows
    2N-1 / 2N the left / right branch for N revolutions (NaN where none exists).
    """
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    shape = np.broadcast_shapes(r1.shape, r2.shape)[:-1]
    r1, r2 = np.broadcast_to(r1, shape + (3,)).reshape(-1, 3), np.broadcast_to(r2, shape + (3,)).reshape(-1, 3)
    dt = np.broadcast_to(np.asarray(dt, float), shape).reshape(-1)
    v1_0, v2_0 = lambert(r1, r2, dt, mu, prograde_axis)
    V1, V2 = [v1_0], [v2_0]
    r1n, r2n = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cos_dth = np.clip(np.sum(r1 * r2, -1) / (r1n * r2n), -1.0, 1.0)
    dth = np.arccos(cos_dth)
    dth = np.where(np.cross(r1, r2) @ np.asarray(prograde_axis, float) < 0, 2 * np.pi - dth, dth)
    A = np.sin(dth) * np.sqrt(r1n * r2n / np.maximum(1 - cos_dth, 1e-15))
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_grid)))  # clustered toward both ends
    u = np.clip(u, 1e-7, 1 - 1e-7)
    for N in range(1, n_max + 1):
        z0, z1 = TWO_PI_SQ * N**2, TWO_PI_SQ * (N + 1) ** 2
        zg = z0 + (z1 - z0) * u
        T, *_ = _tof(zg[None, :], r1n[:, None], r2n[:, None], A[:, None], mu)
        m = np.argmin(T, axis=1)
        tmin = T[np.arange(len(dt)), m]
        exists = dt > tmin
        for branch in ("left", "right"):
            idx = np.arange(n_grid)[None, :]
            on = (idx <= m[:, None]) if branch == "left" else (idx >= m[:, None])
            # left: TOF decreasing in z -> bracket where TOF crosses dt going down; right: increasing
            cross = on & ((T >= dt[:, None]) if branch == "left" else (T <= dt[:, None]))
            if branch == "left":
                k = np.where(cross.any(1), n_grid - 1 - np.argmax(cross[:, ::-1], axis=1), 0)
                k = np.minimum(k, np.maximum(m - 1, 0))
                lo, hi = zg[k], zg[np.minimum(k + 1, n_grid - 1)]
            else:
                k = np.where(cross.any(1), n_grid - 1 - np.argmax(cross[:, ::-1], axis=1), m)
                k = np.minimum(k, n_grid - 2)
                lo, hi = zg[k], zg[k + 1]
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                Tm, *_ = _tof(mid, r1n, r2n, A, mu)
                go_up = (Tm < dt) if branch == "right" else (Tm > dt)
                lo, hi = np.where(go_up, mid, lo), np.where(go_up, hi, mid)
            z = 0.5 * (lo + hi)
            Tz, y, C, S = _tof(z, r1n, r2n, A, mu)
            f = 1 - y / r1n
            g = A * np.sqrt(y / mu)
            gdot = 1 - y / r2n
            with np.errstate(divide="ignore", invalid="ignore"):
                v1 = (r2 - f[:, None] * r1) / g[:, None]
                v2 = (gdot[:, None] * r2 - r1) / g[:, None]
            bad = ~exists | (y <= 0) | ~np.isfinite(g) | (np.abs(Tz - dt) > 1e-6 * dt + 1.0)
            v1[bad], v2[bad] = np.nan, np.nan
            V1.append(v1)
            V2.append(v2)
    return (np.stack(V1).reshape((2 * n_max + 1,) + shape + (3,)),
            np.stack(V2).reshape((2 * n_max + 1,) + shape + (3,)))
