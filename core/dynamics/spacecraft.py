"""6-DOF spacecraft dynamics around a central body (Earth, Sun or Bennu).

State  x = [r (3), v (3), q (4), omega (3), m (1), m_prop_used (1)]   (15 values, SI)
Control u = [f_body (3) thrust force N, tau (3) torque N m]

Translation (lecture L1 eq. 2, central-body form):
    r'' = gravity(central + third bodies [+ J2]) + drag + SRP + R(q) f_body / m
Attitude (lecture L1 eqs. 3-4 with quaternions): q' = 1/2 q (x) [0, w],  J w' = tau + tau_gg - w x J w
Propellant: m_prop_used' = |f| / (Isp g0); mass decreases only when unlimited_fuel is False.
"""

from dataclasses import dataclass, field

import numpy as np

from core.constants import BENNU, G, G0, GM, RADIUS
from core.dynamics import attitude as att
from core.dynamics.asteroid import bennu_attitude
from core.dynamics.gravity import NBody, point_mass, third_body
from core.dynamics.perturbations import cylindrical_shadow, drag, srp

IDX_R, IDX_V, IDX_Q, IDX_W, IDX_M, IDX_MP = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), 13, 14
EARTH_PERTURBERS = ["sun", "moon", "venus", "mars_bary", "jupiter_bary"]
SUN_PERTURBERS = ["mercury", "venus", "earth", "moon", "mars_bary", "jupiter_bary", "saturn_bary"]
BENNU_PERTURBERS = ["sun", "earth", "moon", "jupiter_bary"]


@dataclass
class SpacecraftParams:
    mass: float = 10_000.0  # kg (dry + payload; propellant tracked separately in unlimited-fuel mode)
    inertia: np.ndarray = field(default_factory=lambda: np.diag([2.4e4, 2.1e4, 1.8e4]))  # kg m^2
    isp: float = 3000.0  # s (solar-electric propulsion)
    cd: float = 2.2
    drag_area: float = 20.0  # m^2
    cr: float = 1.3
    srp_area: float = 60.0  # m^2 (solar arrays)
    unlimited_fuel: bool = True


def initial_state(r, v, q=(1, 0, 0, 0), omega=(0, 0, 0), mass=10_000.0):
    return np.concatenate([r, v, q, omega, [mass, 0.0]]).astype(float)


class SpacecraftDynamics:
    """central: 'earth' | 'sun' | 'bennu'. For 'bennu', pass `bennu_state(t) -> [r, v]` (barycentric)
    and optionally `bennu_gravity(rho_body) -> accel_body` (e.g. Polyhedron.acceleration)."""

    def __init__(self, eph, central, params=None, perturbers=None, j2=True, use_drag=True, use_srp=True,
                 gravity_gradient_torque=True, bennu_state=None, bennu_gravity=None):
        self.eph, self.central = eph, central
        self.p = params or SpacecraftParams()
        self.J_inv = np.linalg.inv(self.p.inertia)
        self.use_drag = use_drag and central == "earth"
        self.use_srp, self.use_gg = use_srp, gravity_gradient_torque
        if central == "bennu":
            if bennu_state is None:
                raise ValueError("central='bennu' needs bennu_state(t)")
            self.bennu_state = bennu_state
            self.bennu_gravity = bennu_gravity
            self.mu_c = BENNU["gm"]
            self.grav = NBody(eph, central="bennu", bodies=BENNU_PERTURBERS if perturbers is None else perturbers,
                              center_pos=lambda t: bennu_state(t)[:3], center_mu=BENNU["gm"])
        else:
            default = EARTH_PERTURBERS if central == "earth" else SUN_PERTURBERS
            self.grav = NBody(eph, central=central, bodies=default if perturbers is None else perturbers,
                              earth_j2=j2)
            self.mu_c = GM[central]

    def center_pos(self, t):
        return self.bennu_state(t)[:3] if self.central == "bennu" else self.eph.pos(self.central, t)

    def gravity(self, t, r, m):
        if self.central != "bennu":
            return self.grav.accel(t, r)
        a = np.zeros(3)
        rc = self.center_pos(t)
        for (b, mu), rb in zip(self.grav.perturbers, self.grav.body_positions(t)):
            a += third_body(r, rb - rc, mu)
        if self.bennu_gravity is None:
            a += point_mass(r, BENNU["gm"])
        else:
            Rb = bennu_attitude(t)
            a += Rb @ self.bennu_gravity(Rb.T @ r)
        return a + point_mass(r, G * m)  # Bennu is also pulled toward the spacecraft (relative motion)

    def accel_parts(self, t, x, f_body):
        r, v, q, m = x[IDX_R], x[IDX_V], x[IDX_Q], x[IDX_M]
        parts = {"gravity": self.gravity(t, r, m), "thrust": att.q_to_R(q) @ f_body / m}
        if self.use_drag:
            parts["drag"] = drag(r, v, self.p.cd * self.p.drag_area / m)
        if self.use_srp:
            r_bary = self.center_pos(t) + r
            sun = self.eph.pos("sun", t)
            nu = 1.0
            if self.central == "earth":
                nu = cylindrical_shadow(r_bary, sun, self.eph.pos("earth", t), RADIUS["earth"])
            parts["srp"] = srp(r_bary - sun, self.p.cr * self.p.srp_area / m, nu)
        return parts

    def rhs(self, t, x, u):
        u = np.zeros(6) if u is None else np.asarray(u, float)
        f_body, tau = u[:3], u[3:6]
        q, w, m = x[IDX_Q], x[IDX_W], x[IDX_M]
        a = sum(self.accel_parts(t, x, f_body).values())
        torque = tau.copy()
        if self.use_gg:
            torque += att.gravity_gradient_torque(x[IDX_R], q, self.p.inertia, self.mu_c)
        mdot_prop = np.linalg.norm(f_body) / (self.p.isp * G0)
        return np.concatenate([x[IDX_V], a, att.q_dot(q, w), att.omega_dot(w, self.p.inertia, torque, self.J_inv),
                               [0.0 if self.p.unlimited_fuel else -mdot_prop, mdot_prop]])
