"""Boulder capture with a robotic arm in MuJoCo (Enhanced Gravity Tractor, Mazanek et al. 2015).

World frame: the local site frame at the boulder (x east, y north, z up = -effective gravity), origin on
the terrain under the boulder. The terrain is a heightfield cut from the OSIRIS-REx 5 cm Nightingale
model (the default site), or ray-cast on the global shape model at any other site the scan picked
(core/proximity/boulders.py: Site). Physics: MuJoCo rigid-body contact; gravity = the effective gravity at the site (polyhedron +
centrifugal, ~7e-5 m/s^2); because this frame rotates with Bennu, every body also gets the Coriolis force
-2 m w x v (at 2 cm/s that is ~20 % of the weight, so it is not negligible).

Bus: free-floating 6-DOF body. Its thrusters (total force limited to the tractor thrust) hold the commanded
position, its control-moment gyros hold the attitude (quaternion PD, torque-limited).
Arm: yaw, shoulder, elbow, wrist + a 3-finger microspine gripper (fingers: position servos; microspine
grip: MuJoCo adhesion actuators). Arm joints use computed-torque control (lecture 1, Lagrange form):
    tau = M(q) (qdd_d + Kd (qd_d - qd) + Kp (q_d - q)) + C(q, qd) qd + G(q)
with M and C qd + G from MuJoCo. After the grip the controller switches to a second model in which the
boulder is a rigid part of the palm (the merged-mass model), so it compensates the payload.
Sequence: deploy -> reach -> grip -> lift (bus thrusters) -> stow -> done.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.assets import tagsite
from core.constants import BENNU

BOULDER_DENSITY = 2000.0  # kg/m^3, a dense boulder (Bennu's bulk density includes macro-porosity)
SITE_XY = np.array([-0.1, -7.3])  # flattest 3 m spot of the Nightingale grid (local frame, m)
HFIELD_HALF = 10.0  # m
ARM = {"upper": 3.0, "fore": 3.0, "wrist": 0.6, "palm_r": 0.45, "finger": 0.9}
ARM_MASS = {"upper": 150.0, "fore": 110.0, "wrist": 40.0, "palm": 30.0, "finger": 15.0, "base": 40.0}
ARM_JOINTS = ["j_yaw", "j_shoulder", "j_elbow", "j_wrist"]
FINGERS = 3
BUS_HALF = np.array([1.2, 1.2, 1.4])
TORQUE_MAX_BUS = 200.0  # N m, control-moment gyros
JOINT_TORQUE_MAX = 5e4  # N m
FINGER_OPEN, FINGER_CLOSED = 1.35, 0.3  # rad; positive swings the claws outward


def _min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4), (60 * s - 180 * s**2 + 120 * s**3)


def boulder_mesh(mass, seed=7, aspect=(1.3, 1.0, 0.75), density=BOULDER_DENSITY):
    """Irregular convex rock: a jittered ellipsoid scaled so its hull has volume mass / density."""
    import trimesh

    rng = np.random.default_rng(seed)
    ico = trimesh.creation.icosphere(subdivisions=2)
    v = ico.vertices * np.asarray(aspect) * (1.0 + 0.12 * rng.uniform(-1, 1, (len(ico.vertices), 1)))
    hull = trimesh.Trimesh(v, process=True).convex_hull
    s = (mass / density / hull.volume) ** (1.0 / 3.0)
    hull.apply_scale(s)
    hull.apply_translation(-hull.center_mass)
    return hull


def site_frame():
    """Local site frame at the boulder: origin (body frame, m), R (local -> body), g_eff (local)."""
    s = tagsite()
    R, o = s["R"], s["origin"]
    xs, H = s["xs"], s["heights"]
    from scipy.interpolate import RegularGridInterpolator

    h = float(RegularGridInterpolator((xs, xs), H)([SITE_XY[1], SITE_XY[0]])[0])
    origin = o + R @ np.array([SITE_XY[0], SITE_XY[1], h])
    return origin, R, R.T @ s["g_eff"]


def _hfield(half=HFIELD_HALF):
    s = tagsite()
    xs, H = s["xs"], s["heights"]
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((xs, xs), H)
    n = int(round(2 * half / 0.1)) + 1
    g = np.linspace(-half, half, n)
    Y, X = np.meshgrid(g + SITE_XY[1], g + SITE_XY[0], indexing="ij")
    h0 = float(f([SITE_XY[1], SITE_XY[0]])[0])
    Z = f(np.column_stack([Y.ravel(), X.ravel()])).reshape(n, n) - h0  # height relative to the site origin
    return g, Z


def _mjcf(params, hfield_rows, boulder_vertices, attached=False):
    """MJCF text. attached=True: controller model with the boulder rigidly inside the palm body."""
    zmin, zmax = params["zmin"], params["zmax"]
    m_bus = params["m_bus"]
    bv = " ".join(f"{x:.5f}" for x in boulder_vertices.ravel())
    finger_xml = []
    for i in range(FINGERS):
        phi = 2 * np.pi * i / FINGERS
        c, s = np.cos(phi), np.sin(phi)
        r = ARM["palm_r"]
        finger_xml.append(f"""
            <body name="finger{i}" pos="0.03 {r * c:.4f} {r * s:.4f}">
              <joint name="j_f{i}" type="hinge" axis="0 {-s:.4f} {c:.4f}" range="-0.3 1.45" damping="40"/>
              <geom name="finger{i}" type="capsule" fromto="0 0 0 {ARM['finger']} 0 0" size="0.06" mass="{ARM_MASS['finger']}" rgba=".75 .75 .78 1"/>
              <geom name="pad{i}" type="box" pos="{ARM['finger'] - 0.15} {-0.07 * c:.4f} {-0.07 * s:.4f}" size="0.15 0.03 0.08" mass="1" friction="1.5 0.02 0.002" rgba=".9 .55 .2 1"/>
            </body>""")
    boulder_free = "" if attached else f"""
    <body name="boulder" pos="0 0 3">
      <freejoint name="boulder"/>
      <geom name="boulder" type="mesh" mesh="boulder" density="{params['boulder_density']:.4f}" friction="1.0 0.02 0.002" rgba=".45 .42 .38 1"/>
    </body>"""
    boulder_in_palm = "" if not attached else """
            <body name="payload" pos="0 0 0">
              <geom name="payload" type="mesh" mesh="boulder" density="{d:.4f}" contype="0" conaffinity="0"/>
            </body>""".format(d=params["boulder_density"])
    g = params["gravity"]
    return f"""
<mujoco model="egt_capture">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{params['dt']}" gravity="{g[0]:.6e} {g[1]:.6e} {g[2]:.6e}" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <asset>
    <hfield name="terrain" nrow="{hfield_rows}" ncol="{hfield_rows}" size="{HFIELD_HALF} {HFIELD_HALF} {max(zmax - zmin, 0.01):.4f} 1.0"/>
    <mesh name="boulder" vertex="{bv}"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {zmin:.4f}" friction="0.8 0.02 0.002" rgba=".4 .37 .33 1"/>
    <body name="bus" pos="0 0 8">
      <freejoint name="bus"/>
      <geom name="bus" type="box" size="{BUS_HALF[0]} {BUS_HALF[1]} {BUS_HALF[2]}" mass="{m_bus:.3f}" rgba=".85 .8 .55 1"/>
      <geom name="array_l" type="box" pos="4.3 0 0.4" size="3.0 1.0 0.03" mass="0" contype="0" conaffinity="0" rgba=".15 .2 .45 1"/>
      <geom name="array_r" type="box" pos="-4.3 0 0.4" size="3.0 1.0 0.03" mass="0" contype="0" conaffinity="0" rgba=".15 .2 .45 1"/>
      <body name="arm_base" pos="0 0 {-BUS_HALF[2]}">
        <joint name="j_yaw" type="hinge" axis="0 0 1" damping="200"/>
        <geom name="arm_base" type="cylinder" size="0.25 0.08" mass="{ARM_MASS['base']}" rgba=".6 .6 .62 1"/>
        <body name="upper" pos="0 0 -0.15">
          <joint name="j_shoulder" type="hinge" axis="0 1 0" damping="200"/>
          <geom name="upper" type="capsule" fromto="0 0 0 {ARM['upper']} 0 0" size="0.12" mass="{ARM_MASS['upper']}" rgba=".8 .8 .82 1"/>
          <body name="fore" pos="{ARM['upper']} 0 0">
            <joint name="j_elbow" type="hinge" axis="0 1 0" damping="200"/>
            <geom name="fore" type="capsule" fromto="0 0 0 {ARM['fore']} 0 0" size="0.1" mass="{ARM_MASS['fore']}" rgba=".8 .8 .82 1"/>
            <body name="wrist" pos="{ARM['fore']} 0 0">
              <joint name="j_wrist" type="hinge" axis="0 1 0" damping="100"/>
              <geom name="wrist" type="capsule" fromto="0 0 0 {ARM['wrist']} 0 0" size="0.09" mass="{ARM_MASS['wrist']}" rgba=".8 .8 .82 1"/>
              <body name="palm" pos="{ARM['wrist']} 0 0">
                <geom name="palm" type="cylinder" fromto="0 0 0 0.06 0 0" size="{ARM['palm_r']}" mass="{ARM_MASS['palm']}" friction="1.5 0.02 0.002" rgba=".9 .55 .2 1"/>
                <site name="palm" pos="0.06 0 0"/>{''.join(finger_xml)}{boulder_in_palm}
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>{boulder_free}
  </worldbody>
  <actuator>
    {''.join(f'<motor name="m_{j}" joint="{j}" ctrlrange="{-JOINT_TORQUE_MAX} {JOINT_TORQUE_MAX}"/>' for j in ARM_JOINTS)}
    {''.join(f'<position name="p_f{i}" joint="j_f{i}" kp="3000" ctrlrange="-0.3 1.45"/>' for i in range(FINGERS))}
    {''.join(f'<adhesion name="grip{i}" body="finger{i}" ctrlrange="0 1" gain="600"/>' for i in range(FINGERS))}
    <adhesion name="grip_palm" body="palm" ctrlrange="0 1" gain="600"/>
  </actuator>{'' if attached else '''
  <equality>
    <weld name="anchor" body1="palm" body2="boulder" active="false" solref="0.2 1"/>
  </equality>'''}
</mujoco>"""


@dataclass
class CaptureLog:
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    q_des: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    bus: list = field(default_factory=list)
    bus_force: list = field(default_factory=list)
    boulder: list = field(default_factory=list)
    palm: list = field(default_factory=list)
    stage: list = field(default_factory=list)


class Capture:
    """The capture sequence, stepped in time; poses are available for rendering at any moment."""

    def __init__(self, m_spacecraft, m_boulder, thrust_max, bus_pos0=None, bus_vel0=None, dt=0.01,
                 gravity=None, omega_local=None, terrain=True, coriolis=True, seed=7, site=None,
                 aspect=(1.3, 1.0, 0.75), density=BOULDER_DENSITY):
        """site: a core/proximity/boulders.py Site (origin, R, g_local, heightfield); None = the Nightingale
        default. seed / aspect / density: the boulder's shape and true density (its mass is m_boulder)."""
        if site is None:
            origin, R_site, g_local = site_frame()
        else:
            origin, R_site, g_local = site.origin, site.R, site.g_local
        self.R_site, self.site_origin = R_site, origin
        self.gravity = np.asarray(g_local if gravity is None else gravity, float)
        self.omega = (R_site.T @ np.array([0.0, 0.0, BENNU["omega"]])) if omega_local is None else np.asarray(omega_local)
        self.coriolis = coriolis
        self.thrust_max = float(thrust_max)
        self.m_boulder = float(m_boulder)
        m_arm = sum(ARM_MASS[k] for k in ("base", "upper", "fore", "wrist", "palm")) + FINGERS * (ARM_MASS["finger"] + 1.0)
        self.m_spacecraft = float(m_spacecraft)
        self.boulder_hull = boulder_mesh(m_boulder, seed=seed, aspect=aspect, density=density)
        g, Z = _hfield() if site is None else (site.grid, site.heights)
        zmin, zmax = float(Z.min()), float(Z.max())
        params = dict(dt=dt, gravity=self.gravity, zmin=zmin, zmax=zmax, m_bus=m_spacecraft - m_arm,
                      boulder_density=m_boulder / self.boulder_hull.volume)
        self.model = mujoco.MjModel.from_xml_string(_mjcf(params, len(g), self.boulder_hull.vertices))
        self.ctrl_model = mujoco.MjModel.from_xml_string(_mjcf(params, len(g), self.boulder_hull.vertices, attached=True))
        for M in (self.model, self.ctrl_model):
            M.hfield_data[:] = ((Z - zmin) / max(zmax - zmin, 0.01)).ravel()
        if not terrain:
            for M in (self.model, self.ctrl_model):
                gid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
                M.geom_contype[gid] = M.geom_conaffinity[gid] = 0
        self.data = mujoco.MjData(self.model)
        self.cdata = mujoco.MjData(self.ctrl_model)
        m, d = self.model, self.data
        self.jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS}
        self.arm_dof = np.array([m.jnt_dofadr[self.jid[n]] for n in ARM_JOINTS])
        self.arm_qpos = np.array([m.jnt_qposadr[self.jid[n]] for n in ARM_JOINTS])
        self.bus_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bus")
        self.boulder_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "boulder")
        self.palm_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        self.payload_id = mujoco.mj_name2id(self.ctrl_model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.palm_body_c = mujoco.mj_name2id(self.ctrl_model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.bodies = [b for b in range(1, m.nbody)]
        self.terrain = terrain

        # boulder resting on the terrain under the site origin
        self._place_boulder(g, Z if terrain else None)
        # stowed arm: folded under the bus
        self.q_stow = np.array([0.0, 0.25, 2.75, 0.0])
        d.qpos[self.arm_qpos] = self.q_stow
        for i in range(FINGERS):  # claws start open (no servo transient)
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"j_f{i}")]] = FINGER_OPEN
        self.bus_ref0 = self._default_bus_pos() if bus_pos0 is None else np.asarray(bus_pos0, float)
        d.qpos[0:3] = self.bus_ref0
        d.qpos[3:7] = [1, 0, 0, 0]
        if bus_vel0 is not None:
            d.qvel[0:3] = bus_vel0
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.stage = "deploy"
        self.stage_t0 = 0.0
        self.q_from = self.q_stow.copy()
        self.q_to = self._ik_pregrasp(0.8)
        self.T_stage = 90.0
        self.bus_ref = self.bus_ref0.copy()
        self.bus_from = self.bus_ref0.copy()
        self.fingers = FINGER_OPEN
        self.grip = 0.0
        self.attached = False
        self.result = None
        self.log = CaptureLog()
        self.Kp, self.Kd = 0.25, 1.0  # computed torque: omega_n = 0.5 rad/s, critically damped
        self._palm_to_boulder = None
        self.peak_tau = 0.0
        self.retries = 0
        self.peak_track = 0.0
        self.force_used = 0.0  # integral |F| dt (propellant proxy)

    # ------------------------------------------------------------------ setup helpers
    def _place_boulder(self, g, Z):
        """The boulder at rest under the site origin, lying on the local slope (its flattest side on the ground,
        tilted with the terrain's mean plane within 2 m): a level rock on a 20-degree slope stands on one edge and
        rolls away from the claws."""
        m, d = self.model, self.data
        adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "boulder")]
        v = self.boulder_hull.vertices
        quat = np.array([1.0, 0, 0, 0])
        if Z is None:
            z = -v[:, 2].min() + 0.01
        else:
            from scipy.interpolate import RegularGridInterpolator

            X, Y = np.meshgrid(g, g)
            near = np.hypot(X, Y) <= 2.0
            (a, b, _), *_ = np.linalg.lstsq(np.column_stack([X[near], Y[near], np.ones(near.sum())]), Z[near], rcond=None)
            n = np.array([-a, -b, 1.0]) / np.sqrt(a * a + b * b + 1.0)
            axis = np.cross([0, 0, 1.0], n)
            ang = float(np.arcsin(min(1.0, np.linalg.norm(axis))))
            if ang > 1e-6:
                quat = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis / np.linalg.norm(axis)])
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, quat)
            v = v @ Rb.reshape(3, 3).T
            f = RegularGridInterpolator((g, g), Z, bounds_error=False, fill_value=0.0)
            terrain_under = f(np.column_stack([v[:, 1], v[:, 0]]))
            z = float(np.max(terrain_under - v[:, 2])) + 0.005
        d.qpos[adr:adr + 3] = [0.0, 0.0, z]
        d.qpos[adr + 3:adr + 7] = quat
        self.boulder_qadr = adr
        self.boulder_top = z + v[:, 2].max()

    def _default_bus_pos(self):
        """Bus hovering so that the palm reaches the boulder top with the elbow comfortably bent."""
        return np.array([0.0, 0.0, self.boulder_top + 3.4 + BUS_HALF[2] + 0.15])

    def _palm(self, data):
        return data.site_xpos[self.palm_site].copy(), data.site_xmat[self.palm_site].reshape(3, 3)[:, 0].copy()

    def _ik_pregrasp(self, clearance):
        """Joint angles putting the palm `clearance` above the boulder top, palm facing down (numeric IK)."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        return self._ik(d, self.boulder_top_point() + np.array([0, 0, clearance]), self.data.qpos[0:3].copy())

    def boulder_top_point(self):
        """Highest point of the boulder near its vertical axis (where the palm lands)."""
        d = self.data
        Rb = d.xmat[self.boulder_id].reshape(3, 3)
        c = d.xpos[self.boulder_id]
        w = self.boulder_hull.vertices @ Rb.T
        near = np.hypot(w[:, 0], w[:, 1]) < 0.6
        return np.array([c[0], c[1], c[2] + (w[near, 2].max() if near.any() else w[:, 2].max())])

    def _ik(self, d, target, bus_pos, iters=200):
        m = self.model
        d.qpos[0:3] = bus_pos
        d.qpos[3:7] = [1, 0, 0, 0]
        q = d.qpos[self.arm_qpos].copy()
        if np.allclose(q, self.q_stow):
            q = np.array([0.0, 0.6, 1.6, 0.0])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.arm_qpos] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, ax = self._palm(d)
            err = np.concatenate([target - p, np.cross(ax, [0, 0, -1.0])])  # position + palm axis down
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.palm_site)
            J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        return q

    # ------------------------------------------------------------------ control
    def _arm_reference(self):
        s = (self.t - self.stage_t0) / self.T_stage
        f, fd, fdd = _min_jerk(s)
        dq = self.q_to - self.q_from
        if s >= 1.0:
            return self.q_to, np.zeros(4), np.zeros(4)
        return self.q_from + dq * f, dq * fd / self.T_stage, dq * fdd / self.T_stage**2

    def _bus_reference(self):
        if self.stage != "lift":
            return self.bus_ref, np.zeros(3)
        s = (self.t - self.stage_t0) / self.T_stage
        f, fd, _ = _min_jerk(s)
        delta = self.bus_lift_to - self.bus_from
        return self.bus_from + delta * f, delta * fd / self.T_stage if s < 1 else np.zeros(3)

    def _computed_torque(self, q_d, qd_d, qdd_d):
        d = self.data
        if self.attached:  # merged-mass model: palm + boulder as one rigid body
            cm, cd = self.ctrl_model, self.cdata
            nq_c = cm.nq
            cd.qpos[:] = d.qpos[:nq_c]
            cd.qvel[:] = d.qvel[:cm.nv]
            mujoco.mj_forward(cm, cd)
            M = np.zeros((cm.nv, cm.nv))
            mujoco.mj_fullM(cm, cd, M)
            bias = cd.qfrc_bias
        else:
            M = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, d, M)
            bias = d.qfrc_bias
        a = self.arm_dof
        q, qd = d.qpos[self.arm_qpos], d.qvel[a]
        v = qdd_d + self.Kd * (qd_d - qd) + self.Kp * (q_d - q)
        tau = M[np.ix_(a, a)] @ v + bias[a] - d.qfrc_passive[a]  # cancel the joint damping too
        return np.clip(tau, -JOINT_TORQUE_MAX, JOINT_TORQUE_MAX)

    def _bus_wrench(self, p_ref, v_ref):
        d = self.data
        m_eff = self.m_spacecraft + (self.m_boulder if self.attached else 0.0)
        wn = 2 * np.pi / 120.0
        e, ed = p_ref - d.qpos[0:3], v_ref - d.qvel[0:3]
        F = m_eff * (wn**2 * e + 2 * wn * ed) - m_eff * self.gravity
        n = np.linalg.norm(F)
        if n > self.thrust_max:
            F *= self.thrust_max / n
        # attitude: quaternion PD toward level (identity), torque in world frame
        q = d.qpos[3:7]
        qv = q[1:] * np.sign(q[0] if q[0] != 0 else 1.0)
        R = d.xmat[self.bus_id].reshape(3, 3)
        I_world = R @ np.diag(self.model.body_inertia[self.bus_id]) @ R.T
        wa = 2 * np.pi / 60.0
        w = d.qvel[3:6]  # world-frame angular velocity for a free joint? (MuJoCo: local frame)
        w_world = R @ w
        tau = I_world @ (-2 * wa**2 * (R @ qv) - 2 * wa * w_world)
        nt = np.linalg.norm(tau)
        if nt > TORQUE_MAX_BUS:
            tau *= TORQUE_MAX_BUS / nt
        return F, tau

    def _coriolis(self):
        """-2 m w x v at each body's center of mass (rotating site frame)."""
        m, d = self.model, self.data
        d.xfrc_applied[:] = 0.0
        if not self.coriolis or not np.any(self.omega):
            return
        d.xfrc_applied[:, :3] = -2.0 * m.body_mass[:, None] * np.cross(self.omega, self.com_velocities())

    def com_velocities(self):
        """Linear velocity of every body's center of mass, from MuJoCo's com-based velocities cvel
        (spatial velocity referred to the subtree center of mass of each body's root)."""
        m, d = self.model, self.data
        ref = d.subtree_com[m.body_rootid]
        return d.cvel[:, 3:] + np.cross(d.cvel[:, :3], d.xipos - ref)

    # ------------------------------------------------------------------ sequence
    def _next_stage(self):
        d = self.data
        self.stage_t0 = self.t
        if self.stage == "deploy":
            self.stage = "reach"
            self.reach_depth = 0.05
            self.q_from, self.q_to = self.q_to, self._ik_pregrasp(-self.reach_depth)
            self.T_stage = 60.0
        elif self.stage == "reach":
            self.stage = "grip"
            self.q_from = self.q_to = d.qpos[self.arm_qpos].copy()
            self.T_stage = 40.0
            self.grip = 1.0
        elif self.stage == "grip":
            if not self._palm_contact() and self.reach_depth < 0.4:  # missed: reach a little deeper
                self.stage = "reach"
                self.reach_depth += 0.1
                self.retries += 1
                self.grip = 0.0
                self.q_from, self.q_to = d.qpos[self.arm_qpos].copy(), self._ik_pregrasp(-self.reach_depth)
                self.T_stage = 30.0
                return
            self.stage = "lift"
            self._attach()
            m_tot = self.m_spacecraft + self.m_boulder
            a_avail = self.thrust_max / m_tot - np.linalg.norm(self.gravity)
            self.lift_ok = a_avail > 0.0
            self.bus_from = d.qpos[0:3].copy()
            self.bus_lift_to = self.bus_from + np.array([0.0, 0.0, 3.0])
            # min-jerk peak acceleration 5.77 h / T^2 limited to half the available acceleration
            self.T_stage = float(np.sqrt(5.77 * 3.0 / (0.5 * a_avail))) if self.lift_ok else 600.0
            self.T_stage = max(self.T_stage, 120.0)
            self.q_from = self.q_to = d.qpos[self.arm_qpos].copy()
        elif self.stage == "lift":
            self.stage = "stow"
            self.bus_ref = self.bus_lift_to.copy()
            self.q_from = d.qpos[self.arm_qpos].copy()
            bus = d.qpos[0:3].copy()
            target = bus - np.array([0, 0, BUS_HALF[2] + 0.15 + 1.6])
            dd = mujoco.MjData(self.model)
            dd.qpos[:] = d.qpos
            self.q_to = self._ik(dd, target, bus)
            self.T_stage = 200.0
        elif self.stage == "stow":
            self.stage = "done"  # hold until the bus has settled (or 10 minutes)
            self.T_stage = 30.0
            self.done_since = self.t
            self.q_from = self.q_to = self.q_to.copy()
        elif self.stage == "done":
            if np.linalg.norm(d.qvel[0:3]) > 0.005 and self.t - self.done_since < 600.0:
                self.T_stage += 10.0
                self.stage_t0 = self.t - self.T_stage + 10.0
                return
            self._finish()

    def _palm_contact(self):
        d = self.data
        bg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "boulder")
        hand = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["palm"] + [f"pad{i}" for i in range(FINGERS)]}
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 == bg and g2 in hand) or (g2 == bg and g1 in hand):
                return True
        return False

    def _attach(self):
        """Spines anchored: weld palm-boulder at the current relative pose; the controller adopts the
        merged-mass model (boulder fixed in the palm)."""
        m, d = self.model, self.data
        self.attached = True
        palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        Rp = d.xmat[palm_id].reshape(3, 3)
        Rb = d.xmat[self.boulder_id].reshape(3, 3)
        p_rel = Rp.T @ (d.xpos[self.boulder_id] - d.xpos[palm_id])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Rp.T @ Rb).ravel())
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "anchor")
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = p_rel
        m.eq_data[eid, 6:10] = q
        m.eq_data[eid, 10] = 1.0
        d.eq_active[eid] = 1
        self.ctrl_model.body_pos[self.payload_id] = p_rel
        self.ctrl_model.body_quat[self.payload_id] = q
        self._palm_to_boulder = np.linalg.norm(d.xpos[self.boulder_id] - d.site_xpos[self.palm_site])

    def step(self):
        m, d = self.model, self.data
        if self.t - self.stage_t0 >= self.T_stage and self.stage != "finished":
            self._next_stage()
        q_d, qd_d, qdd_d = self._arm_reference()
        tau = self._computed_torque(q_d, qd_d, qdd_d)
        p_ref, v_ref = self._bus_reference()
        F, T = self._bus_wrench(p_ref, v_ref)
        self._coriolis()
        d.xfrc_applied[self.bus_id, :3] += F
        d.xfrc_applied[self.bus_id, 3:] += T
        d.ctrl[:4] = tau
        if self.stage in ("grip", "lift", "stow", "done", "finished"):
            s = min(1.0, (self.t - self.stage_t0) / 30.0) if self.stage == "grip" else 1.0
            self.fingers = FINGER_OPEN + (FINGER_CLOSED - FINGER_OPEN) * s
        d.ctrl[4:4 + FINGERS] = self.fingers
        d.ctrl[4 + FINGERS:] = self.grip
        mujoco.mj_step(m, d)
        self.t += m.opt.timestep
        self.peak_tau = max(self.peak_tau, float(np.abs(tau).max()))
        if self.stage in ("deploy", "stow"):  # free-space motion (in reach the palm presses by design)
            self.peak_track = max(self.peak_track, float(np.abs(q_d - d.qpos[self.arm_qpos]).max()))
        self.force_used += np.linalg.norm(F) * m.opt.timestep
        self._F = F
        return q_d, tau, F

    def advance(self, duration, log_every=1.0):
        t_end = self.t + duration
        next_log = self.log.t[-1] + log_every if self.log.t else self.t
        while self.t < t_end - 1e-9 and self.stage != "finished":
            q_d, tau, F = self.step()
            if self.t >= next_log:
                d = self.data
                L = self.log
                L.t.append(self.t)
                L.q.append(d.qpos[self.arm_qpos].copy())
                L.q_des.append(q_d.copy())
                L.tau.append(tau.copy())
                L.bus.append(d.qpos[0:3].copy())
                L.bus_force.append(F.copy())
                L.boulder.append(d.xpos[self.boulder_id].copy())
                L.palm.append(d.site_xpos[self.palm_site].copy())
                L.stage.append(self.stage)
                next_log = self.t + log_every
        return self.stage == "finished"

    def run(self, max_time=3600.0):
        self.advance(max_time)
        return self.result

    # ------------------------------------------------------------------ results
    def _finish(self):
        d = self.data
        self.stage = "finished"
        held = np.linalg.norm(d.xpos[self.boulder_id] - d.site_xpos[self.palm_site])
        lifted = d.xpos[self.boulder_id][2] - self.log.boulder[0][2] if self.log.boulder else 0.0
        self.result = {  # (a boulder "lifted" far beyond the 3 m lift and the stow was thrown by a contact blow-up)
            "success": bool(abs(held - (self._palm_to_boulder or held)) < 0.15 and 2.0 < lifted < 10.0
                            and np.all(np.isfinite(d.qpos))),
            "lifted_m": float(lifted), "grip_slip_m": float(abs(held - (self._palm_to_boulder or held))),
            "peak_joint_torque": self.peak_tau, "peak_tracking_rad": self.peak_track,
            "duration_s": self.t, "thrust_impulse_Ns": self.force_used,
            "mass": self.merged_mass_properties(),
        }

    def merged_mass_properties(self):
        """Spacecraft + arm + boulder as one rigid body (parallel-axis theorem), local frame."""
        m, d = self.model, self.data
        parts = []
        for b in self.bodies:
            R = d.ximat[b].reshape(3, 3)
            parts.append((m.body_mass[b], d.xipos[b].copy(), R @ np.diag(m.body_inertia[b]) @ R.T))
        return merge_mass(parts)

    def poses(self):
        """[(geom name, type, size, rgba, position, rotation)] for every geom except the terrain (site frame)."""
        m, d = self.model, self.data
        out = []
        for gi in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if name == "terrain":
                continue
            out.append((name, int(m.geom_type[gi]), m.geom_size[gi].copy(), m.geom_rgba[gi].copy(),
                        d.geom_xpos[gi].copy(), d.geom_xmat[gi].reshape(3, 3).copy()))
        return out

    def craft_shape(self, with_boulder):
        """The spacecraft's geoms in the bus frame (for rendering it away from the capture site)."""
        d = self.data
        pb, Rb = d.xpos[self.bus_id], d.xmat[self.bus_id].reshape(3, 3)
        return [(n, ty, sz, rgba, Rb.T @ (p - pb), Rb.T @ R) for n, ty, sz, rgba, p, R in self.poses()
                if with_boulder or n != "boulder"]

    def boulder_mesh_data(self):
        """Boulder vertices/faces as MuJoCo stores them (in the geom frame)."""
        m = self.model
        mid = m.geom_dataid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "boulder")]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return m.mesh_vert[v0:v0 + nv].copy(), m.mesh_face[f0:f0 + nf].copy()

    def momentum(self):
        """Total linear momentum of all bodies [kg m/s]."""
        m, d = self.model, self.data
        jacp = np.zeros((3, m.nv))
        p = np.zeros(3)
        for b in self.bodies:
            mujoco.mj_jacBodyCom(m, d, jacp, None, b)
            p += m.body_mass[b] * (jacp @ d.qvel)
        return p

    def bus_state(self):
        d = self.data
        return d.qpos[0:3].copy(), d.qvel[0:3].copy()


def merge_mass(parts):
    """parts: [(m_i, c_i, J_i about c_i)] -> (m, c, J about c), J = sum J_i + m_i (|d_i|^2 I - d_i d_i^T)."""
    M = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / M
    J = np.zeros((3, 3))
    for mi, ci, Ji in parts:
        dd = ci - c
        J += Ji + mi * (dd @ dd * np.eye(3) - np.outer(dd, dd))
    return {"mass": float(M), "com": c, "inertia": J}
