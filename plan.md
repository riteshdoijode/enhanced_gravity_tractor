# Bennu Planetary-Defense Mission Simulator (Interactive, Mac-only)

**DART-style kinetic impactor (primary) + Enhanced Gravity Tractor (precision trim)**

## Context

Course project for Autonomous Space Robotics, Fall 2026. Everything (Steps 1–8) is in this project, and everything runs on your MacBook (M3 Pro, 18 GB RAM).

The professor's requirements:

- send actions and get sensor values back;
- validate the simulator with math from class;
- load a 3D model and make an animation movie.

The movie comes from an in-app record/replay feature and can render slowly. It is not the main goal.

The main goal is an interactive "mission control" simulator. It shows the mission in 3D, and you change parameters and see the consequences live:

- **Tractor thrust:** raise it and the spacecraft reaches Bennu sooner. Fuel is unlimited but consumption is tracked.
- **Solar-storm launch delay, or a mid-cruise safe mode:** the display shows the new arrival date, the lost lead time, and whether the mission still succeeds.
- **Bigger or faster DART-style impactor (assembled in Earth orbit):** how far Bennu's trajectory moves, and how much thrust and Δv it takes to reach Bennu.

Every spacecraft starts in Earth orbit.

After you approve, I implement Step 1 only and stop for your review.

**Target:** 101955 Bennu (JPL Small-Body Database, fetched).

| Quantity | Value |
| --- | --- |
| GM | 4.8904 m³/s² (M ≈ 7.33×10¹⁰ kg) |
| Density | 1.194 g/cm³ |
| Effective diameter | 484.44 m |
| Extent | 505 × 492 × 457 m |
| Rotation period | 4.296061 h |
| Spin pole | RA/Dec 85.45°, −60.37° |
| Orbit | a = 1.13 au, e = 0.204, i = 6.03° |
| Earth approaches | 2054-09-30 (0.0393 au), 2060-09-23 (0.00501 au), 2080-09-22 (0.0156 au), 2135-09-25 (0.00136 au), which is the keyhole gateway to a 1-in-2700 chance of impact in 2182 |

## 1. What the simulator looks like

```text
┌──────────────────────── Bennu Defense — Mission Control ─────────────────────────┐
│ CONTROLS               │              3D VIEW (PyVista/VTK)        │ TELEMETRY    │
│ Scenario [Real 2135 ▾] │  camera: [Solar sys][Earth][Bennu][Chase] │ 2031-04-02   │
│ Launch delay   ▮▮▯ 40d │                                           │ Phase CRUISE │
│ Tractor thrust ▮▮▮ 2 N │    Sun ·  Earth orbit  ·  Bennu orbit     │ ETA 2032-11  │
│ Tractor Isp    3000 s  │    planned (dashed) vs flown (solid)      │ Fuel 412 kg  │
│ Boulder mass   20 t    │    deflected Bennu (exaggerated ×1e3)     │ Δv 4.1 km/s  │
│ Impactor mass  580 kg  │                                           │ Bennu Δv     │
│ Impact speed   6.1km/s │                                           │  0.17 mm/s   │
│ β (ejecta)     3.6     │                                           │ Δb 2135 ...  │
│ Time warp      5 d/s   │                                           │ ✓ MISSION OK │
│ [Plan] [Launch Tractor] [Launch Impactor] [Solar storm!] [● Rec]   │ event log    │
├────────────────────────┴───────────────── PLOTS (pyqtgraph) ──────────────────────┤
│ fuel used vs t │ range to Bennu │ along-track shift Δs(t) │ B-plane: Δb + β cloud │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Two engines share one physics library.

1. **Instant planner (< 1 s per slider change).** Runs in a background thread, with sliders debounced.
   - Fast models: vectorized Lambert grids, an analytic Earth-escape spiral, coarse closed-loop low-thrust guidance, and a precomputed state-transition matrix (STM) of Bennu's orbit.
   - The STM turns any Δv into a B-plane shift instantly. That is linearization from class, and it is validated against full nonlinear runs.
   - A refined answer replaces the estimate a few seconds later.
2. **Live run (time-warped high fidelity).** Press Launch and the full simulation steps forward: N-body dynamics, 6-DOF attitude, sensors, Kalman filter, controllers.
   - You can change the throttle mid-flight or trigger "Solar storm!", which forces safe mode with thrust off for N days.
   - Guidance re-plans from the current state and the ETA and success flags update live.

**Assumptions** (shown in an "Assumptions" panel):

- **Unlimited fuel.** Default is constant-mass mode: propellant use ṁ is integrated and displayed but not subtracted from the mass. A toggle switches to realistic mass depletion.
- **Assembled in Earth orbit:** impactor mass, tractor mass and stage thrust are free parameters.
- **Unknowns are labeled sliders:** β, boulder density, and sensor noise.

**Scenario presets:**

- **Real Bennu, 2135 keyhole avoidance.** The B-plane shift needed, Δb_req, is a slider. The emphasis is precision, using the gravity-tractor trim.
- **Hypothetical-impact exercise** (clearly labeled fictional, in the style of NASA's Planetary Defense Conference exercises). Pretend Bennu's 2060-09-23 approach is an Earth impact. Success then requires Δb ≥ Earth's capture radius $b_\oplus$ (§E12). Here the launch delay, impactor mass and impact speed visibly decide success or failure.

## 2. Step-by-step procedure (each step ends with a checkpoint for you; all on the Mac)

| Step | Build | Checkpoint you review |
| --- | --- | --- |
| 1 | uv project (Python 3.12), dependencies, data fetch (Horizons, SBDB, Bennu shape models, spacecraft GLB, Earth texture), app skeleton: Qt window with a 3D Solar-System view from JPL data and a date slider | The app opens, shows real Earth + Bennu orbits, scrubs through dates, and loads the Bennu and spacecraft 3D models |
| 2 | Core physics library + validation: N-body, J2/drag/SRP, quaternion attitude, polyhedron gravity, rotating frame, Bennu nominal orbit + STM precompute | Validation plots/tests; Bennu vs Horizons agreement |
| 3 | Instant planner: Earth departure (impulsive or low-thrust spiral), heliocentric Lambert grids, impactor sizing (Δv, thrust, propellant vs mass and speed), deflection → B-plane via STM, β cloud, success logic; wired to sliders | Move sliders and watch arrival, fuel, Δb and ✓/✗ update live |
| 4 | Live run: closed-loop low-thrust guidance (ZEM/ZEV) with throttle and safe-mode events, time warp, telemetry, sensors + KF during cruise | Launch, change thrust mid-flight, trigger a storm, and watch the ETA and success react |
| 5 | Bennu proximity: approach, hover LQR vs MPC, LIDAR/camera + EKF, rotating-frame descent, boulder capture (MuJoCo arm), mass merge | Hover and capture shown in the Bennu view; controller and estimator plots |
| 6 | Impactor terminal phase: vision-in-the-loop proportional navigation, impact momentum transfer, spin change, ejecta, DART validation | Impact in the Bennu view; miss distance; Δv vs β; the −33 min DART check |
| 7 | Gravity-tractor tow + trim guidance (closed outer loop on the B-plane), long-term nominal vs deflected propagation, final success evaluation | Δs(t) vs Clohessy–Wiltshire (CW) prediction; B-plane with and without trim |
| 8 | Visual polish (star skybox, Sun lighting, PBR models, labels, scale indicators), Record/Replay → MP4, report figures, README | mission.mp4 + report figures |

**Deadline risk.** If Project 1 is still due 10/19, that's four days for eight steps, which is tight.

- **Core in every step** (the controls you described): Steps 1–4, the hover LQR, the impact physics, and tow + B-plane.
- **Showcase items:** MPC, EKF, MuJoCo arm, vision-in-the-loop. These go in right after the core of their step, so a working simulator always exists.

## 3. Software, libraries, data, requirements (Mac only)

**Base:** uv with Python 3.12 pinned (your default 3.14 may lack VTK/MuJoCo/PySide6 wheels). Already installed: uv, ffmpeg.

| Library | Purpose |
| --- | --- |
| numpy, scipy | physics, solve_ivp (DOP853), solve_discrete_are, expm, Hermite interpolation, root finding |
| PySide6 | desktop application: windows, sliders, buttons, threads (QThreadPool) |
| pyvista + pyvistaqt | 3D view embedded in Qt (QtInteractor); GLB/OBJ models, PBR, skybox, MP4 recording |
| pyqtgraph | fast live 2D plots in the app |
| matplotlib | static report figures |
| imageio + imageio-ffmpeg | MP4 export |
| h5py | run logs (lecture1 convention) |
| trimesh | shape-model mass properties (volume, center of mass, inertia) |
| requests | JPL Horizons and SBDB APIs |
| astropy | UTC↔TDB conversion, calendar dates |
| mujoco | arm + boulder contact grasp (Step 5) |
| cvxpy (Clarabel) | hover MPC (Step 5) |
| joblib | parallel Monte Carlo / porkchop refinement |
| pytest | validation suite |
| numba (opt.) | speeding up polyhedron gravity |

**Data and assets:**

- JPL Horizons ephemerides: Sun, Venus, Earth, Moon, Mars, Jupiter, Saturn, Bennu 101955;. The window is 2026–2140, cached once to `data/*.npz`.
- SBDB: Bennu physical parameters and its non-gravitational A2 term.
- Bennu shape model: OSIRIS-REx OBJ, decimated to about 3k facets for gravity and about 200k for display.
- Bennu render model: NASA glTF.
- Tractor model: NASA 3D Resources Dawn.glb (ion-propelled). The OSIRIS-REx model is an alternative if its format works.
- Impactor model: procedural DART-like model (box bus + two roll-out solar arrays), since no official DART GLB exists.
- Textures: Earth (NASA Blue Marble) and a star-field cubemap.

**Simulation requirements:**

- **FR1:** All spacecraft start in configurable Earth parking orbits.
- **FR2:** Live parameters:
  - tractor thrust, Isp and mass; boulder mass; hover distance;
  - impactor mass, impact speed (or departure Δv), stage thrust and Isp; β;
  - launch delays; Δb_req; scenario; time warp.
- **FR3:** Commands are app buttons and a scriptable API: `env.command("LAUNCH_TRACTOR")`, `env.step(u) -> y` (Project 1's action/sensor requirement).
- **FR4:** Events: launch hold, and safe mode with thrust off for N days.
- **FR5:** Live telemetry: date, phase, ETA, Δv, propellant used, thrust, range, sensor readings, estimate error, Bennu Δv, Δb, and success flags with reasons.
- **FR6:** Instant planner under 1 s; live run at ≥ 20 fps rendering.
- **FR7:** Record/replay to MP4; HDF5 logs; pytest validation suite.
- **NFR:** SI units internally; seeded noise; config file; lecture3-style System/Controller/Estimator interfaces.

## 4. Equations

### E0. Frames, time, state

- **Time:** TDB seconds past J2000; the UI shows UTC.
- **Inertial frame:** ICRF. Bennu is propagated about the Solar System Barycenter (SSB).
- Spacecraft position is relative to a central body c:
  - c = Earth inside its sphere of influence ($r_{SOI}\approx9.25\times10^5$ km);
  - c = Sun during cruise;
  - c = Bennu when within 1000 km.
- **Bennu body frame:** $R_B(t)=R_{pole}R_z(W_0+\omega_Bt)$ with $\omega_B=2\pi/4.296061$ h.
- **States:**
  - spacecraft $x=[\mathbf r,\mathbf v,q,\boldsymbol\omega,m]$;
  - control $u=[\mathbf f_B,\boldsymbol\tau]$ (lecture L1 form);
  - Bennu $x_A=[\mathbf r_A,\mathbf v_A]$.

### E1. Translation (lecture L1 eqs 1–2, central-body form)

$$
\ddot{\mathbf r}=-\frac{\mu_c\mathbf r}{|\mathbf r|^3}+\sum_{k\neq c}\mu_k\!\left[\frac{\mathbf s_k-\mathbf r}{|\mathbf s_k-\mathbf r|^3}-\frac{\mathbf s_k}{|\mathbf s_k|^3}\right]+\mathbf a_{J2}+\mathbf a_{drag}+\mathbf a_{SRP}+\frac{R(q)\mathbf f_B}{m}\,\eta_{throttle}\,\chi_{safe}
$$

- $\mathbf s_k=\mathbf r_k-\mathbf r_c$.
- $\chi_{safe}=0$ during a storm safe mode.
- When c = Bennu, use $\mu_c=G(M_A+m)$.

J2:

$$
\mathbf a_{J2}=-\frac{3J_2\mu_ER_E^2}{2r^5}\begin{bmatrix}x(1-5z^2/r^2)\\y(1-5z^2/r^2)\\z(3-5z^2/r^2)\end{bmatrix}
$$

Drag:

$$
\mathbf a_{drag}=-\tfrac12\tfrac{C_DA}{m}\rho_0e^{-(h-h_0)/H}|\mathbf v_{rel}|\mathbf v_{rel}
$$

Solar radiation pressure:

$$
\mathbf a_{SRP}=\nu P_\odot C_R\tfrac{A}{m}\left(\tfrac{1\text{ AU}}{|\mathbf r_{sc}-\mathbf r_\odot|}\right)^2\hat{\mathbf s},\qquad P_\odot=4.56\times10^{-6}\text{ N/m}^2
$$

### E2. Bennu motion, coupling, and the precomputed sensitivity (STM)

$$
\ddot{\mathbf r}_A=\sum_k\frac{\mu_k(\mathbf r_k-\mathbf r_A)}{|\mathbf r_k-\mathbf r_A|^3}+\frac{A_2}{r_{A\odot}^2}\hat{\mathbf t}+\frac{Gm_T(\mathbf r_{sc}-\mathbf r_A)}{|\mathbf r_{sc}-\mathbf r_A|^3}+\Delta\mathbf v_{KI}\,\delta(t-t_{imp})
$$

STM, precomputed once at startup and cached. It is integrated with the nominal orbit from 2026 to 2140:

$$
\dot\Phi=\begin{bmatrix}0&I\\G(\mathbf r_A)&0\end{bmatrix}\Phi,\quad G=\sum_k\frac{\mu_k}{|\mathbf r_k-\mathbf r_A|^3}\left(3\hat{\mathbf d}_k\hat{\mathbf d}_k^\top-I\right),\quad \Phi(t_{ca},t)=\Phi(t_{ca},t_0)\Phi(t,t_0)^{-1}
$$

Using it:

- Impulse: $\Delta\mathbf r_A(t_{ca})=\Phi_{rv}(t_{ca},t)\Delta\mathbf v$.
- Continuous tow: $\Delta\mathbf r_A(t_{ca})=\int\Phi_{rv}(t_{ca},t)\mathbf a_{GT}(t)dt$.
- Both are instant for any slider value. Truth check: a full nonlinear differential propagation (nominal vs deflected, same model) runs in the background.

### E3. Bennu gravity (Werner & Scheeres 1997 polyhedron; point mass far away)

$$
\nabla U=-G\sigma\sum_eE_e\mathbf r_eL_e+G\sigma\sum_fF_f\mathbf r_f\omega_f,\qquad L_e=\ln\frac{r_1+r_2+e_{12}}{r_1+r_2-e_{12}}
$$

$$
\omega_f=2\,\mathrm{atan2}\!\left(\mathbf r_1\cdot(\mathbf r_2\times\mathbf r_3),\ r_1r_2r_3+r_1\mathbf r_2\cdot\mathbf r_3+r_2\mathbf r_3\cdot\mathbf r_1+r_3\mathbf r_1\cdot\mathbf r_2\right)
$$

- $F_f=\hat{\mathbf n}_f\hat{\mathbf n}_f^\top$.
- $E_e=\hat{\mathbf n}_A\hat{\mathbf n}^{A\top}_e+\hat{\mathbf n}_B\hat{\mathbf n}^{B\top}_e$.
- Checks: $\sum_f\omega_f$ = 4π inside and 0 outside, and the far field equals $\mu/r^2$.

### E4. Attitude (lecture L1 eqs 3–4; quaternions avoid the 3-2-1 Euler singularity, det B = cos θ)

$$
\dot q=\tfrac12\begin{bmatrix}-\mathbf q_v^\top\\q_0I+[\mathbf q_v\times]\end{bmatrix}\boldsymbol\omega,\quad J\dot{\boldsymbol\omega}=-\boldsymbol\omega\times J\boldsymbol\omega+\boldsymbol\tau_c+\boldsymbol\tau_{gg},\quad R(q)=(q_0^2-\mathbf q_v^\top\mathbf q_v)I+2\mathbf q_v\mathbf q_v^\top+2q_0[\mathbf q_v\times]
$$

**Live-run speed switch.** During time-warped cruise, attitude is assumed to track its command; full attitude dynamics run near Bennu and during burns.

### E5. Propulsion, fuel tracking, unlimited-fuel mode

$$
\dot m_p=\frac{T\eta_{throttle}}{I_{sp}g_0}\ (\text{always tracked}),\qquad m=\begin{cases}m_{dry}+m_{payload}&\text{unlimited-fuel mode (default)}\\m_0-m_p&\text{realistic mode}\end{cases}
$$

- Rocket equation: $\Delta v=I_{sp}g_0\ln(m_0/m_f)$.
- Propellant for an impulsive Δv: $m_p=m_f\left(e^{\Delta v/I_{sp}g_0}-1\right)$.

### E6. Earth departure (works for any thrust slider value)

**High thrust** (burn shorter than about 10% of the parking-orbit period): a finite prograde burn at the escape point, integrated in Cartesian form. The hyperbola and the burn point are

$$
\Delta v_{dep}=\sqrt{v_\infty^2+\tfrac{2\mu_\oplus}{r_p}}-\sqrt{\tfrac{\mu_\oplus}{r_p}},\qquad e=1+\tfrac{r_pv_\infty^2}{\mu_\oplus},\ \nu_\infty=\arccos(-1/e),\ \hat{\mathbf p}=\cos\nu_\infty\hat{\mathbf v}_\infty-\sin\nu_\infty(\hat{\mathbf h}\times\hat{\mathbf v}_\infty)
$$

- The burn starts when $\hat{\mathbf r}\approx\hat{\mathbf p}$.
- Gravity loss: shown live as the difference between the integrated Δv and the impulsive Δv.

**Low thrust** (spiral, orbit-averaged tangential thrust; Gauss variational equation):

$$
\dot a=\frac{2a^{3/2}}{\sqrt{\mu_\oplus}}\frac{T}{m}\ \Rightarrow\ \sqrt{\mu_\oplus/a(t)}=v_{c0}-\frac{T}{m}t,\qquad t_{esc}\approx\frac{v_{c0}}{T/m}\ (\Delta v_{esc}\approx v_{c0})
$$

- Switch to Cartesian integration once the orbit leaves the near-circular regime, and continue until the sphere-of-influence exit.
- This is where "more thrust = faster" shows clearly. For 10 t from a 400 km orbit:

| Thrust | Escape time |
| --- | --- |
| 1 N | ≈ 2.4 yr |
| 10 N | ≈ 89 d |
| 100 N | ≈ 9 d |
| kN class | effectively impulsive |

### E7. Heliocentric guidance

**Lambert's problem** (universal variables, Curtis ch. 5). Vectorized over (departure date, flight time) grids for porkchops, and used for the impulsive impactor.

$$
A=\sin\Delta\theta\sqrt{\tfrac{r_1r_2}{1-\cos\Delta\theta}},\ y(z)=r_1+r_2+A\tfrac{zS-1}{\sqrt C},\ \left(\tfrac{y}{C}\right)^{3/2}S+A\sqrt y=\sqrt\mu\Delta t
$$

$$
\mathbf v_1=\tfrac{\mathbf r_2-f\mathbf r_1}{g},\ \mathbf v_2=\tfrac{\dot g\mathbf r_2-\mathbf r_1}{g},\quad f=1-\tfrac{y}{r_1},\ g=A\sqrt{y/\mu},\ \dot g=1-\tfrac{y}{r_2}
$$

**Continuous-thrust rendezvous (tractor):** generalized ZEM/ZEV feedback guidance (Guo, Hawkins & Wie, JGCD 2013). This closed-loop law re-plans every step, so throttle changes and safe modes are handled automatically. With $t_{go}=t_f-t$:

$$
\mathbf{ZEM}=\mathbf r_T(t_f)-\hat{\mathbf r}(t_f),\quad \mathbf{ZEV}=\mathbf v_T(t_f)-\hat{\mathbf v}(t_f),\quad \mathbf a_{cmd}=\frac{6}{t_{go}^2}\mathbf{ZEM}-\frac{2}{t_{go}}\mathbf{ZEV},\quad |\mathbf a|\le\frac{T\eta}{m}
$$

- $\hat{\mathbf r},\hat{\mathbf v}$ are the spacecraft state coasted (Kepler f-and-g propagation) to $t_f$.
- $\mathbf r_T,\mathbf v_T$ are Bennu's state plus a 50 km standoff.
- Minimum time (the ETA you see): $t_f^*=\min\{t_f:\max_t|\mathbf a_{cmd}|\le T\eta/m\}$, found by bisection. Higher thrust gives an earlier $t_f^*$.
- After any throttle change or safe mode, $t_f^*$ is re-solved from the current state. If no $t_f$ before the deadline is feasible, the display shows ✗ "cannot reach Bennu in time".
- Low-thrust impactor (optional): intercept-only law $\mathbf a_{cmd}=3\,\mathbf{ZEM}/t_{go}^2$. There is no velocity matching, because we want to hit.

**Impactor sizing** (what bigger/faster DART costs):

- Search the Lambert grid for the minimum $\Delta v_{dep}$ such that the impact speed $|\mathbf U|=|\mathbf v_2-\mathbf v_A(t_2)|\ge U_{target}$.
- Prefer solutions with large along-track momentum $\mathbf U\cdot\hat{\mathbf v}_A$.
- Required stage thrust for a burn time $t_b$:
  - constant-mass mode: $T_{req}=m\,\Delta v_{dep}/t_b$;
  - realistic mode: $T_{req}=I_{sp}g_0\,m_f\left(e^{\Delta v/I_{sp}g_0}-1\right)/t_b$.
- A gravity-loss guideline gives the minimum sensible thrust: $T_{min}\approx m\Delta v_{dep}/(0.1P_{park})$.
- Full-model correction (shooting) and mid-course corrections: $\delta\mathbf v=-\Phi_{rv}^{-1}[\mathbf r(t_2)-\mathbf r_2]$.

**Impactor terminal guidance:** proportional navigation (Zarchan). Here $\mathbf r=\mathbf r_A-\mathbf r_{KI}$:

$$
\boldsymbol\Omega=\frac{\mathbf r\times\dot{\mathbf r}}{\mathbf r\cdot\mathbf r},\ V_c=-\frac{\mathbf r\cdot\dot{\mathbf r}}{|\mathbf r|},\ \mathbf a_{cmd}=NV_c(\boldsymbol\Omega\times\hat{\mathbf r}),\ N\in[3,5]
$$

### E8. Hover and gravity tractor (Lu & Love 2005, verified)

Hover linearization (same structure as lecture L3):

$$
\ddot{\boldsymbol\rho}=-\nabla U+\mathbf a_{SRP}+\mathbf u;\quad \boldsymbol\rho^*=d\hat{\mathbf e},\ \mathbf u^*=\tfrac{\mu}{d^2}\hat{\mathbf e}-\mathbf a_{SRP};\quad \delta\ddot{\boldsymbol\rho}=\Gamma\delta\boldsymbol\rho+\delta\mathbf u,\ \Gamma=\tfrac{\mu}{d^3}(3\hat{\mathbf e}\hat{\mathbf e}^\top-I)
$$

- Eigenvalues: $\pm\sqrt{2\mu/d^3}$ (unstable; e-folding ≈ 57 min at d = 485 m) and $\pm i\sqrt{\mu/d^3}$.
- Required thrust and resulting pull (φ = plume half-width, 20°):

$$
T\cos[\sin^{-1}(R/d)+\phi]=\frac{GM_Am_T}{d^2},\qquad a_A=\frac{Gm_T}{d^2},\qquad m_T=m_{sc}+m_{boulder}
$$

- Live check: if the required T > the thrust slider, the display shows "cannot hover at d" and suggests the minimum d.
- Defaults: 10 t + 20 t boulder at d = 485 m. Required T ≈ 1.0 N; 0.27 mm/s per year of tow (0.09 without the boulder); propellant ≈ 1.1 t/yr at 3000 s.

### E9. Boulder capture (Enhanced Gravity Tractor, Mazanek et al. 2015)

Rotating-frame descent:

$$
\ddot{\boldsymbol\rho}_B=-2\boldsymbol\omega_B\times\dot{\boldsymbol\rho}_B-\boldsymbol\omega_B\times(\boldsymbol\omega_B\times\boldsymbol\rho_B)+\nabla U+R_B^\top(\mathbf a_{SRP}+\mathbf u)
$$

Jacobi check: $\tfrac12|\dot{\boldsymbol\rho}_B|^2-\tfrac12|\boldsymbol\omega_B\times\boldsymbol\rho_B|^2-U$ is constant.

MuJoCo grasp (lecture L1 Lagrange form):

$$
M(q)\ddot q+C(q,\dot q)\dot q+G(q)=\boldsymbol\tau+J^\top\boldsymbol\lambda,\qquad \boldsymbol\tau=M(\ddot q_d+K_d\dot e+K_pe)+C\dot q+G\ \ (\text{computed torque})
$$

- The model has a free-floating base, a heightfield terrain cut from the shape model, and local $\mathbf g_{eff}$.
- Capture merge:

$$
m\leftarrow m+m_b,\quad \mathbf c=\frac{\sum m_i\mathbf c_i}{\sum m_i},\quad J=\sum_i\left[J_i+m_i(|\mathbf d_i|^2I-\mathbf d_i\mathbf d_i^\top)\right]
$$

### E10. Kinetic impact (Cheng et al. 2023, eq. 1, verified)

$$
M_A\Delta\mathbf v=m\mathbf U+m(\beta-1)(\hat{\mathbf E}\cdot\mathbf U)\hat{\mathbf E},\qquad \Delta\boldsymbol\omega_A=J_A^{-1}[\mathbf r_{imp}\times M_A\Delta\mathbf v]
$$

- DART measured β = 3.61 (+0.19/−0.25), and 2.2–4.9 across the density range. The β slider plus a Monte Carlo cloud show the uncertainty.
- Defaults: 580 kg at 6.1 km/s gives Δv ≈ 0.17 mm/s. Δv scales as m·U, which you see live.

### E11. Control and estimation (lectures L2/L3)

**LQR:**

$$
A=I+\Delta tA_c,\ B=\Delta tB_c;\quad P=Q+A^\top PA-A^\top PB(R+B^\top PB)^{-1}B^\top PA;\quad K=(R+B^\top PB)^{-1}B^\top PA;\quad \rho(A-BK)<1
$$

**MPC** (cvxpy; terminal cost = DARE solution P):

$$
\min\sum_k\tilde x_k^\top Q\tilde x_k+u_k^\top Ru_k+\tilde x_N^\top P\tilde x_N\ \ \text{s.t.}\ x_{k+1}=Ax_k+Bu_k,\ |\mathbf f|\le T_{max}\cos\alpha,\ \hat{\mathbf e}^\top\mathbf f\ge|\mathbf f|\cos\gamma,\ \hat{\boldsymbol\rho}_0^\top\boldsymbol\rho_k\ge r_{ko}
$$

**Attitude:** quaternion PD, $\boldsymbol\tau=-K_p\,\mathrm{sgn}(\delta q_0)\delta\mathbf q_v-K_d\delta\boldsymbol\omega+\boldsymbol\omega\times J\boldsymbol\omega$, plus the L3 small-angle LQR.

**Sensors** (noise levels are config values):

| Sensor | Model |
| --- | --- |
| GPS (Earth orbit) | σ 5 m |
| Deep Space Network (DSN) fix (cruise) | σ 1 km, 1 mm/s |
| Star tracker | σ 10″ |
| Gyro | with bias |
| LIDAR | σ 0.1 m |
| Camera bearing | σ 0.1 mrad |

**Kalman filter:**

$$
\hat x^-=A\hat x+Bu,\ P^-=APA^\top+W,\ L=P^-C^\top(CP^-C^\top+V)^{-1},\ \hat x=\hat x^-+L(y-C\hat x^-)
$$

- Steady state: `dare(A.T, C.T, W, V)`.
- Separation principle: eigenvalues are $\lambda(A-BK)\cup\lambda(A-LC)$.

**EKF** on raw range and bearing:

$$
h=[|\boldsymbol\rho|;\ R^\top\hat{\boldsymbol\rho}],\qquad H=\begin{bmatrix}\hat{\boldsymbol\rho}^\top&0\\R^\top(I-\hat{\boldsymbol\rho}\hat{\boldsymbol\rho}^\top)/|\boldsymbol\rho|&0\end{bmatrix}
$$

**Vision in the loop.** Render the camera image off-screen with VTK, take the brightness centroid, and invert the pinhole model:

$$
\hat{\mathbf b}\propto R(q)R_{cam}K_{cam}^{-1}[u_c,v_c,1]^\top,\qquad K_{cam}=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}
$$

The phase-angle bias of the centroid is real and shown.

### E12. Deflection, success test, trim

Energy and semi-major axis:

$$
\dot a=\frac{2a^2(\mathbf v\cdot\mathbf a_p)}{\mu_\odot},\qquad \Delta n=-\tfrac32\tfrac{n}{a}\Delta a
$$

CW secular drift (lecture L3 eqs 3–5):

$$
y(t)=-\tfrac32a_Tt^2+\tfrac{4a_T}{n^2}(1-\cos nt),\qquad \Delta s\approx-3\Delta v(t-\tau/2)
$$

B-plane: $\Delta\mathbf b=(I-\hat{\mathbf s}\hat{\mathbf s}^\top)\Delta\mathbf r_A(t_{ca})$, with $\hat{\mathbf s}=\hat{\mathbf v}_\infty$ (Bennu relative to Earth).

Earth capture radius (gravitational focusing), used by the hypothetical-impact scenario:

$$
b_\oplus=R_\oplus\sqrt{1+\frac{2\mu_\oplus}{R_\oplus v_\infty^2}}\ \ (\approx13{,}000\text{ km for }v_\infty=6.19\text{ km/s})
$$

**Success test:**

- $|\Delta\mathbf b|\ge b_{req}$ (keyhole margin, or $b_\oplus$ + margin).
- Both spacecraft arrive before the deadline.
- Hover is feasible at the chosen d.
- The display shows ✓/✗ with the failing reason.

Rule of thumb shown live (lead time $t_{lead}$; a launch delay shrinks it):

$$
\Delta v_{req}\approx\frac{b_{req}}{3\,t_{lead}\sin\theta}
$$

**Trim guidance:**

$$
\Delta\mathbf v_{trim}=\Phi_{bv}^{+}\left(\Delta\mathbf b_{target}-\Phi_{bv}\widehat{\Delta\mathbf v}_{KI}\right),\qquad \hat{\mathbf e}=\frac{\Delta\mathbf v_{trim}}{|\Delta\mathbf v_{trim}|},\qquad \tau=\frac{|\Delta\mathbf v_{trim}|}{a_A}
$$

### E13. Numerics

- RK4 fixed step with zero-order-hold control for the live run. dt adapts per phase: about 1 h in cruise, 1–10 s in proximity.
- DOP853 (rtol 1e-12) for the STM and truth propagations.
- Ephemerides: cubic Hermite interpolation of the cached Horizons data.
- Kepler f and g functions for ZEM prediction (microseconds per call).
- Floating origin and scale exaggeration in the renderer, because VTK uses float32.

## 5. Validation suite (pytest + figures for the report)

| # | Check |
| --- | --- |
| 1 | Two-body energy and angular momentum conserved (RK4 vs forward Euler) |
| 2 | Kepler period |
| 3 | J2 node drift $\dot\Omega=-\tfrac32nJ_2(R_E/p)^2\cos i$ |
| 4 | Lambert solution re-propagates to r₂ (< 1 m) |
| 5 | Bennu vs Horizons: 2060 and 2135 close-approach times and distances match SBDB |
| 6 | STM linear prediction vs full nonlinear deflection (justifies the instant planner) |
| 7 | Polyhedron gravity: Laplacian, far field, volume × σ vs GM/G |
| 8 | Jacobi constant |
| 9 | Hover Jacobian eigenvalues (finite difference vs analytic) |
| 10 | ρ(A−BK) < 1, ρ(A−LC) < 1; separation principle |
| 11 | ZEM/ZEV: in high-thrust limits its Δv approaches the Lambert two-impulse Δv; minimum TOF decreases monotonically as thrust increases |
| 12 | Spiral escape time vs $v_{c0}/(T/m)$ |
| 13 | Gravity-tractor Δv(t) and momentum balance; Lu & Love thrust formula |
| 14 | DART check: $\Delta P/P=3\Delta v_T/v$ with −2.70 mm/s and 0.1776 m/s gives ≈ −33 min (measured −33.0 ± 1.0) |
| 15 | Capture conserves momentum; merged inertia is symmetric positive-definite |
| 16 | N-body Δs(t) vs CW prediction |
| 17 | ‖q‖ = 1, RᵀR = I, det R = 1 |
| 18 | Tsiolkovsky equation vs integrated ṁ |
| 19 | Kalman filter / EKF errors inside ±3σ |

## 6. Software layout

New uv project at `asr/project/bennu_defense/`. The professor's repo is left untouched.

```text
app/main.py                 PySide6 window: controls, 3D view (pyvistaqt), telemetry, pyqtgraph plots
app/controls.py, telemetry.py, plots.py, scene3d.py (camera presets, floating origin, models, trails, skybox)
app/workers.py              QThreadPool tasks: planner (debounced), live-run stepping, truth propagation
core/constants.py, config.py, ephemeris.py (Horizons/SBDB fetch + cache + interpolation)
core/dynamics/              gravity.py, polyhedron.py, perturbations.py, attitude.py, rotating.py, asteroid.py, stm.py
core/propulsion.py          thrust, ṁ tracking, unlimited-fuel mode
core/guidance/              lambert.py (vectorized), departure.py (impulsive + spiral), zemzev.py, targeting.py, pronav.py, trim.py
core/control/               dare.py, lqr.py, mpc.py, attitude.py
core/estimation/            kalman.py, ekf.py
core/sensors.py, vision.py, impact.py, capture.py, bplane.py
core/arm/                   mujoco_scene.xml, grasp.py
core/planner.py             instant what-if computation → PlanResult (ETA, Δv, fuel, Δb, success + reasons)
core/mission.py             live state machine + events (launch hold, safe mode) + env API
scripts/                    fetch_data.py, run_headless.py, replay_to_mp4.py, validate.py
tests/                      §5 checks
```

Reuse from the course repo (copy with attribution):

- `dare` and the PBH stabilizability test from `lecture3/util/dare.py`.
- The System/Controller/Estimator interfaces and finite-difference `linearize` from `lecture3/systems/system.py`.
- The output-folder convention from `lecture3/util/outs.py`.
- `euler_to_rotation_matrix` from `lecture1/spacecraft_sim.py` for Euler-angle plots.

## 7. Verification

1. `uv run python scripts/fetch_data.py`: caches ephemerides and assets.
2. `uv run pytest`: all §5 checks pass.
3. `uv run python -m app.main`: the app opens. Then check each behavior:
   - Increasing tractor thrust moves the ETA earlier and raises fuel use.
   - A launch delay shrinks the lead time and Δb, and can flip ✓ to ✗.
   - A bigger or faster impactor increases Bennu's Δv and Δb, and raises the required thrust and Δv.
   - "Solar storm!" during a live run delays the ETA, and the event log explains why.
4. `uv run python scripts/replay_to_mp4.py outs/<run>`: produces the movie for the professor's requirement.

Reminder of the course AI policy: export this chat as a PDF and attach it to the report.

## References (checked during planning)

- Lu & Love, Nature 438, 177–178 (2005), arXiv:astro-ph/0509595.
- Mazanek et al., "Enhanced Gravity Tractor Technique for Planetary Defense," IAA-PDC-15-04-11 (2015), NTRS 20150010968.
- Cheng et al., Nature 616, 457–460 (2023).
- DART period change, arXiv:2303.02077.
- Guo, Hawkins & Wie, "Applications of generalized zero-effort-miss/zero-effort-velocity feedback guidance algorithm," JGCD 36(3), 810–820 (2013).
- JPL SBDB (Bennu: Chesley 2020 GM; Daly 2020 shape and density; Hergenrother 2019 spin); JPL Horizons API.
- Bennu impact probability 1/2700 for 2182 (Farnocchia et al. 2021, Icarus).
- Werner & Scheeres, Celest. Mech. Dyn. Astron. (1997).
- Clohessy & Wiltshire, J. Aerospace Sci. 27(9) (1960).
- Curtis, Orbital Mechanics for Engineering Students; Vallado; Markley & Crassidis (2014); Zarchan, Tactical and Strategic Missile Guidance.
- NASA 3D Resources (Bennu glTF, Dawn GLB), NASA SVS #5069; pyvistaqt docs.
