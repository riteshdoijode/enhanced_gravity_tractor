# Bennu Defense — Mission Control

Interactive planetary-defense simulator for asteroid **101955 Bennu**: a DART-style kinetic impactor
(primary deflection) plus an Enhanced Gravity Tractor (precision trim), both starting from Earth orbit.
Autonomous Space Robotics, Fall 2026. Developed and tested on macOS (Apple Silicon). Windows and Linux are untested; see
[Running on Windows or Linux](#running-on-windows-or-linux). Papers and data sources: [references.md](references.md).

**Mission goal (keyhole scenario):** Bennu's 2135 flyby, between the Earth and the Moon, is assumed to pass
through the gravitational keyhole that brings it back to hit Earth on 2182-09-24. The DART (one or more, one at
a time) and then the gravity tractor must push that flyby out to at least the Moon's distance. See
[Mission goal: the 2135 keyhole](#mission-goal-the-2135-keyhole-and-the-2182-impact).

## Setup

```bash
uv sync                                   # Python 3.12 + dependencies
uv run python -m scripts.fetch_data       # JPL ephemerides, SBDB, NASA 3D models, star sky, Milky Way (~285 MB, ~4 min)
uv run python -m scripts.fit_bennu        # estimate Bennu's Yarkovsky/SRP parameters vs JPL (~45 s)
uv run python -m scripts.precompute       # Bennu nominal orbit + state-transition matrix 2026-2185 (~100 s)
uv run python -m scripts.make_scenario    # keyhole scenario: 2135 keyhole -> 2182 impact (~2 min; --rescan ~12 min)
uv run pytest                             # validation suite (130 tests), 6 parallel workers (~6 min)
uv run python -m app.main                 # the simulator (all 12 cores; --cores N for fewer)
```

Other tools:

```bash
uv run python -m scripts.validate         # report figures -> outs/validation/, in parallel (~1.5 min)
uv run python -m scripts.snapshot --all   # render each 3D view off-screen to outs/snapshots/
uv run python -m scripts.fetch_data --only ephem|sbdb|assets|sky
uv run python -m app.impact_show          # Bennu hits Earth in 2182 (no deflection), on its own
```

### Running on Windows or Linux

The project was only ever run on a Mac, so nothing below has been tested elsewhere. The code is written to degrade
gracefully: the macOS-only calls (thread quality-of-service classes, `sysctl` for core counts in `core/parallel.py`)
fail quietly and fall back to `os.cpu_count()`, and worker processes always use the `spawn` start method, which is
what Windows uses. Install [uv](https://docs.astral.sh/uv/) and run the same commands from PowerShell or a terminal.

- `data/` is not in the repository (about 190 MB, regenerated). The four setup commands above recreate it, in order:
  `fetch_data` (needs internet; JPL Horizons and NASA servers), `fit_bennu`, `precompute`, `make_scenario`.
  Run them before `app.main`.
- To only read the code, none of that is needed: start with `core/` (physics), `scripts/validate.py` (figures) and
  `tests/` (what each piece is checked against).
- `--cores N` limits the CPU use; the default is every core.
- If the 3D window does not open, the usual causes are the graphics driver (PyVista/VTK needs OpenGL) or a missing
  wheel for Python 3.12; the project pins Python 3.12 (`requires-python`).

## Using the app

**Layout.** Controls on the left. The **Solar System** stays in the center: the Sun, the eight planets and Pluto
at their JPL positions (Neptune's and Pluto's orbits, longer than the 2026–2185 ephemerides, are closed with a
two-body arc). **🪐** zooms out to the whole system, out to Pluto's orbit, and back. **🏠** shows it from the side
as it moves around the Milky Way, with the paths the planets trace; see [below](#the-sun-around-the-solar-system-barycenter-coredynamicssolar_systempy).
On the right: **Bennu** at true
scale (always), the **spacecraft** view below it (Auto follows the DART until its impact, then the tractor;
DART / Tractor pin one; ⟲ resets the chase camera), and the info tabs (Mission plan, Live flight, Bennu now).
Bottom tabs: Earth–Bennu distance with the DART opportunities, the 2135 B-plane, the Earth–Moon view of the
keyhole flyby, the DART porkchop, live plots, near-Bennu plots, tractor arrival vs thrust, the Sun.
Mouse in any 3D view: left-drag rotate, right-drag/scroll zoom, middle-drag pan.

**The background of every view, the Solar System included, is the real sky** (core/sky.py, app/scenes/sky.py),
in one of two looks:
- **The Milky Way** (default): NASA SVS's *Deep Star Maps 2020*, built from Gaia DR2, Hipparcos-2 and Tycho-2
  (1.7 billion stars). The 8192 × 4096 map is used in the main views and a 4096 × 2048 copy in the small ones.
  It is converted once from linear light to sRGB with an exposure of 3, so the band and its dust lanes show
  without saturating the core.
- **Constellations** (the Display toggle "Constellation background"): the 41,000 catalogue stars down to
  magnitude 8, sized by magnitude and coloured by their B−V index, with the 88 IAU figures and names.

Switching cross-fades over 3 s: the Milky Way fades out as the stars, figures and names fade in, in every view
at once. Both are at their J2000 (ICRF) positions and rotated into each view's axes. The sky is infinitely far:
turning a camera turns it, panning or zooming does not move it.

**Session log.** Every session writes `outs/logs/session_YYYYmmdd_HHMMSS.jsonl` (the "log" label in the status
bar shows the path), one JSON object per line:
- every click: the widget, its text, and whether it was enabled;
- every parameter change, and whether it re-flies a live run;
- every state change of the launch buttons, with the function that made it;
- what a launch press did, or why it did nothing;
- the commands sent to the simulation;
- the DART verdicts, the tractor and next-DART plans;
- the mission's event log and a state summary;
- planner-job errors, Python exceptions and Qt warnings.

It is flushed line by line, so a session that crashes is still recorded. It is for tracing a problem seen in an
interactive run.

**Planning (before launch).**
- **Mission goal**: the 2135 flyby distance to reach (≥ 1 LD) and the **DART target shift** (km), i.e. the
  aim point the tractor will steer the flyby to.
- **DART opportunities** at every close pass of Bennu: markers on the distance plot (click one to select it, or
  use the combo box). Each marker shows what **the DART as set** (mass, max speed, β, β uncertainty, sizing) does
  in that window; see [DART window markers](#dart-window-markers-the-distance-plot). **Show DART window markers**
  (next to the plot's tab) hides them for a clean plot.
- **Launch no earlier than** (DART section): the plan uses only launches from this date on (plus 40 days of
  orbit raising): the best arc of the selected window after it, or the recommended window still open. Dragging the
  time slider by hand sets it to the slider's date ("follow the time slider"; Play and Jump to… do not), and a
  date typed or picked moves the clock there. A date before a window's best arc leaves the plan unchanged.
- **DART**: recommended or manual mass; **orbit-raising cut-off speed (max velocity)**: the DART raises its
  orbit with burns at perigee and leaves Earth when its perigee speed reaches the planned departure speed, at
  most this value; β and its uncertainty.
- **Mission plan** tab: verdict and probability over β, the shift needed / targeted / delivered, the DART
  plan (dates, impact speed, mass, point of contact, every orbit-raising burn with its fuel), the tractor that
  would follow, the opportunity table and the checks. The Bennu view marks the DART aim point.
- **The gravity tractor's launch window** (blue, on the distance plot): a translucent band over the launch dates
  that still reach Bennu in time for a tow that closes the remaining gap (from the planned DART's verdict on;
  its dates are in the legend), and the planned tractor's Earth distance rising from Earth to meet Bennu's
  orange curve at the intercept. In a live run it is recomputed for the 2135 point measured after the verdict.
  The Solar System view draws the tractor's planned path in the same blue.
- **Recommended mass, two stages.** The mass is first sized with the linear estimate (instant). About 90 s
  later it is re-sized with the full force model, run in a background process, and the plan says "sized with
  the full force model". Large deflections through the 2060 flyby grow faster than linear, so the
  full-model mass is smaller.

**Planning mode vs live run.** Before a launch the views *preview the plan* (planned trajectories; no physics is
run, and ▶ Play just animates that preview). **🚀 Launch DART** starts the *live run*: from then on everything
on screen is computed by the physics in the simulation process, nothing is replayed.

**Live run.** Pick a launch window (the DART combo box, or click its marker on the distance plot) and press
**🚀 Launch DART**. If that window is not planned yet, the launch waits for its plan. The run starts in that
window, 10 days before the DART's orbit raising, with both spacecraft in the parking orbit. The DART flies the
plan for the **Launch no earlier than** date (a slider dragged a moment before the press is taken first). If the
clock has run past the window's orbit raising (Play), the DART is re-planned for a launch from the clock's time in
the same window; if the window is over, a dialog lists the windows still ahead and offers the next one.
- **Time runs at the speed you choose.** If the physics cannot keep up (long proximity phases at a high
  speed), the clock shows only what has been computed and the Live flight tab says so. If it must wait for a
  computation (Bennu's orbit, an orbit determination, a re-simulation), the 3-D views make way for a
  **Recalculating physics** page with its progress.
- **Slow motion at Bennu** (on by default; the checkbox under the speed list switches it off). The display slows
  down so the close-ups are not missed, then returns to your speed. A skip stops where a close-up begins. Only
  the display speed changes; the physics is the same at any speed. The banner on the Bennu view and the Live
  flight tab say what is shown. The Bennu view turns to the aim point when the DART's close-up begins.

  | Close-up | Display speed | Real time |
  |---|---|---|
  | DART, last 5 minutes before the impact (until 1,000 km) | 30 s per second | ~7 s |
  | DART, last 1,000 km to Bennu (~80 s at 12 km/s) | 5 s per second | ~15 s |
  | DART, last 10 seconds (it enters the Bennu view 150 km out) | real time | ~10 s |
  | Impact: flash and ejecta curtain (first 40 s) | 2 s per second | ~20 s |
  | Impact: ejecta cone spreading (to 20 min) | 1 min per second | ~20 s |
  | Tractor descent above / below 150 m | 5 min / 1.5 min per second | ~20 s / ~40 s |
  | Boulder capture (deploy, reach, grip, lift, stow) | 15 s per second | ~45 s |
  | Lift-off with the boulder (first 15 min) | 1 min per second | ~15 s |
  | Tractor, last 1,000 km to Bennu (until the proximity operations) | 10 min per second | depends on the closing speed |
  | Bennu, last 1,000 km to Earth (2182, when it still hits) | 5 s per second | ~16 s |
  | After an Earth impact: the plume (20 min) | 1 min per second | ~20 s |
- **Bennu is integrated, not looked up.** From the first push on, Bennu's orbit is integrated with the full
  force model (core/truth.py), with every impulse it actually received and the tractor's actual pull. Before
  any push it is on the scenario orbit, itself a full-force-model integration. The 2135 flyby and the 2182
  impact (or miss) are what that integration gives when the clock gets there.
- **Changing a parameter during the run re-flies the mission.** Any control (DART window, mass, maximum
  speed, β, goal, target, tractor settings) flies the whole mission again from its start with the new value.
  Launches and storms are replayed at their times. A later DART or the tractor is re-planned for the state
  the new run reaches, and dropped if that run no longer needs it. The same parameters give the same run back,
  bit for bit; the physics does not depend on the display speed (every craft steps on its own physical grid).
  - A change to the **gravity tractor's** settings keeps the DART(s) that flew. Before, the re-fly re-planned
    DART #1 with the recommended mass as it was by then (often the full-model re-sizing, finished after the
    launch), so a tractor setting could turn a success into a different run.
  - The re-fly goes up to where the clock had stopped, **including a DART verdict due there**. The run pauses
    on each verdict for your decision; before this fix, a change made at that moment re-flew to the verdict's
    date without the verdict, so the tractor could not launch.
- The DART raises its orbit (fuel shown in the Live flight tab and the craft view), injects at perigee,
  cruises with trajectory corrections and homes on the point of contact; the clock stops at the impact
  instant. The ejecta are drawn as sunlit dust, with an impact flash. ⏭ skips to the next event.
- Ground tracking after the impact gives β, and the full-model prediction of the 2135 flyby re-anchors the
  linear one; the verdict comes 30 days after the impact, and the clock pauses for your decision (press Play
  to go on without launching):
  - **success** (≥ 90 % of the target): **🚀 Launch gravity tractor** is available at once. The tractor is
    planned ahead, at the impact, for the verdict's date and the shift the team expects, so the button is ready
    the instant the verdict comes; the verdict then refreshes the launch window for the measured shift. While the
    button is available the plan is re-made for "launch now" every 30 mission days, so a later press flies a
    plan for that date (and the arrival it shows). A coloured **status strip** above the button says why it can or
    cannot launch: red (not available, and why: launch a DART first, the DART fell short, the tractor cannot reach
    Bennu with these settings), amber (waiting for the verdict, with its date and whether the clock is paused),
    green (**READY**: launch now → arrival date; launch window open until date), grey (launched: phase and
    arrival). A change to a tractor setting before its launch re-plans the tractor only; nothing that has flown
    depends on it, so the mission is not flown again;
  - **short / missed**: the recommendation for the **next DART** (velocity, mass, point of contact at the
    next opportunities, with the measured β) appears in the Live flight tab. The distance plot's markers then
    show what that next DART does in each remaining window, from the deflection reached so far. Click a marker to
    pick another window; **🚀 Launch the next DART**.
- The tractor flies as in Steps 4–7: escape spiral, low-thrust cruise, proximity operations, boulder
  capture, tow with the trim law to the DART aim point. ☀ Solar storm; MPC/LQR. At the hand-over from the
  escape spiral its guidance starts from the ground's optimized flight plan, the one the status strip quoted
  ("Flight plan: the ground profile"), and refines it from the actual state.
- **End of the tow**: **⤒ Release the boulder and end the tow** (enabled while the tractor holds its station,
  towing or parked) lets the boulder go where the craft is. The boulder stays there as a marker and the
  spacecraft is 10 t again. The craft holds 30 minutes, climbs radially to 50 km (`depart`), burns 0.5 m/s and
  leaves (`departed`); Bennu is then flown on with the full force model and its prediction is requested. The tow
  also ends by itself when `tow_years` is up, or once the trim has held (parked, tow off) for 60 days.
- **Bennu hits Earth** (centre page): when the clock is in the last two days before an Earth impact, or the 20
  minutes after it, the centre shows it (app/scenes/earth_impact.py). In planning mode that is the scenario
  without a deflection (Jump to… "Earth impact 2182"); in a live run, the run's verified orbit if it still hits.
  It shows Bennu's track, the last 1,000 km in slow motion, a flash and a ballistic plume (display model), and a
  caption with the site and the entry numbers. A live run stops at the impact (the physics ends there); the page
  then plays the 20 minutes after it and returns to the Solar System. `uv run python -m app.impact_show` plays it
  on its own; see [Bennu hits Earth](#bennu-hits-earth-in-2182-coreearth_impactpy).
- ⏩ **Run to the end**: the clock runs as fast as the physics allows through the 2135 flyby to 2185; the
  distance plot shows the full-model orbit in green and the clock runs on to "Earth is safe" or the impact.

## Data sources

| Data | Source |
|---|---|
| Sun, planets, Moon, Bennu states 2026–2185 (Bennu to 2135-09-30) | JPL Horizons API (DE441; Bennu solution JPL#118, Farnocchia et al. 2021) |
| Bennu GM, density, size, spin, Earth approaches | JPL Small-Body Database API |
| Bennu shape (3.37 M faces, 88 cm) and Nightingale TAG-site patch (5 cm) | NASA SVS #5069, OSIRIS-REx OLA v20 |
| OSIRIS-REx spacecraft model | NASA Science 3D Resources (glTF) |
| Earth, Moon, Sun textures | PyVista example datasets |
| Star sky: 41,411 stars to magnitude 8 (J2000, V, B−V) and the 88 constellation figures | d3-celestial (Olaf Frohn, BSD-3): stars from XHIP, Anderson & Francis 2012 (VizieR V/137D); figures from the IAU constellation pages |
| Milky Way background (8192 × 4096, ICRF plate carrée) | NASA/Goddard Space Flight Center Scientific Visualization Studio, Deep Star Maps 2020 (SVS 4851); Gaia DR2: ESA/Gaia/DPAC |

## Step 1 validation (tests/test_step1_data.py)

- Cubic-Hermite ephemeris interpolation vs off-node Horizons samples: Earth < 55 m, Moon < 11 m,
  Bennu < 18 m (including through the 2135 flyby), Jupiter < 1 m.
- Earth–Bennu close approaches recomputed from our ephemeris match all 36 SBDB entries (2030–2135)
  to < 1 min and < 1.5 km, including **2135-09-25 at 0.0013639 au** (inside the Moon's orbit).
- Shape-model volume × SBDB density gives GM = 4.888 m³/s² vs measured 4.8904 (0.05 %);
  max-inertia axis coincides with the body z (spin) axis.
- Physics mesh (5120 faces, radially remeshed) is watertight with outward normals; volume within 0.25 %.

**Finding:** before ~2031-06, Horizons' Solar-System-Barycenter Bennu states are offset by ~116 km from
the DE441 barycenter (Bennu's trajectory was integrated with DE424); Sun-centered states are consistent,
so Bennu is fetched heliocentric and shifted with the DE441 Sun.

## Step 2 physics (core/dynamics/)

| Module | Model | Lecture link |
|---|---|---|
| `gravity.py` | N-body point masses, inertial or central-body form (indirect term), Earth J2, gravity gradient, solar GR | L1 eqs 1-2 |
| `perturbations.py` | exponential atmosphere drag (Vallado Table 8-4), cannonball SRP + cylindrical shadow, Yarkovsky A2 | L1 special cases |
| `attitude.py` | quaternion kinematics, Euler's equation, gravity-gradient torque, 3-2-1 Euler conversion | L1 eqs 3-4, SO(3) |
| `spacecraft.py` | 6-DOF spacecraft around Earth / Sun / Bennu, thrust + torque inputs, propellant tracking (unlimited-fuel mode) | L1 |
| `asteroid.py` | Bennu barycentric propagation: planets, Moon, 4 largest asteroids, Earth J2, GR, SRP, Yarkovsky | L1 eq 2 |
| `nominal.py` | Bennu nominal orbit + state-transition matrix (symplectic), deflection mapping to Earth encounters | L2/L3 linearization |
| `polyhedron.py` | constant-density polyhedron gravity (Werner & Scheeres 1997) from the OSIRIS-REx shape | |
| `rotating.py` | equations of motion in Bennu's rotating frame, Jacobi integral | |

Step 2 validation (tests/test_step2_physics.py, figures in outs/validation/):

- RK4 holds two-body energy to 3e-10 over 10 orbits; the lecture's forward Euler drifts 11 % even at 1 s steps.
- Kepler period closure; J2 node drift within 0.4 % and drag decay within 0.5 % of analytic theory.
- Central-body (indirect-term) form equals lecture eq. 2 in the inertial frame to 1e-12.
- Tsiolkovsky equation; propagation of propellant in unlimited-fuel mode.
- Torque-free tumbling conserves angular momentum (7e-13) and energy (1e-14); quaternions match the
  lecture's 3-2-1 Euler kinematics to 1e-12.
- Polyhedron gravity: Laplacian = -4πGσ inside / 0 outside, far field -> point mass, a = ∇U;
  Jacobi constant conserved to 1e-11 in the rotating frame; rotating and inertial propagation agree to µm.
- Bennu vs JPL: non-gravitational parameters re-estimated by least squares on JPL's 2026-2059 trajectory
  (fitted A2 = -4.541e-14 au/d² vs published -4.618e-14; rms 0.69 km). Over 110 years our propagation
  stays within JPL's own 3σ uncertainty at the 2060 (±3 km), 2080 (±1,283 km) and 2135 (±9,200 km)
  Earth approaches.
- STM is symplectic to 1e-19; its linear deflection prediction matches two full nonlinear propagations
  to 1.5e-4 through the 2060 flyby and tracks them until the shift reaches ~10⁶ km (~2090).
  Bennu's Earth flybys amplify any pre-2060 change ~100×.

## Step 3 planner (core/planner.py, core/guidance/, core/deflection.py)

- **Tractor**: Earth-escape spiral for any thrust (orbit-averaged Gauss equation + numerical tail), then a
  heliocentric Lambert arc with 0–3 revolutions whose departure + arrival burns must fit in the trip,
  (Δv₁+Δv₂)/a ≤ 0.8·TOF (impulsive → finite-burn estimate). "Fastest" = earliest feasible arrival;
  "Efficient" = least Δv within 6 years. Boulder capture, Lu & Love hover sizing, tow until the encounter
  in the direction that grows the shift most.
- **Impactor**: porkchop of Lambert intercepts from the parking orbit (patched conic); picks the arc with the
  largest B-plane shift within the launch-Δv budget, preferably arriving after the tractor (observed).
  Momentum transfer: Cheng et al. (2023) eq. 1.
- **Deflection**: impact Δv and integrated tow mapped to the 2060 B-plane through Bennu's STM (Step 2).
  Exercise scenario: success if the shift exceeds Earth's capture radius (gravitational focusing) +10 %.
- **Scope of the estimate**: heliocentric legs are two-body and burns are finite-burn approximations of
  impulses; Step 4 flies the plan closed-loop in the full N-body model.

Step 3 validation (tests/test_step3_planner.py):

- Lambert reproduces Curtis Example 5.2; 1,600 Earth→Bennu porkchop arcs re-propagate onto Bennu
  (median closure < 10 m); multi-revolution solutions close to < 1e-7 au.
- Universal-variable Kepler propagation (safeguarded Newton) matches RK4 to < 1 mm for elliptic,
  hyperbolic, backward and e ≈ 0.97 arcs.
- Escape: C3 = 0 burn = (√2−1)v_c; low-thrust spiral costs ≈ v_c (classical result); more thrust is faster.
- DART: the momentum model with measured Δv_T reproduces the −33 min period change.
- Gravity tractor: reproduces Lu & Love's 20 t / 1 N example and their Δv-per-year formula.
- Planner: arrival is monotone in thrust; impact shift is exactly linear in mass and β; success logic;
  a launch delay changes the outcome (launch-window geometry); plans in well under 1.5 s.

## Step 4 live run (core/mission.py, core/guidance/lowthrust.py, core/estimation/navfilter.py)

The plan from Step 3 is flown in the full model: the truth state is integrated with RK4 (steps ≤ 12 h in cruise)
under Sun, Earth, Moon, Venus, Mars and Jupiter gravity (geocentric with Sun + Moon during the escape),
with thrust execution errors (1 % magnitude, 0.3° pointing, re-drawn at every guidance update).

- **Tractor escape**: thrust along the velocity in the orbit-averaged spiral, then a Cartesian tail; the
  heliocentric phase starts only once the orbit is hyperbolic (C3 ≥ 0) and beyond 0.25 of Earth's
  sphere of influence.
- **Tractor cruise guidance (direct transcription)**: the remaining trip is split into N segments, each a
  Kepler coast followed by a velocity impulse (Sims–Flanagan style). The impulses that meet Bennu's
  position and velocity with the least Σ|Δv|² are solved by damped Newton / minimum-norm least squares,
  with the Jacobian chained from finite-difference Kepler state-transition matrices. Many arrival dates
  are solved at once (batched); starting guesses come from every multi-revolution Lambert branch.
  "Fastest" = earliest arrival whose peak acceleration is ≤ 0.9 of the available thrust/mass and whose
  perihelion stays ≥ 0.3 au.
- **Closed loop**: the profile is re-solved from the navigation estimate every 2 days, more often near
  arrival (receding horizon, warm-started). After 3 failed re-solves, or after a throttle change or solar storm, a "ground" re-plan
  searches arrival dates again. After an outage the original arrival date is kept if still reachable,
  otherwise it slips.
- **Navigation**: Kalman filter with dynamic model compensation. The state is position, velocity and an
  unknown acceleration (the thrust error), modelled as a Gauss–Markov process (τ = 2 d). It takes daily
  radiometric fixes (σ 1 km, 1 cm/s) with a Joseph-form update. A plain 6-state filter was overconfident
  (69 km errors with a 2 km 3σ bound); with the bias state the errors stay inside 3σ.
- **Arrival**: handed to proximity operations (Step 5) within 100 km of Bennu and below 2 m/s relative.
- **Impactor**: patched-conic departure from the parking orbit, then four Lambert correction burns at
  2, 30, 70 and 95 % of the flight, using the filter estimate. The terminal phase is a zero-effort-miss
  intercept law, a = 3·ZEM/t_go², with 0.05 m/s² of thruster authority and optical navigation relative to
  Bennu. Impact is detected on the closest approach within each step. Momentum transfer uses Cheng et al.
  eq. 1 with a β drawn from the slider's uncertainty (unknown until the impact).
- **Outlook**: the projected B-plane shift is updated live from the actual impact Δv plus the remaining tow,
  through the Step 2 state-transition matrix.

Typical results (exercise scenario, 40 t impactor, seeds vary):

| Case | Result |
|---|---|
| 20 N tractor | arrives ≈ 2028-06 at 0.3–15 km from Bennu, 1–2 m/s relative |
| 2 N tractor | arrives 2029-12-19 within 39 km |
| Impactor | closest approach 2–115 m from Bennu's center (radius 242 m), impact at 12.0 km/s |
| Deflection | Bennu Δv 22–25 mm/s → 2060 shift 33,000–38,000 km vs 14,647 km required |
| Optimized vs instant plan | 2 N: 2029-12-20 vs 2029-09-05 · 20 N: 2028-07-19 vs 2027-11-30 · 200 N: 2027-05-14 vs 2027-03-13 |

The instant estimate (Step 3) assumes the Lambert burns can be spread over the trip, so it is optimistic
for low thrust. The optimized plan is what the live run flies, and the panel shows both.

Step 4 validation (tests/test_step4_live.py):

- Direct transcription: the converged impulses, re-flown, meet Bennu to < 5 km and < 0.5 m/s; the peak
  acceleration stays within 90 % of the thrust limit; perihelion ≥ 0.3 au.
- Full live mission in under 2 min: escape → cruise → arrival within 100 km; launch → corrections →
  impact; Bennu's Δv equals Cheng et al. eq. 1 to 1e-9; navigation error inside 3σ more than 95 % of the time;
  the projected outcome is a success.
- A 40-day solar storm logs safe mode and the post-storm re-plan, and never makes the arrival earlier.
- Throttling up from 10 N to 40 N re-plans and never delays the arrival.
- A storm while the impactor is in flight (correction burn deferred) does not stall the simulation.
- Dynamic model compensation recovers an unknown constant acceleration to within 30 %.

**Limits.** The cruise profile minimizes Σ|Δv|² for the earliest feasible date. That is close to, but not
strictly, minimum time. On this Mac the live run reaches about 40 days per second of time warp.

## Step 5 proximity operations (core/proximity/, core/control/, core/dynamics/fastgrav.py)

The live mission now continues after the cruise. The cruise aims at a point 30 km sunward of Bennu; from
the hand-over the tractor flies, in the Bennu-centered frame (all on its navigation estimate):

| Phase | What happens | Model / method |
|---|---|---|
| approach | to a hold point | ZEM/ZEV guidance with Bennu's gravity (Guo, Hawkins & Wie 2013) |
| stand-off | hold 10 km out, sunlit side, off the impactor's path, until the impact | LQR; the EKF carries β as a state and measures it from Bennu's velocity jump |
| survey | hover 780 m above the capture site, co-rotating with Bennu (6 h) | body-frame LQR/MPC; lift-off thrust check |
| descent | down the site's vertical, min-jerk profile, peak 15 cm/s | linearized rotating-frame dynamics (Coriolis + centrifugal + polyhedron gravity gradient); MPC glide-slope cone 60° |
| capture | arm grasps and lifts a 20 t boulder | MuJoCo (below) |
| ascent | climb to 200 m with spacecraft + boulder | body frame |
| transfer | radial climb + arc to the tractor station | inertial tracking with a keep-out sphere |
| station | hover at d = 2 R along the tow direction | thrusters canted so the plume misses Bennu (Lu & Love): thrust confined to a cone around the outward axis |

- **Gravity**: the 5,120-face polyhedron (Werner & Scheeres), compiled with numba (~0.1 ms per call,
  ~5 × 10⁻¹⁵ relative difference from the NumPy reference), with its gravity-gradient tensor for the
  linearizations and the filter. Point mass beyond 3 km (difference 3 × 10⁻¹⁰ m/s²). Truth also includes
  the solar tide, solar pressure with Bennu's shadow, and 1 % / 0.3° thrust errors.
- **Control** (lecture 3): ZOH discretization, DARE from the lecture code (`core/control/dare.py`), LQR
  u = u_ff − K δx with feedforward of gravity, solar pressure and the estimated bias. **MPC** (cvxpy +
  Clarabel, 30 steps, terminal cost = DARE solution) adds hard thrust-magnitude and plume-cone limits
  (second-order cones) and soft keep-out / glide-slope constraints. States are scaled by the bandwidth for
  conditioning, and a solve takes ~5 ms.
- **Navigation**: EKF on raw LIDAR range and camera bearing. The LIDAR is ray-cast on the 400k-face
  global model, or the 5 cm Nightingale model near the site; its Jacobian is −nᵀ/(nᵀb). Beyond 7 km the
  range comes from Bennu's apparent size. State: position, velocity, unmodeled acceleration
  (Gauss-Markov) and β.
- **Capture** (MuJoCo 3.16): site frame from the real Nightingale terrain (heightfield from the
  OSIRIS-REx 5 cm model, lat 56.05°, lon 43.15°, at the flattest 3 m spot). Gravity is the effective
  gravity there (7.3 × 10⁻⁵ m/s²), plus the Coriolis force on every body.
  - The bus is free-floating. Its thrusters (limited to the tractor thrust) and control-moment gyros hold
    position and attitude with quaternion PD.
  - Arm: 4 joints (3 m + 3 m + 0.6 m) and 3 claws, with computed-torque control
    τ = M(q)(q̈_d + K_d ė + K_p e) + C q̇ + G. M and the bias come from MuJoCo. After the grip the
    controller switches to a merged-mass model (boulder rigid in the palm).
  - Microspine grip: adhesion while pressing, then a palm–boulder weld once the palm touches.
  - The boulder is a synthetic convex rock of 2,000 kg/m³ (3.6 × 2.7 × 2.1 m for 20 t).
  - Sequence: deploy → reach → grip → lift (bus thrusters) → stow. Mass properties then merge with the
    parallel-axis theorem.
- **Interactions**:
  - If the thrust cannot lift spacecraft + boulder (1.3 × weight, 2.86 N for 30 t), the capture waits
    for the throttle to be raised. After 14 days the tractor continues as a classic gravity tractor.
  - The projected deflection uses the β measured by the tractor and the actual tow mass and start.

Typical live run (20 N, 40 t impactor, plan timed against the optimized arrival):

| Event | Result |
|---|---|
| Stand-off | arrives, holds at 10 km, watches the impact |
| β measurement | 3.81–3.84 ± 0.01–0.10 over three runs (true 3.84) |
| Descent | ends 0.04–0.13 m from the grasp point, ≤ 0.5 m from the site's vertical |
| Capture | lifted 4.6 m, grip slip 4 cm, ~11 min |
| Station | reached 0.1 m from the station; navigation error inside 3σ 99.6 % of the time, 6–12 cm at the end |
| Compute | arrival → station in about 1 minute of computation, plus the stand-off |

With 2 N the capture waits for thrust, then the tractor proceeds without the boulder.

Step 5 validation (tests/test_step5_prox.py, figures `prox_hover`, `prox_ops`, `capture`):

- Compiled polyhedron = NumPy reference (1e-12); gradient = finite differences (1e-6), symmetric and
  trace-free outside; trace = −4πGσ inside.
- Hover linearization eigenvalues ±√(2μ/d³), ±i√(μ/d³) (e-folding ~1 h at 2 R).
- Lecture 3: ρ(A−BK) < 1, ρ(A−LC) < 1, and the observer-based closed loop has exactly
  eig(A−BK) ∪ eig(A−LC) (separation principle). ZOH discretization checked.
- Thrust projection onto the plume cone.
- **LQR vs MPC** acquiring the station (2 N, 30 t, 135 m off, 7 cm/s drift, rotating polyhedron field):
  - Both respect the cone (LQR only by clipping; its raw command points outside the cone > 20 % of the
    time).
  - The MPC settles in 1.8 h vs 3.7 h, with a 2.6× lower quadratic cost and 7 % less Δv.
- EKF consistency in closed loop (errors inside 3σ > 95 %).
- Parallel-axis merge = brute-force point clouds; free-floating capture conserves linear momentum.
- Capture: lifts > 3 m, slip < 10 cm, free-space tracking error < 0.02 rad.
- Full timeline in the live mission:
  - phases in order;
  - station error < 1 m;
  - navigation filter consistent;
  - the descent stays inside the glide-slope cone.

**Limits.**
- Outside the capture, attitude is assumed to track its command; full 6-DOF dynamics run only in
  MuJoCo.
- The filter uses the same shape model as the truth, with noise inflated for model error.
- A single arm (the ARRM concept used two arms plus legs).
- The boulder is synthetic.
- The tow itself (years of station-keeping and trim) is Step 7.

## Multi-core execution (core/parallel.py, core/fullmodel.py, app/livesim.py, core/live_runner.py)

The hot paths are compiled with numba and release Python's GIL, so they run truly in parallel. Batches
are split over each process's thread pool. Where Python itself is the bottleneck, separate processes are
used instead. All of it shares one core budget (below).

| What | How | Before → after |
|---|---|---|
| Kepler propagation (low-thrust guidance, flight plans) | compiled kernel, batches split over threads | optimized flight plan 4.7 s → 0.17 s |
| Polyhedron gravity batches (MPC look-ahead, ejecta) | compiled, GIL released, split over threads | 2,000 points 495 ms → 32 ms |
| Earth-escape spiral | compiled RK4 | thrust sweep 1.6 s → 0.38 s |
| Impactor porkchop (38k Lambert arcs) | NumPy batches on threads | full plan 0.44 s → 0.20 s |
| Capture Coriolis forces | vectorized from MuJoCo's cvel | capture 13 s → 7 s |
| Live mission, launch → capture (scripted) | the kernels above | 85 s → 47 s |
| Live simulation in the app | own **process**, started with the app and kept warm; the UI process only renders | physics no longer shares the GIL with rendering; instant launch |
| Bennu's live orbit (full model, both runs in one solver) | full-model worker process, ahead of the mission clock | ~3 years of orbit per second; a 2135 prediction ~35 s |
| Full-model DART mass refinement | 5 candidate masses + the run without pushes in 6 processes, then a ±1.5 % check round | 89 s → 39 s; 20 s when an earlier result narrows the first round |
| Test suite | 6 pytest-xdist worker processes | 193 s → ~90 s (more tests) |
| Report figures | one process per figure | ~5 min → ~1.5 min |

Not everything benefits. Threads made the thrust sweep slower while it was still pure Python (GIL
contention), so the two spacecraft are not stepped on separate threads. Threaded and single-threaded
results are bit-identical, and so are full-model runs in worker processes and in the calling thread (tests).

### Core budget (`--cores`)

`uv run python -m app.main --cores N` sets how many cores the simulator may use; the default is all of them
(12 on the M3 Pro: 6 performance + 6 efficiency). The number reaches every process in `BENNU_CORES`, and
`budget()` shares it out. For 12 cores:

| Process | Share |
|---|---|
| GUI | main thread + 2 kernel threads |
| simulation | main thread + 4 kernel threads, plus 2 full-model processes for Bennu's live orbit |
| 3 planner workers | 2 kernel threads each (Lambert porkchops, Kepler batches) |
| refinement worker | 6 full-model processes (candidate DART masses side by side) |

Batches are cut into more slices than threads when efficiency cores are in use, so the faster cores take more
of them. The status bar shows the split ("12 cores", tooltip for details). Measured in the live smoke run, the
app peaked at about **11.4 cores** busy (6 refinement processes, 2 full-model processes and the simulation at
330 %).

macOS cannot reserve cores for one program: there is no CPU-affinity call. Instead every thread of the app
asks for the *user-initiated* quality-of-service class (the window's thread for *user-interactive*), so the
scheduler prefers the performance cores for it and does not throttle it in the background. Worker processes
exit on their own if the app dies, and a superseded refinement stops its workers at once.

### The app's processes and the GUI thread (app/compute.py, app/main.py, app/scenes/base.py)

**Processes.** Qt and VTK must draw on one thread, so the app splits the work across processes:

| Process | Does |
|---|---|
| GUI (main) | widgets and the four 3-D views; draws only |
| simulation | the live run (core/live_runner.py), with numba threads; 2 full-model processes |
| `plan` worker | mission plans: opportunities, DART, tractor estimate |
| `tractor` worker | optimized flight plan, arrival vs thrust, the Sun plot |
| `live` worker | next-DART recommendation, tractor plan after a success |
| refinement worker | full-model DART mass (core/refine.py); 6 full-model processes |

Workers keep their caches warm and are terminated on exit, including any job still running. Before this,
planner jobs ran on threads of the GUI process; while a porkchop or ray cast held the interpreter, the window
stalled.

**GUI thread.** These changes were measured with cProfile on the GUI thread, over 40 s at 1 year/s:

| Change | Effect |
|---|---|
| Explicit drawing: no pyvistaqt 5 Hz timer, no thread per render request, implicit "render after change" disabled; a draw is a Qt repaint, which renders once | VTK renders 5,169 → 929 per 40 s |
| Labels and overlay text kept as persistent actors moved in place, not re-created every frame | 2,468 re-creations (3 s) → none |
| Static plot items (the 60,000-point distance curves, axes, markers) cached as pixmaps; curves min/max-decimated | plot painting 20.5 s → 1.5 s during playback |
| Bennu and craft views and the time cursors at ~10 Hz, info panels and plots at ~4 Hz on screen and ~1 Hz hidden (the live panel at once on any change) | |

Overall in a live run the GUI process now uses **20–35 % of one core** (it used more than 60 %), with the
simulation process at 135–150 % on its own cores. Function calls on the GUI thread went from 14.8 million
to 1.7 million per 40 s.

**GPU.** The 3-D drawing already runs on the GPU (VTK OpenGL). What loaded the CPU was the Python work
around it, removed above. The plots stay on Qt's raster painter; they are cached now, so OpenGL plotting is
not needed.

## Step 6 impact physics (core/impact.py)

- **Impact point on the real shape**: the impactor's last step is ray-cast on the OSIRIS-REx shape
  model. This gives the site (latitude/longitude), the angle from the local vertical and the surface
  normal, averaged over 20 m (crater scale).
- **Momentum transfer**: Cheng et al. (2023) eq. 1 with the ejecta momentum along the surface normal.
  The planner keeps the head-on assumption (it cannot know the site). An oblique hit transfers less:
  |Δv| = (mU/M)√(β²cos²θ + sin²θ).
- **β measured by the tractor**: its navigation filter models the jump as Δv = c + βk, with c and k from
  the known impact geometry. The estimate stays within 3σ of the truth.
- **Spin change**: ΔL = r × MΔv, Δω = J⁻¹ΔL, with J from the shape model.
  - Bennu's rotation rate changes for the rest of the run (gravity field, capture site and display all
    follow).
  - The small tilt of the spin axis (a wobble) is reported, not simulated.
  - A 40 t, 11 km/s impactor changes Bennu's day by minutes; a DART-class impactor by seconds.
- **Ejecta**: launch speeds follow the point-source scaling M(>v) ~ v^(−3μ) with μ = 0.41 (Housen &
  Holsapple 2011), in a 45° cone around the normal, with the total mass set so that the escaping ejecta
  carry the recoil momentum.
  - About 800 tracers fly in the polyhedron gravity on a background thread, using the multi-threaded
    gravity batches, until they land (shape radius map) or escape.
  - The app reframes the Bennu view at the impact and slows the time warp for 6 h so the cloud can be
    watched.
- **Storms near Bennu** (found while testing the faster engine):
  - After safe mode, the tractor re-approaches whatever phase it was in.
  - The navigation filter rejects grazing LIDAR returns (more than 70° from the surface normal) and
    gates outliers with a χ² 99.9 % test, so it stays consistent through the drift.

The impactor's terminal guidance is unchanged from Step 4.

Step 6 validation (tests/test_step6_impact.py, figure `impact`):

- Cheng eq. 1:
  - MΔv = mU + p_ej exactly;
  - β acts only along the ejecta direction;
  - head-on reduces to βmU/M.
- Impact points lie on the surface (polyhedron containment 8 m inside/outside along the normal); misses
  are detected.
- Spin: no torque for a hit through the center; a prograde equatorial push shortens the day;
  J Δω = r × MΔv; the attitude is continuous at the change.
- Ejecta speeds follow the power law to 5 %; escaping ejecta carry exactly the recoil momentum; tracers
  land or escape (escape speed ≈ 20 cm/s).
- Live mission: the tractor's β estimate is within 3σ, Bennu's day changes by the computed amount, and
  the ejecta cloud is produced.
- DART: Δv_T = −2.70 mm/s gives −32.6 min with ΔP/P = 3Δv_T/v, against the measured −33.0 ± 1.0 min.

**Limits.**
- Bennu is a rigid body: reshaping or disruption of the rubble pile is not modeled. The 40 t, 11 km/s
  impactor deposits ~30 J/kg.
- The ejecta model is illustrative (one scaling exponent, no cohesion, no solar pressure on dust).
- The wobble after an off-axis hit is not simulated.

## Step 7 tow, trim and final verification (core/tow.py)

- **Deflection state**: Bennu's deviation from its nominal orbit, driven by the impact and by the
  tractor's gravity a = G m ρ/|ρ|³ (whenever it is near, at any phase).
  - Propagated with the STM as the epoch-referred deviation z = Φ⁻¹δx. z only changes when Bennu is
    pushed, so a query at any time costs one matrix product.
  - Impulses map exactly; the daily-step tow integral agrees with a 20,000-step reference to 1.5·10⁻⁵.
  - Bennu's position everywhere in the live run (spacecraft targets, impactor, display) includes it.
- **Steering the tow**: a push dv at time t moves the 2060 encounter by B(t)dv, with
  B = [ξ; ζ]Φ_rv(t_ca, t).
  - Towing along e(t) = B(t)ᵀw/|B(t)ᵀw| grows the miss about 30× more over 10 years than the best fixed
    inertial direction, which largely cancels over Bennu's 1.2-year orbits.
  - The planner now uses the same law, so its tow estimate rose from ~200 km to ~4,700 km for 30 t at
    485 m over 10 years.
- **Trim guidance** (closed outer loop, re-planned at least weekly on the latest deflection):
  - *Exercise scenario*: maximize the miss.
  - *Keyhole scenario*: aim at b_aim = 1.1 × required along the impact's direction.
    - Tow along the steepest descent Bᵀ(b_aim − b) for the time that minimizes the error along it, then
      park 5 km out (pull ~1 %).
    - The trim resumes if the error grows again.
    - A target the tow cannot reach (most of it along ξ) is approached as closely as possible instead of
      wasting thrust.
- **Years of towing in the app**: at time warps of 1 day/s or more, station-keeping is averaged.
  - The craft sits on its hold point (Step 5 showed the closed loop holds it to centimetres), with the
    hover thrust and propellant integrated daily.
  - At slower warps the full closed loop (LQR/MPC + EKF) runs.
  - The tow lasts `tow_years`, ending at most 0.5 yr before the encounter.
- **Final verification**: when the tow ends, Bennu is propagated twice with the full nonlinear force
  model (once with the recorded impact and tow, once without) to the 2060 encounter, on a background
  thread.
  - The tow history is merged into weekly blocks. Thousands of 1-hour pieces made the two runs take
    different steps, and Bennu's 2054 flyby amplified the difference (6.7 % → 0.07 %).

Typical results:

| Case | Result |
|---|---|
| Keyhole scenario, DART-class impactor (0.43 mm/s) | towed 1.4 years (0.41 mm/s) and parked; final shift 1,100 km; aim error 2 km; linear vs nonlinear 0.07 % |
| Exercise scenario, 40 t impactor | 10-year tow 2.72 mm/s; shift 25,300 km; linear vs nonlinear 1.0 % (a 25,000 km change through the 2054 flyby is mildly nonlinear) |

## The Sun around the Solar System barycenter (core/dynamics/solar_system.py)

The planets pull on the Sun as it pulls on them, so the Sun orbits the common center of mass:
Σ mᵢ rᵢ = 0, so r_sun = −Σ (mᵢ/m_sun) rᵢ.

- **Size of the motion**: the Sun's center wanders 0.02–1.97 solar radii from the barycenter (2026–2140).
- **Who pulls hardest** (largest share):

  | Planet | Pull on the Sun's position |
  |---|---|
  | Jupiter | 779,000 km |
  | Saturn | 430,000 km |
  | Neptune | 234,000 km |
  | Uranus | 131,000 km |
  | Earth | 460 km |
  | Pluto | 54 km |

- **Pluto**: the 9th body (a dwarf planet since 2006). Its ephemeris comes from JPL Horizons, and its
  system GM from Horizons' GM(Pluto) = 869.326 km³/s² × (1 + Charon/Pluto 0.122).
- **Our own N-body integration** of the Sun + 9 planets (Earth and Moon separate; compiled RHS, DOP853,
  2 s) reproduces JPL DE441's Sun to 31 km over 114 years, with energy conserved to 3·10⁻¹³. The
  center-of-mass relation matches to 116 km; the asteroids make up the rest.
- **Simulator dynamics**: these already contain this motion. Bennu and the spacecraft are integrated in
  the barycentric frame (or Sun-centered with the indirect terms) against the JPL Sun, which moves.
- **App**:
  - The Sun is now drawn at its barycentric position, around a barycenter marker. An optional ×30
    exaggeration makes the 0.009 au wobble visible.
  - A "Sun around the barycenter" tab plots its path (with the solar disk to scale) and the distance:
    JPL, our N-body, Jupiter only, and Jupiter + Saturn.
- **The Solar System's motion through the Milky Way (core/galaxy.py, the 🏠 button of the Solar System view).**
  - The barycenter moves at (12.9, 245.6, 7.78) km/s relative to the Galactic center: toward the center, along the
    Galaxy's rotation, toward the north Galactic pole. These are astropy's Galactocentric "v4.0" parameters, from
    Drimmel & Poggio 2018 (RNAAS 2, 210), GRAVITY Collaboration 2018 (A&A 615, L15) and Reid & Brunthaler 2004
    (ApJ 616, 872).
  - Rotated into our ICRF axes with astropy: **246 km/s (52 AU per year) toward Cygnus** (l = 87.0°, b = +1.8°;
    RA 313.0°, Dec +47.3°). The planets' orbital plane is tilted **60.2°** to the Galactic plane.
  - **🏠 side view** (click again to leave it): the camera looks toward the Galactic center with the north Galactic
    pole up, so the motion runs to the left. The orbits appear nearly edge-on, tilted about 60° from horizontal.
    A line marks a plane parallel to the Galactic plane through the barycenter, and an arrow the motion.
  - **Traced paths**: for the Sun and the planets, the JPL positions plus the barycenter's Galactic velocity times
    the time since now. Solid: the last 11.9 years (one orbit of Jupiter, less near the ephemerides' 2026 start);
    faint: the next 11.9 years.
  - At the true speed the paths are long, stretched helices: in one year Earth circles at 1 AU while the Sun moves
    52 AU, so on screen they look almost straight. **"🏠 side view: slow the Galactic motion ×1/50"** (Display,
    display only, stated in the caption) makes the turns visible, as in the popular "helix" pictures.
  - The background is the real sky (core/sky.py), rotated into the view's ecliptic axes, so the Milky Way lies
    along the Galactic plane, and the side view looks toward Sagittarius with Cygnus to the left. The
    constellation background (Display toggle) names them.

Step 7 validation (tests/test_step7_tow.py, figures `tow`, `sun_barycenter`):

- Deflection state = STM mapping (10⁻¹²); tow integral; vectorized queries.
- Along-track tow drift follows Clohessy–Wiltshire y = (4a/n²)(1 − cos nt) − (3/2)at² within 30 % in
  Bennu's eccentric orbit (the drift has the CW sign: pushed forward, Bennu falls behind).
- Trim reaches the aim point within 2 km and switches the tow off.
- Steered tow beats any fixed direction by more than 5×.
- Linear vs nonlinear agree to better than 1 % for small deflections.
- The Sun's barycentric motion:
  - range 0–2 solar radii;
  - center-of-mass relation within 200 km;
  - our N-body within 100 km of DE441;
  - energy conserved.
- Live keyhole mission end-to-end: impact, capture, trim tow to the aim point within 5 km, nonlinear
  check within 1 %.

## Mission goal: the 2135 keyhole and the 2182 impact

### The scenario (core/scenario.py, scripts/make_scenario.py)

- **Why a keyhole.** By 2135 only one thing about Bennu's orbit is uncertain: where along one line of the
  2135 B-plane it passes. The 2060 flyby stretches any small early difference about 4,000 times along the
  timing axis ζ. The sensitivity of the 2135 point to a push in 2026 is rank one: 1·10⁷ km per mm/s along
  that line and a few km per mm/s across it. A keyhole is a few-km window on that line whose post-flyby orbit
  returns to hit Earth. JPL's most likely Bennu impact is 2182-09-24, probability 3.7·10⁻⁴; the cumulative
  probability through 2300 is 5.7·10⁻⁴ (Farnocchia et al. 2021, Icarus 369:114594).
- **Finding it in our force model.**
  1. The line of variations: the dominant singular direction of B(t₀) = [ξ; ζ]Φ(t_ca, t₀).
  2. A scan of ±300,000 km along it, in 2,000 km steps: 301 full propagations from 60 days before the flyby
     to 2185, run in parallel processes and cached in `data/scenario_scan.json`. A zero crossing of the
     timing coordinate of a later approach is a resonant return.
  3. Each crossing passing within 4 Earth radii is refined with the secant method on the asymptotic B-plane
     of the return (geocentric hyperbola from 0.01 au).
  4. The refined crossing is an impact if |B| is below Earth's capture radius.
- **Result.** Three impact keyholes lie inside the Moon's distance:
  - 2175 (flyby 0.98 LD);
  - 2180 (0.88 LD, grazing);
  - **2182** (0.49 LD), the one used.

  The 2182 return at the neighbouring crossing misses by 22,000 km.
- **The scenario orbit.** Before t_pre (60 days before the flyby) it is the real nominal plus Φ(t, t₀)·c·v₁,
  a change of 3.4·10⁻⁶ m/s in 2026: within ~1–3 km of the real orbit until 2060. After t_pre it is the full
  nonlinear force model with the STM, through the keyhole flyby (2135-09-25 07:23, **186,945 km = 0.49 LD**)
  to the **Earth impact on 2182-09-24 20:29 UTC** (12.2 km/s). The integration is stopped at the surface,
  with 20 s steps inside 200,000 km so a grazing pass is not stepped over.

| Item | Value |
|---|---|
| Keyhole width along the line of variations | 3.2 km |
| Required shift to reach 1 LD (+ζ side; the −ζ side needs 3× more) | 218,900 km |
| Default DART target (flyby at 1.25 LD) | 318,500 km |

**Nonlinear checks never re-integrate the undeflected orbit from 2026.** A millimetre of integration
difference before 2060 moves the 2135 point by kilometres, more than the keyhole is wide. The check
propagates twice from just before the first push to t_pre, once with the pushes and once without, then adds
the *difference* to the stored scenario state.

### Planner (core/planner.py, core/dart.py)

- **B-plane.** The 2135 B-plane (modified target plane) has the miss vector b with |b| = flyby distance. The
  needed shift λ solves |b + λd| = R_goal along the line of variations d.
- **DART opportunities.** One per close pass inside 0.3 au, passes closer than 300 days merged: 15 passes
  2030–2128.
  - Each has a porkchop: launches from 1.5 yr before to 0.3 yr after the pass (on a 4-day grid anchored at the
    window's start, and not before the earliest launch + 40 days), flights of 30–700 days, departure speed
    √(v∞² + 2μ/r_p) at most the cut-off, impact at least 2 yr before the flyby, and an **impact speed of at least
    3 km/s** (`MIN_IMPACT_SPEED`).
  - The gain per kg (head-on) is d·B(t)βU/M. The arc needing the least mass is recommended, with masses for
    the planned β and for 90 % confidence.
- **Recommended masses** (linear, head-on) for the 318,500 km target at 12.5 km/s cut-off, β = 3.6:

  | Impact | Mass |
  |---|---|
  | 2030-06 | 511 kg |
  | 2037 | 600 kg |
  | 2043 | 780 kg |
  | 2054 | 570 kg |
  | 2059 | 2.5 t |
  | 2068 | 300 t |
  | 2080–2128 | 0.7–4,000 t |

  Before the 2060 flyby a DART-class impactor is enough; after it, hundreds of tonnes and more.
- **Speed and mass trade.** Each opportunity also gives the speed–mass trade for cut-offs of 11.5, 12.5, 14
  and 16 km/s. More departure energy opens faster arcs: 2030, 11.5 km/s → 1,450 kg; 16 km/s → 150 kg.
- **Point of contact.** Parallel rays along U are cast on the shape model within 60 % of the radius. Each
  hit is scored by d·B·Δv with Cheng's momentum transfer and the ejecta along the 20 m crater normal.
  - The best point can beat a head-on hit when U is poorly aligned with the push direction: the sloped
    surface throws the ejecta recoil along it.
  - For the recommended 2030 arc: −19° lat, +21° lon, 32° from the vertical, 2.67× head-on. The linear
    sizing with this gives 191 kg at 11.8 km/s.
  - About 12 ms per arc: the crater normals come from a k-d tree of the 400k facet centers, which finds the same
    facets as the full scan (bit-identical normals) about 50× faster (0.75 s per arc before).
- **Full-model sizing (core/refine.py).** For shifts this large the linear map undershoots the true effect. The
  full model gives more deflection than predicted:

  | Linear shift | Full model vs linear |
  |---|---|
  | 93,000 km | +5.6 % |
  | 270,000 km | +43 % |
  | 630,000 km | +120 % |

  So the recommended mass is refined in a background process. A secant on log(mass) runs over full nonlinear
  propagations, earlier DARTs included, with the run without pushes computed once; it needs 2–4 propagations,
  about 90 s. For the 2030 DART this gives **142 kg (191 kg linear)**, delivering 100.3 % of the target. The
  plan panel says which sizing it shows. The next-DART recommendation after a shortfall is refined the same way.
- **Orbit raising.** From the 400 km parking orbit:
  - burns at perigee of at most min(T/m · 8 min, 500 m/s), apogee kept below 300,000 km so geocentric
    two-body motion is a fair model;
  - then an injection at perigee to v_dep;
  - the total is the single-burn v_dep − v_c (Oberth effect kept);
  - each burn waits one period of the orbit before it, scheduled back from the departure;
  - propellant from the rocket equation, or the unlimited-fuel convention.

  The recommended DART: 7 raising burns over 12 days (apogee 8,900 → 300,000 km, perigee speed 7.67 →
  10.72 km/s), then +1.75 km/s to 12.48 km/s (v∞ 6.17 km/s): 293 kg of propellant.
- **Gravity tractor.** Launched after a successful DART. Same models as Steps 3–7. It steers the flyby along
  the line of variations to the DART aim distance; that is the only direction pushes can move it.
  - Its reach is large before 2060: 30 t at 484 m for 10 years delivers 2.7 mm/s, amplified to millions of km
    at the 2135 flyby. After 2060 it is thousands of km.
  - An **overshoot is held, not pulled back.** The tow's sensitivity is linearized about the scenario orbit,
    and pulling a large overshoot back through the 2060 flyby is too nonlinear to steer safely: a test run
    that tried it ended at 0.40 LD in the full model.

### Bennu hits Earth in 2182 (core/earth_impact.py)

The impact page (and `app/impact_show.py`) follows Bennu's last two days on the scenario orbit, which ends 60 s
above the surface (minute nodes in the last hour), flown on to the surface under Earth's gravity. A live run whose
orbit still hits uses its own full-model track instead. The numbers in the caption follow the Earth Impact Effects
Program (Collins, Melosh & Marcus 2005, *Meteoritics & Planetary Science* 40, 817–840), checked against the paper:

| Quantity | Value (scenario, no deflection) | From |
|---|---|---|
| Impact | 2182-09-24 20:29 UTC, 57.8° S 61.8° E, open ocean (read from the Earth texture), local solar time 00:56 | the track |
| Entry (100 km) | 12.68 km/s, 76° to the horizon | the track |
| Kinetic energy | 5.9·10¹⁸ J = 1,408 Mt TNT (Bennu 7.33·10¹⁰ kg, 484 m) | ½mv² |
| Breakup | at 72 km (strength 2·10⁴ Pa from Bennu's 1,230 kg/m³) | Eqs. 9–12 |
| Airburst | none: the pancake reaches the surface before it spreads 7×; 12.19 km/s at the surface | Eqs. 15–18, Eq. 17 integrated numerically |
| Transient cavity in the water | ~6.8 km | Eq. 21 with the water constant 1.365 |

- Collins et al. give a fireball radius (Eq. 32) only above ~15 km/s, where the impactor vaporizes; Bennu arrives
  at 12.7 km/s, so none is quoted. The flash on screen is a display effect.
- The plume is a display model: 1,500 particles thrown from the site in a 40° cone around the vertical at
  0.3–5 km/s, flown ballistically in Earth's gravity (no air) and left on the turning Earth where they land.
- The tsunami is not modelled.

### DART window markers (the distance plot)

Each marker on the "Earth–Bennu distance · DART opportunities" tab shows what **one DART as currently set** does
in that launch window (`Planner.window_capability`, computed with the plan in the planner worker).

- **The DART as set**: the mass on the slider (with "Use recommended mass" on, the recommended mass the slider
  shows), the max speed, β and the sizing. With "for 90 % confidence" sizing the shift is taken at
  β − 1.28σ, so green then means "reaches the target with 90 % confidence".
- **The shift** is what the Mission plan would give for that window: its arc (the largest head-on shift within
  the max speed), aimed at that arc's point of contact, mapped linearly (STM) to the 2135 B-plane. The
  selected window's marker matches the plan exactly.
- **Classes** (status colour plus symbol, so colour never carries the meaning alone):

  | Marker | Meaning |
  |---|---|
  | green ● | reaches the DART target: counted as a success by the verdict's own test (≥ 90 % of the target, `dart_tolerance`) |
  | yellow ◆ | pushes the 2135 flyby past the goal distance but falls short of the target (the verdict would ask for another DART) |
  | red ▼ | falls short of the goal |
  | red ✕ | no arc within the max speed |

- **Tooltip**: launch and impact dates, impact speed, the point-of-contact factor, the shift and the share of the
  target, the predicted 2135 flyby in LD, the masses needed (for the target, for a success, for the goal) and
  the chance of reaching the target over β ~ N(β, σ).
- **Legend**: a strip under the plot title names the DART the markers describe, the four classes, the selected
  window and the caveat: these are **linear (STM) planner estimates**, and the full-model refinement can differ
  by tens of percent for big shifts. With the full-model mass on the slider (142 kg for the 2030 DART), the selected
  window can show yellow: linearly that mass gives 74 % of the target, while the full model puts it at 100 %.
  The selected line says so.
- **Recomputed** whenever a setting changes: the plan job is debounced (150 ms) and runs in the planner worker,
  reusing its cached porkchops and each arc's ray-cast contact candidates (a new β only re-scores them). About
  0.2 s for all 15 windows the first time, about 1 ms afterwards; the GUI thread never waits.
- **Live run**: after a shortfall the markers show the remaining windows for the next DART (its recommended mass,
  or the slider's), with the β measured by ground tracking, starting from the deflection reached so far. Green
  then means the DARTs together reach the success threshold. As before, changing a setting during a run flies the
  mission again.
- **Show DART window markers** (a checkbox next to the plot's tab) hides the markers, the selection ring, the
  legend strip and the selected line together. It is display only: the selected window, the plan and the
  mission do not change.

**Finding: the arc choice ignores the point of contact.** The planner picks each window's arc by its head-on
shift; the point of contact then multiplies it by 1.0–7×, depending on the arc's geometry and on Bennu's
orientation at its arrival time. A slower arc can therefore beat the faster one the planner picks: for the 2030
DART (2031 pass) the 11.25 km/s arc is 2.28× head-on and the 11.75 km/s one 1.23×. So with the point of contact
a lower max speed can occasionally improve a window. The head-on shift, which picks the arc, never improves
(tested).
Choosing the arc with its point of contact (and the arrival hour) would remove this; it is not done here.

`uv run python -m app.main --smoke-windows` saves `outs/snapshots/app_windows_*.png`: the markers shown and
hidden, and a manual mass of 193 kg (the recommendation) and 1,990 kg (the 2031, 2037, 2043 and 2054 passes turn
green, the 2060 pass yellow).

### Live run (core/mission.py, core/live_runner.py)

**Start.** Both craft start in the parking orbit on today's date. Each leg is a command: `launch_dart` (any
number, one at a time) and `launch_tractor`. The planner keeps these gates in the UI; the core accepts any
order, and the Step 6 test uses that to launch the tractor first.

**DART leg:**
1. Parking orbit, then orbit raising: Kepler arcs with the burns as events; a solar storm slips a burn by
   one revolution.
2. Injection, hyperbolic departure to 231,000 km, heliocentric cruise with four TCMs.
3. The terminal ZEM law, aimed at the point of contact rotated to the predicted impact time. This is a small
   extension of the Step 4 law, made so the recommended point can be hit: a 15 m aim error in the
   2030 run.
4. Impact on the real shape.

**After the impact:**
- Ground orbit determination: a full nonlinear propagation of the deflected orbit to the 2135 flyby. Its
  offset from the linear prediction re-anchors every later prediction and the trim aim. For mission-size
  shifts the second-order part is tens of percent, about 3.6 % at 38,000 km.
- 30 days after the impact, the result:
  - **success**: at least 90 % of the target delivered;
  - **short**: less than that, so the UI computes the next DART for the remaining shift with the measured β;
  - **missed**.

**Tow and end of run:**
- After the tow the tractor's pull is no longer counted (it leaves).
- The verification runs after the tow, on request, or automatically 10 days before t_pre. The with/without
  pair restarts at the same times; their adaptive steps still differ by centimetres, which the 2060 flyby
  amplifies into a noise floor of ~150 km at 2135. A fixed common grid brought it to 0.8 km but took 10× longer.
  It flies the deflected orbit to 2185 and reports:
  - the 2135 flyby distance;
  - whether Bennu still hits Earth;
  - every later approach, with a warning for any inside the goal.
- Bennu's position after t_pre then comes from that track, and the clock runs to the 2182 date.

**Typical runs (seed-dependent: the true β is drawn from N(3.6, 0.7)):**

| Run | Result |
|---|---|
| Recommended 2030 DART, β 3.84 (linear sizing, 191 kg) | 123 % of the target (2135 flyby 1.43 LD), success; the tractor arrives Dec 2030, captures the boulder; the overshoot is held |
| Recommended 2030 DART, β 1.7 (a natural shortfall) | 42 % of the target, short; DART #2 recommended for the Feb 2031 pass (1.6 t at 10.8 km/s with the measured β), launched from the panel |
| Undersized DART (30 % of the recommendation) | short (34 %); DART #2 (758 kg, 2031 pass) brings the flyby to 1.21 LD |
| Full-model sizing of the 2030 DART | 142 kg instead of 191 kg linear; delivers 100.3 % of the target (4 propagations, 93 s) |

## Findings and fixes (2026-10-17)

- **"The spacecraft moves back and forth" at slow speeds** was a display sawtooth, not the guidance. The DART's
  cruise steps on its own 12-hour grid that ends past the mission clock, and so do the tractor's escape and
  cruise. The views drew the craft at the end of its step while Earth and Bennu were at the clock's time. At 10
  min/s the craft froze for about 70 s and then jumped half a day ahead. The craft are now drawn at the clock,
  interpolated (cubic Hermite) between their last two states; the physics grid is unchanged, so runs stay bit
  identical (tests/test_display_interp.py).
- **The tractor re-parked every week after the trim** (a 175° transfer around Bennu, then 5°, …). Two causes:
  1. with the aim reached the error was ~0, so its direction g/|g| was numerical noise that flipped from one
     update to the next, and each flip moved the parking point to the other side. The direction now changes
     only while towing and with an error above 100 m;
  2. parked 5 km out, the tractor still pulls about 1 %, enough to re-arm the tow every ~50 days. It now
     resumes only when the error is worth 3 days of towing.

  The tow also ends now: the boulder is released after 60 days of holding the trim, when `tow_years` is up,
  or on command (tests/test_trim_hold.py, tests/test_release.py).
- **A different launch date gave the same run.** "Launch DART" flew the window's best arc planned from today
  whatever the time slider said. The new "Launch no earlier than" date (bound to the slider) restricts the
  porkchop to later launches. The porkchop's launch grid is now anchored at the window's start, so a date that
  does not constrain the arc leaves the plan as it was (tests/test_earliest_launch.py).
- **Slow DART arcs.** Late in a window the best remaining arc could hit at 0.65–1.1 km/s: a 1.3 t DART for
  the 2030 window from April 2030, and a 0.8 km/s next-DART recommendation in a live run. β is measured (DART,
  6.1 km/s) and modelled for impacts of several km/s; below that it would be an extrapolation. The planner now
  requires 3 km/s at impact.
- **Review of the post-impact physics** (`apply_impact` → `Deflection.impulse` → outlook and verdict): Bennu's
  velocity change is M·Δv = m·U + m(β−1)(n̂·U)n̂, with n̂ the outward surface normal; head-on it reduces to βmU/M.
  The same impulse goes to the linear deflection state (exact through the STM) and to the full-model truth. The
  verdict uses the 2135 point re-anchored on the full-model prediction. No other defect found.
- **The tractor launch** looked unavailable while waiting for the verdict (a grey line under a disabled
  button, clicked twice in a session log). It now has the coloured status strip, is planned ahead of the
  verdict, and a late next-DART recommendation can no longer disable it (a race in `_on_recommendation`).
- **Colours**: Bennu's curve on the distance plot is now orange (as in the Solar System view); the gravity
  tractor is blue everywhere.

## Layout

```
core/           constants, time, Horizons/SBDB client, ephemeris interpolation, 3D assets, integrators, plot style
core/dynamics/  physics models (table above)
core/guidance/  Kepler/Lambert (single + multi-rev), Earth departure, low-thrust transcription, ZEM/ZEV
core/estimation/ cruise navigation filter (dynamic model compensation)
core/control/   DARE/LQR (lecture 3 code), MPC (cvxpy), thrust-cone projection
core/proximity/ proximity operations: ops (phases), ekf, sensors (LIDAR ray casting), capture (MuJoCo),
                scenarios (stand-alone LQR-vs-MPC and navigation runs)
core/impact.py  impact point, momentum transfer, spin change, ejecta cloud
core/tow.py     deflection state, gravity-tractor tow, B-plane trim guidance, nonlinear verification
core/dynamics/solar_system.py  the Sun around the barycenter: center of mass and our own N-body
core/parallel.py, core/live_runner.py   thread pool for compiled kernels; simulation process
core/planner.py, core/deflection.py   mission planner (DART opportunities, tractor, outcome), impact / tractor / B-plane physics
core/scenario.py, core/dart.py   keyhole scenario (virtual impactor, close approaches); DART orbit raising, point of contact
core/refine.py  full-model (nonlinear) sizing of a DART's mass, run in a background process
core/mission.py live N-body mission: spacecraft state machines, events, snapshots for the UI
core/sky.py     star catalogue and constellations (d3-celestial), B−V star colours
core/earth_impact.py  Bennu hits Earth: approach track, site, Collins et al. entry and crater numbers, plume
app/            main window (main.py), panels, plots, widgets, background workers, live-run thread
                (livesim.py), planner worker processes (compute.py), procedural DART model (models.py)
app/scenes/     the 3D views: solar (center), bennu, craft (chase camera), earth_moon (bottom tab),
                earth_impact (center, around an Earth impact); sky (the star layer behind every view)
app/impact_show.py  the 2182 impact as a stand-alone program
scripts/        fetch_data, fit_bennu, precompute, make_scenario, validate, snapshot
tests/     validation suite
data/      cached ephemerides + SBDB record (generated)
assets/    processed models (raw downloads in assets/raw/)
```