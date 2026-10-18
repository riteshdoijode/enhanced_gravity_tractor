# What we have done so far

A record of this chat session (October 15–18, 2026): the Bennu planetary-defense simulator for
Autonomous Space Robotics (NYU, Fall 2026, Prof. Benjamin Riviere). Steps 1–7 of 8 are complete. The mission
goal was then changed to the 2135 keyhole scenario (section 2), followed by two rounds of fixes and features
(October 16–17) and the preparation for sharing the project on GitHub (October 18). Step 8 remains. Project 1
is due 10/19.

For the details behind each item, see [README.md](README.md). For the sources behind the physics, see
[references.md](references.md). For the full conversation, which the course's AI-use policy asks to be exported
as a PDF, see the chat itself.

---

## 1. Goal and decisions

**What was asked for.** An interactive "mission control" simulator of a gravity tractor, built on the
course lectures:
- send actions and receive sensor values;
- validate the simulator against the math from class;
- load a 3D model and make an animation.

**Decisions made along the way:**

| Topic | Decision |
|---|---|
| Target | Asteroid **101955 Bennu**: OSIRIS-REx left accurate shape, gravity and spin data. |
| Mission | **Enhanced gravity tractor** (picks up a boulder) plus a **DART-style kinetic impactor** as the primary deflection. Both start in Earth orbit; the impactor is "assembled in orbit", so its mass and speed are free. |
| Goal | The **2135 keyhole** scenario: push Bennu's 2135 flyby out to at least the Earth–Moon distance, so the 2182-09-24 impact does not happen. One DART at a time, then the tractor. |
| Scope | All 8 steps in this project (nothing deferred to Project 2). |
| Hardware | Everything on the MacBook (M3 Pro, 12 cores). No Isaac Sim, no second machine. The movie may render slowly; it is not the main goal. |
| Interactivity | Live thrust and throttle, with unlimited fuel but consumption tracked; solar-storm delays and mid-flight outages; a bigger or faster impactor, showing the effect on Bennu's path. |
| Physics first | The clock runs at exactly the speed you choose, every frame is computed, and a changed parameter re-flies the run. Nothing is replayed or scripted. |
| Working style | Step by step. You say "start step N", and each step ends with a checkpoint summary. |

**Ground rules:**
- **Professor's repository:** `fall-2026-space-robotics/` is never modified. Code reused from it is copied with attribution.
- **References:** cite only verified ones ([references.md](references.md) marks which details were checked).
- **Git:** `project/bennu_defense/` is a git repository (tags `checkpoint-2026-10-17` and `checkpoint-2026-10-18`),
  with the GitHub remote `riteshdoijode/enhanced_gravity_tractor` and an Apache-2.0 `LICENSE`. Commit or push only when asked.

---

## 2. Step-by-step record

### Step 1: setup, real data, 3D models, app skeleton
- **Project:** a uv project with Python 3.12, PySide6, PyVista, pyqtgraph, MuJoCo, cvxpy, numba, astropy and others.
- **Downloaded data:**
  - JPL Horizons ephemerides for the Sun, planets, Moon and Bennu (later extended to 2185);
  - Bennu's record from the JPL Small-Body Database;
  - NASA's Bennu shape model (3.37 M faces) and the 5 cm Nightingale site model;
  - the OSIRIS-REx spacecraft model.
- **Processed assets:** a 400k-face display mesh, a watertight 5,120-face mesh for gravity, and mass properties.
- **App skeleton:** a Qt window with Solar System, Earth–Moon, Bennu and Spacecraft views, a date slider, and Bennu colored by surface gravity.
- **Checks:**
  - ephemeris interpolation within 55 m;
  - all 36 Earth close approaches in the Small-Body Database reproduced to under 1 minute and 1.5 km;
  - mass from shape model × density within 0.05% of the measured value.
- **Found:** JPL's barycentric Bennu states before about 2031 are offset by about 116 km, so Bennu is fetched Sun-centered.

### Step 2: core physics library and validation
- **Models:** N-body gravity (lecture 1, eqs 1–2), Earth J2, atmospheric drag, solar pressure with shadow, Yarkovsky force, relativity, quaternion attitude, polyhedron gravity, and the rotating-frame equations.
- **Bennu's orbit:** propagated nominally together with its state-transition matrix (STM), giving instant deflection mapping.
- **Fitted parameters:** Bennu's Yarkovsky and solar-pressure parameters were re-estimated against JPL (A2 = −4.54·10⁻¹⁴ against the published −4.62·10⁻¹⁴).
- **Checks:**
  - RK4 against the lecture's forward Euler;
  - J2 orbit drift and drag decay against theory;
  - the polyhedron gravity's built-in identities (Laplacian, Jacobi constant);
  - Bennu stays within JPL's own 3σ through 2135;
  - the STM matches nonlinear runs to 1.5·10⁻⁴ through the 2060 flyby.

### Step 3: instant mission planner wired to the sliders
- **Tractor:** a low-thrust escape spiral from Earth for any thrust, then 0–3-revolution Lambert transfers sized as finite burns.
- **Impactor:** a porkchop of about 38,000 Lambert arcs that picks the largest deflection within the launch Δv budget, preferably arriving after the tractor so the impact is watched.
- **Deflection:** Cheng et al. (2023) momentum transfer, the Lu & Love (2005) gravity tractor, and the B-plane of the Earth encounter.
- **App:** verdict, β-uncertainty cloud and success probability; tabs for the B-plane, porkchop, arrival-vs-thrust and distance; presets.
- **Checks:**
  - Curtis Lambert example;
  - Kepler propagation against RK4;
  - DART period change −32.6 min against the measured −33.0 ± 1.0 min;
  - Lu & Love's worked example.

### Step 4: live run
- **Tractor flight:** escape spiral, then heliocentric cruise in the full N-body model. Guidance is direct transcription (minimum-energy impulses, multi-start), flown as a receding horizon with ground re-plans.
- **Navigation:** a Kalman filter with dynamic model compensation on daily radio tracking fixes (a plain filter was overconfident: 69 km of error inside a 2 km bound).
- **Impactor:** four trajectory correction burns, then a terminal intercept law.
- **Live controls:** throttle changes, solar storms (thrusters off, keep or slip the schedule), time warp, event log and live plots.
- **Planning panel:** shows both the quick arrival estimate and the optimized flight plan that the live run actually flies.

### Step 5: proximity operations at Bennu
- **Phases:**
  1. approach (ZEM/ZEV guidance);
  2. stand-off 10 km out during the impact;
  3. survey hover;
  4. descent in Bennu's rotating frame inside a glide-slope cone;
  5. MuJoCo boulder capture;
  6. ascent;
  7. transfer to the tractor station.
- **Control:** LQR (the lecture's DARE code) and constrained MPC (cvxpy), switchable live. The constraints are the thrust limit, a plume cone and a keep-out zone.
- **Navigation:** an extended Kalman filter on raw LIDAR range (ray-cast on the shape model) and camera bearing.
- **Capture:** a 4-joint arm with computed-torque control and a microspine grip on the real Nightingale terrain, including local gravity and Coriolis forces. Mass properties merge after the grip.
- **Result:** MPC settled 2× faster than LQR at 2.6× lower cost and stayed inside the plume cone, where LQR needed clipping.
- **Thrust check:** if the thrust is too low to lift the boulder, the capture waits for the throttle to be raised.

### Step 6: multi-core execution and impact physics
**Multi-core (your request).** Numba kernels that release the GIL, a shared thread pool, the live simulation in its own process, and parallel tests and figures.

| What | Before → after |
|---|---|
| Optimized tractor flight plan | 4.7 s → 0.17 s |
| 2,000 Bennu gravity evaluations | 495 ms → 32 ms |
| Mission plan | 0.44 s → 0.20 s |
| Live run (launch → capture) | 85 s → 47 s |
| Test suite | 193 s → about 90 s |
| Report figures | about 5 min → 81 s |

**Impact physics:**
- **Impact point:** on the real shape model, with the crater-scale surface normal.
- **Momentum transfer:** ejecta thrown along the normal, so oblique hits move Bennu less.
- **β from the tractor:** the tractor's filter measures β from the impact.
- **Spin:** Bennu's spin rate changes (minutes for 40 t, seconds for a DART-class impactor).
- **Ejecta:** an ejecta cloud flown in Bennu's gravity on a background thread.
- **Validation:** a DART validation figure.

**Not added:** camera-based homing guidance for the impactor. It keeps its Step 4 terminal correction (since the
mission-goal change it aims at the recommended point of contact).

### Step 7: tow, trim and final verification, plus the Sun around the barycenter
- **Deflection state:** Bennu's deviation from the impact plus the tractor's gravity, propagated with the STM and used everywhere in the live run.
- **Steered tow:** towing along the best direction at each moment beats any fixed direction by about 30×. The planner uses the same law.
- **Trim guidance:** tow to an aim point along the steepest descent, then park 5 km out.
- **Years of towing:** fast-forwarded with averaged station-keeping.
- **Final check:** two nonlinear propagations of Bennu (with and without the pushes) through the encounter; in the keyhole mission this is the full-model verification to 2185.
- **The Sun around the barycenter (your request):**
  - Pluto added as the 9th body, from JPL Horizons;
  - our own Sun-plus-9-planets N-body model matches JPL's Sun to 31 km over 114 years;
  - the Sun now moves in the 3D view, with an optional ×30 exaggeration;
  - a new plot tab.

### Mission goal change: the 2135 keyhole and the 2182 impact (your request, October 16)

**New goal.** Assume Bennu's 2135 flyby, which passes between the Earth and the Moon, goes through the
gravitational keyhole, so Bennu hits Earth on a later return. Push that flyby out to at least the Earth–Moon
distance: the DART moves it by a chosen number of km, then the gravity tractor steers it to that aim point.

**Data and scenario:**
- **Ephemerides** extended to 2185; Bennu's nominal orbit and STM recomputed to 2185.
- **Finding the keyhole.**
  - The sensitivity of the 2135 flyby point to early pushes is rank one: 1·10⁷ km per mm/s along the line of
    variations, a few km per mm/s across it.
  - A scan of 301 full propagations along that line found the resonant returns. Refining them with the
    secant method gave three impact keyholes inside the Moon's distance: 2175, 2180 and 2182.
  - The **2182** one matches JPL's most likely date (Farnocchia et al. 2021, verified).
- **Scenario orbit.** The keyhole flyby is on 2135-09-25 at **186,945 km (0.49 LD)**, and the Earth impact
  on **2182-09-24 20:29 UTC**. The keyhole is 3.2 km wide. Before 2060 the scenario orbit differs from the
  real one by a few km (a 3.4·10⁻⁶ m/s change in 2026).

**Planner:**
- **Opportunities.** One DART opportunity per close pass of Bennu to Earth: 15 between 2030 and 2128. Each
  gets a recommended mass for the planned β and for 90 % confidence, the impact speed, and a speed–mass trade.
- **Early versus late.** Before the 2060 flyby the recommended DART is hundreds of kilograms (2030: about
  500 kg head-on). After 2060 it is hundreds to thousands of tonnes.
- **Orbit raising** from the parking orbit: burns at perigee up to the user's cut-off speed, apogee below the
  Moon, fuel per burn. The total is the single-burn Δv (Oberth effect kept).
- **Point of contact.** The aim point on the real shape, scored with the ejecta along the crater normal. On
  the recommended 2030 arc it gives 2.67× a head-on hit.
- **Full-model sizing.** The linear estimate (191 kg) is refined in the background with nonlinear
  propagations (secant, about 90 s) to **142 kg at 11.8 km/s**, which delivers 100.3 % of the target. Large
  deflections through the 2060 flyby grow faster than linear: +6 % at 93,000 km, +120 % at 630,000 km.
- **Minimum impact speed.** Arcs must hit at 3 km/s or more, because β is only known for impacts of several km/s.

**Live run:**
- **Launches.** It starts today with both spacecraft in orbit. The DART is launched first. The gravity
  tractor can launch only after a successful DART.
- **After each impact:**
  - ground orbit determination re-anchors the linear 2135 prediction with a full nonlinear propagation;
  - the clock pauses for the decision;
  - a short or missed DART gets a recommendation for the next one: velocity, mass and point of contact, at
    the remaining opportunities, using the measured β.
- **Terminal guidance** aims at the recommended point of contact. This is a small extension of the
  Step 4 law; the 2030 run hit within 15 m of the aim.
- **End of the run.** A verification flies the deflected orbit with the full force model to 2185. The clock
  then runs through the flyby to "Earth is safe" or the impact.

**UI.** The Solar System stays in the center. Bennu is top right. Below it is one spacecraft view (your
choice) that switches between the DART and the tractor, with a chase camera: Earth below in orbit, Bennu
growing during the DART's final approach. The info tabs sit under it. Bottom tabs: the distance plot with
clickable opportunity markers, the 2135 B-plane, and the Earth–Moon view of the keyhole flyby.

**Physics notes:**
- **The tractor is strong before 2060.** A 30 t tractor at 484 m for 10 years gives 2.7 mm/s, which the
  2060 flyby turns into millions of km at 2135. So the DART success rule is "at least 90 % of its target",
  not "the tractor can make up the rest".
- **Deflections of this size are nonlinear.** A 270,000 km shift through the 2060 flyby has a second-order
  part of tens of percent. The planner's numbers are linear estimates; the live run re-anchors them after
  each impact.

### Faster GUI: work moved off the GUI thread (your request, October 16)

**Problem.** You saw the front end use one core and lag. Profiling the GUI thread showed four causes:
- VTK drew about 5,000 frames per 40 s, mostly duplicates: pyvistaqt's timer, a new thread per render
  request on macOS, and a render after every actor change.
- Labels were re-created every frame.
- The 60,000-point distance plot was re-stroked whenever the time cursor moved.
- Planner jobs ran on GUI-process threads and held Python's interpreter lock.

**Fix:**
- Every draw is now one explicit Qt repaint.
- Labels and overlay text are kept as persistent actors and moved in place.
- Static plot items are cached as pixmaps.
- The side views, panels and plots refresh less often.
- Planner jobs run in three worker processes (`app/compute.py`), next to the simulation process and the
  refinement worker.

**Result.** The GUI process went from more than 60 % of one core to **20–35 %**. VTK renders fell from
5,169 to 929 per 40 s, and plot painting during playback from 20.5 s to 1.5 s. 3-D drawing was already on
the GPU (VTK OpenGL); the CPU load was the Python work around it.

### All 12 cores: core budget and parallel full-model runs (your request, October 16)

**Hardware.** The M3 Pro has a 12-core CPU (6 performance + 6 efficiency cores); the 18 cores are the GPU's.
macOS has no way to reserve cores for one program (no CPU affinity), so the simulator uses up to the whole
budget and asks for user-initiated scheduling priority, which steers it to the performance cores.

**What used few cores.** Each process sized its thread pool to the 6 performance cores, but the long jobs
were single-threaded Python: the full-model runs (scipy over a century of Bennu's orbit) behind the orbit
determination, the verification and the DART mass refinement ran one after another.

**Change:**
- `--cores N` (default: all cores) sets a budget that every process shares out (`core/parallel.py`):
  - simulation: 4 threads and 2 full-model processes;
  - planner workers: 3 × 2 threads;
  - refinement: 6 full-model processes;
  - GUI: 2 threads.
- `core/fullmodel.py`: a pool of full-model worker processes. The runs with and without the pushes now go
  side by side, and the flyby tail runs in a worker, which frees the simulation's interpreter.
- The mass refinement evaluates 5 candidate masses at once, interpolates, then checks ±1.5 % around the
  result. A newer request stops a running refinement at once.
- QoS class per thread; batch slicing balanced for mixed cores; workers exit if the app dies.

**Result:**
- Orbit determination: 35.4 s → 18.4 s, bit-identical.
- Refinement: 89 s → 39 s, or 20 s when an earlier result narrows the search. Same mass: 142 kg.
- The app peaked at about 11.4 of 12 cores in the live smoke run.
- No processes are left behind on exit.

### Launch in any window, clean relaunch, smoother impact (your request, October 16)

**What you saw, and why:**
- **Lag after the hit.** At high time warp one simulation step could cover 18 days, so the DART's terminal
  approach, the impact and the first two weeks after it fell inside a single step. The clock then froze for
  20–35 s while the ground orbit determination ran, with no ejecta on screen.
- **Rewind to the start.** 🚀 Launch DART always started the run in 2026, whatever window you picked.
- **Stuck second launch.** When "end" and the next "start" reached the simulation process together, the
  "start" was swallowed. The window then showed 2026 and waited forever. A late "ended" message from the
  old run could also mute the new one.

**Fix:**
- **Orbit determination and verification run in the background.** The clock waits only if it reaches the
  date their result is needed (the DART verdict, the flyby) before they finish.
- **Smoother ejecta.**
  - 20 s frames, interpolated between frames.
  - Drawn as fine sunlit dust; tracers that fall back are not drawn.
  - An impact flash, and a side-on view of the plume.
  - The Bennu and craft views draw every frame during the approach and the ejecta.
- **The run starts in the chosen window.** It starts 10 days before the DART's orbit raising. If the time
  slider is past that, the DART is re-planned from the slider time; if the window is over, a dialog offers
  the next one.
- **Runs are numbered.** A "start" that arrives during a run ends that run first. The window ignores
  snapshots from old runs. The old run's ejecta computation stops, and busy worker processes are replaced.

**Checked in the app:**
- About 17 snapshots/s and 1.5 ms of GUI work per snapshot through the approach and the ejecta.
- The clock waited about 0.5 s, at the verdict date.
- Relaunch flows: End then Launch at once; a later window chosen from an earlier slider time; slider past
  the window, then "Launch there". All smoke runs pass.

### Physics first: exact time warp, live Bennu orbit, re-simulation on any parameter (your request, October 16)

**Why.** You wanted the ground truth to be physics, not something replayed or looked up: every frame computed,
and reacting to the variables on screen.

**How it works now:**
- **Exact time warp.** The clock runs at the speed you choose. When the physics cannot keep up, the clock
  shows only what has been computed and the Live flight tab says so. While it waits for a computation, the 3-D
  views make way for a "Recalculating physics" page with progress.
- **Physics independent of the display.** Every craft steps on its own physical grid. Noise is drawn per
  craft, per control tick, or as a fixed function of mission time. The tow model follows mission time, not
  the warp. The same mission at two display speeds hits Bennu at the same instant and point, bit for bit (test).
- **Bennu integrated live (core/truth.py).** From the first push, Bennu is flown with the full force model in a
  background worker. Every impulse and the tractor's pull (30-day block means) are included.
  - The runs with and without the pushes share one solver, so their integration errors cancel. A vanishing
    push leaves Bennu in the 3 km keyhole (test); a DART-class push agrees with the independent nonlinear check
    to 0.03 %.
  - The 2135 flyby and the 2182 outcome are what the integration gives. The verdict waits for the full-model
    prediction (about 35 s).
- **Re-simulation.** Changing any parameter during a run flies the whole mission again from its start, with
  launches and storms replayed. The same parameters give the same run back, bit for bit (test). Picking
  another window restarts the run in that window.
- **Launch waits for the plan** of the selected window.

### Slow motion at Bennu and at Earth (your request, October 16; extended October 17)

**What it does.** Near Bennu the display slows down. This is the only departure from your chosen speed, and you asked for it:
- **DART:** the last 1,000 km at 5 s/s; the final approach (last 5 minutes) at 30 s/s; the last 10 seconds in real
  time; the flash and ejecta curtain at 2 s/s; the spreading cone at 1 min/s.
- **Tractor:** the last 1,000 km at 10 min/s; the descent at 5 min/s above 150 m and 1.5 min/s below; the
  capture at 15 s/s; the lift-off at 1 min/s.
- **Earth (2182 impact):** Bennu's last 1,000 km to the surface at 5 s/s, the plume at 1 min/s.

The display then returns to your speed.
- **On/off.** A checkbox under the speed list switches it off.
- **Display only.** The physics is unchanged: it is independent of the display speed.
- **Labelled.** A banner on the Bennu view and in the Live flight tab says what is being shown.
- **Skips.** A skip stops where a close-up begins.
- **Framing.** When the DART's close-up begins, the Bennu view turns side-on to the aim point. The incoming
  DART is drawn from 150 km out.

**Measured at 1 year/s** (before the 1,000 km stretch was added).
- DART: 10 s approach + 10 s real time + 20 s flash + 19 s ejecta.
- Tractor: 59 s descent + 44 s capture + 15 s lift-off.
- The physics kept up in every close-up.

### DART window markers: what this DART does in each window (your request, October 16)

**Change.** The markers on the distance plot show what the DART as currently set (mass, max speed, β, β
uncertainty, sizing) does in that window:
- **green ●** reaches the DART target, counted as a success by the verdict's own test (≥ 90 % of the target);
- **yellow ◆** pushes the 2135 flyby past the goal distance but falls short of the target;
- **red ▼** falls short of the goal; **red ✕** has no arc within the max speed.

**How the shift is computed.** The shift is what the Mission plan would give for that window: its arc, aimed at
that arc's point of contact, at the planned β (or at β − 1.28σ with the 90 % sizing). You chose this over the
literal "head-on gain per kg × mass": the recommended mass is sized at the point of contact (2.67× head-on for
the 2030 DART), so with head-on gains the recommended DART's own window, and nearly every other one, showed red.

**On the plot:**
- A legend strip under the title names the DART, the four classes (each with its own symbol), the selected
  window and the caveat: these are linear (STM) planner estimates, and the full-model refinement can differ by tens
  of percent for big shifts.
- The tooltip gives the shift, the predicted 2135 flyby in LD, the masses needed (for the target, for a success,
  for the goal) and the chance over β of reaching the target.
- **Show DART window markers** (next to the plot's tab) hides the markers, the ring, the legend and the selected
  line together, without changing the selection or the mission.
- In a live run, after a shortfall, the markers show the remaining windows for the next DART, with the measured β
  and the deflection so far.

**Speed.** The colours are computed in the planner worker with the plan (debounced 150 ms, porkchops cached).
- Each arc's ray-cast contact candidates are kept, so a new β only re-scores them.
- The crater normals come from a k-d tree of the 400k facet centers. They are bit-identical to the full scan
  and about 50× faster: 0.75 s → 12 ms per arc. This also speeds up the plan and the live impact.
- About 0.2 s for all 15 windows the first time, about 1 ms afterwards.

**Found.** The planner picks each window's arc by its head-on shift, but the point of contact multiplies that by
1.0–7× from arc to arc. So a lower max speed can occasionally improve a window: for the 2030 DART the 11.25 km/s arc is
2.28× head-on, the 11.75 km/s arc 1.23×. The speed test therefore checks the head-on shift, which picks the arc;
choosing the arc with its point of contact is a possible next step.

**Checked.** Three tests (below), and `--smoke-windows`:
- markers shown and hidden (selection and plan unchanged);
- 193 kg: the 2031 pass (impact 2030) green, the 2037 pass yellow, the rest red;
- 1,990 kg: the 2031, 2037, 2043 and 2054 passes green, the 2060 pass yellow.

### Side view: the Solar System moving through the Milky Way (your request, October 17)

**Change.** A 🏠 button on the Solar System view shows it from the side as it moves around the Galactic center,
as in the picture you shared, with the paths the Sun and planets trace. Click it again to return to the normal view.
- **The motion** comes from astropy's Galactocentric frame (v4.0 parameters): (12.9, 245.6, 7.78) km/s from
  Drimmel & Poggio 2018, GRAVITY Collaboration 2018 and Reid & Brunthaler 2004, rotated into our axes
  (core/galaxy.py). That is **246 km/s (52 AU per year) toward Cygnus** (l = 87.0°, b = +1.8°).
- **The view**: the camera looks toward the Galactic center, north Galactic pole up, so the Solar System moves to
  the left. Its orbits appear nearly edge-on, tilted 60° to the Galactic plane (60.2°). A line marks a plane
  parallel to the Galactic plane, and an arrow the motion.
- **The traced paths**: the JPL positions of the Sun and planets plus the Galactic motion since. Solid for the last
  11.9 years (one orbit of Jupiter), faint for the next 11.9; they follow the clock.
- **True speed vs helices.** At the true speed the paths are long, stretched helices and look almost straight: in a
  year Earth circles at 1 AU while the Sun moves 52 AU. "🏠 side view: slow the Galactic motion ×1/50" (Display)
  slows the motion for display only, and the caption says so; the turns then show, as in your picture.
- One test checks the direction against the Galactic components (U, V, W), the constellation, the Galactic center
  direction and the 60.2° tilt.

**Fixed on the way.** Rebuilding a 3-D scene (the Sun-wobble ×30 checkbox, Bennu's gravity colouring) left the old
labels and overlay text on screen, frozen, beside the new ones: they are now removed.

### The tractor could not launch after a successful DART (your report, October 17)

**What happened.** You saw that after a DART succeeded, the gravity tractor could not be launched. A session log
(below) of a scripted run that clicks the real buttons reproduced it, and showed two faults in the re-fly that any
setting change during a run triggers:
- **The re-fly changed the DART.** The run had flown the 191 kg DART (linear sizing). The full-model re-sizing
  (142 kg) finished after the launch, and a later change to a *tractor* setting re-planned DART #1 with it.
- **The re-fly stopped before the verdict.** The run pauses on each DART verdict for your decision. A change
  made then re-flew the mission up to that date, but the new verdict waits for ground tracking (a full-model
  prediction in background processes), and nothing processed it while the clock was paused. No verdict meant
  no tractor plan, and the button stayed disabled. A tractor launched at the verdict would also have been
  dropped by the re-fly ("no DART has succeeded yet").

**Fix:**
- A change to the tractor's settings keeps the DART(s) that flew. Only the DART's own settings re-plan it, and
  a recorded next DART is replayed unchanged under the same rule (`dart_params` in core/planner.py).
- The re-fly processes what is due at the instant it stops, and before each replayed command: a verdict there
  comes first, as in the first run (`LiveMission.settle`).
- A line under **🚀 Launch gravity tractor** always says why it can or cannot launch now: waiting for the
  verdict, the clock paused, the DART fell short, planning, cannot reach Bennu, or ready.
- The live smokes now press the real buttons (a disabled button ignores the click) instead of calling the
  handlers, which is why they had not caught this.

**Checked:**
- The same scripted user path (DART success, tractor thrust changed, Launch gravity tractor) now launches the
  tractor.
- A new test re-flies a run paused on a verdict, with the tractor launched there, using the app's background
  full-model processes. It fails without the fix.

**Session log (your suggestion).** Every session writes `outs/logs/session_*.jsonl`: each click (the widget, its
text, whether it was enabled), parameter changes, the launch buttons' state changes and what made them, what a
launch press did or why not, the commands to the simulation, verdicts and plans, the mission's events, a state
summary, and every error. Send me (or just tell me) the session's time and I can read it directly.

### Fixes and features (October 17, from your list)

Before any edit the project was put under git: commit "Checkpoint before fixes (2026-10-17)", tag
`checkpoint-2026-10-17` (`git diff checkpoint-2026-10-17` shows everything below).

**Fixed:**
- **"The spacecraft moves back and forth" at slow speeds.** It was the display, not the guidance. The DART and
  the tractor integrate on their own 12-hour grid, which ends past the mission clock. The views drew them at
  the end of their step while Earth and Bennu were drawn at the clock: at 10 min/s the craft froze for ~70 s,
  then jumped half a day. They are now drawn at the clock, interpolated between their last two states; the
  physics is unchanged.
- **The tractor re-parked every week after the trim.** Two causes:
  - with the aim reached, the trim direction was numerical noise that flipped each week, and each flip
    moved the parking point to the other side of Bennu;
  - parked 5 km out, the tractor's ~1 % pull re-armed the tow every ~50 days.

  Parked now stays parked.
- **No way to end the tow.** New **⤒ Release the boulder and end the tow** button. The boulder is left where
  the craft is; the craft holds 30 min, climbs to 50 km, burns 0.5 m/s and leaves. Bennu is then flown on with
  the full force model. The tow also ends by itself after `tow_years`, or once the trim has held for 60 days.
- **A different launch date gave the same run.** New **Launch no earlier than** date in the DART section,
  bound to the time slider (dragging the slider sets it; Play and Jump do not). The plan, the markers and
  Launch DART follow it. A date before a window's best arc leaves the plan exactly as it was: the porkchop's
  4-day grid is now anchored at the window's start.
- **Slow DART arcs.** A late launch in a window could pick a 0.65–1.1 km/s impact (a 1.3 t DART), and a live
  run had recommended a 0.8 km/s DART #2. Arcs must now hit at ≥ 3 km/s.
- **Tractor launch unclear.** A coloured status strip above the button: red (not available, and why), amber
  (waiting for the verdict, due date), green (READY: arrival, launch window open until …), grey (launched).
  - The tractor is planned at the DART's impact for the verdict date, so the button is ready the instant
    the verdict says success. It is re-planned for "launch now" every 30 mission days.
  - A next-DART recommendation arriving late could disable it; it no longer does.
  - Tractor-only setting changes before the launch re-plan the tractor instead of re-flying the mission.
- **The tractor flies the plan you saw.** At the hand-over its guidance starts from the ground's optimized
  profile. In the tests it arrives within half an hour of the planned date.

**New:**
- **Blue tractor launch window** on the distance plot: a band over the launch dates that still reach Bennu in
  time to close the gap, and the planned flight's Earth distance; Bennu's curve is now orange. The tractor's
  path in the Solar System view is blue too.
- **The real star sky** in every view: 41,411 stars to magnitude 8 at their J2000 positions, coloured by B−V
  (d3-celestial, BSD-3: XHIP stars, IAU constellation figures). The Milky Way lies along the Galactic plane.
  **Constellations** (Display) draws the 88 figures and names.
- **Milky Way background (your follow-up).** Every view, the main one included, defaults to NASA SVS's
  Deep Star Maps 2020 Milky Way (Gaia DR2, Hipparcos, Tycho-2; 8k in the main views, 4k in the small ones). The
  Display toggle "Constellation background" cross-fades over 3 s to the catalogue stars with the 88 figures and
  names, and back. The map's orientation was checked: the Galactic centre and Cygnus are bright, the Galactic
  poles dark. The toggle decides everywhere, the 🏠 view included.
- **All the planets (your follow-up).** The Solar System view also shows Saturn, Uranus, Neptune and Pluto
  at their JPL positions, with their orbits. Neptune's 165-year and Pluto's 248-year orbits are longer than the
  2026–2185 ephemerides, so the rest of each is closed with a two-body arc. The 🪐 button zooms out to the
  whole system and back.
- **Bennu hits Earth in 2182.** A centre page, shown in the last two days before the impact and the 20 minutes
  after (in planning mode, or a live run whose orbit still hits); `uv run python -m app.impact_show` plays it on
  its own.
  - It shows Bennu's track, the last 1,000 km at 5 s/s, a flash and a ballistic plume (display model), and the
    numbers from Collins, Melosh & Marcus (2005), checked against the paper: 57.8° S 61.8° E in the open southern
    Indian Ocean, at night; 12.68 km/s at 76°; 1,408 Mt; breakup at 72 km with no airburst; a ~6.8 km cavity in
    the water.
  - Their fireball formula applies only above 15 km/s, so none is quoted; the tsunami is not modelled.

**Checked:**
- All app smoke runs, each pressing the real buttons:
  - `--smoke` (planning);
  - `--smoke-live`;
  - `--smoke-prox`: DART #1 fell short, DART #2 succeeded, the tractor was ready the second the verdict came,
    then capture, trim, parked, released after 60 days, departed, 2135 flyby at 1.36 LD, "Earth is safe";
  - `--smoke-impact` (the impact page in planning playback).
- New tests: test_display_interp, test_trim_hold, test_release, test_tractor_window, test_earliest_launch,
  test_sky, test_earth_impact.

### Sharing the project on GitHub (October 18)

You wanted to share the project with a friend on a Windows PC so they can study it.
- **Repository.** `project/bennu_defense/` is the repository, pushed to
  `github.com/riteshdoijode/enhanced_gravity_tractor` (the GitHub-created `LICENSE`, Apache-2.0, was merged in).
  Checkpoint tag `checkpoint-2026-10-18`. The professor's repository and the course PDFs are not part of it.
- **Kept small.** `data/` (about 190 MB, regenerated), `assets/raw/` (downloads), `outs/` and `.venv` are
  ignored. The largest tracked files are the 15 MB display mesh and the 12 MB Milky Way image; a fresh clone is
  about 89 MB.
- **`.gitattributes`.** LF line endings for code and text on every system; the data files (`.npz`, `.npy`,
  `.glb`, `.vtp`, `.jpg`, `.png`, `.h5`) are treated as binary.
- **README.** The "macOS only" line now says the project was developed and tested on macOS, with a "Running on
  Windows or Linux" section: the macOS-only calls (thread quality-of-service, `sysctl`) fail quietly, workers use
  `spawn`, and the four setup commands recreate `data/`. Windows itself was not tested.
- **New document.** [references.md](references.md): the papers, books, data sets and software the project builds
  on (22 papers; 20 had their details checked online, and the entries that could not be confirmed are marked).
- **Checked.** A clone of the committed tree contains no absolute paths from this machine.

---

## 3. Key results

| Item | Result |
|---|---|
| Live tractor arrival (20 N) | within 0.3–40 km of Bennu at under 2 m/s; hand-over 30 km sunward |
| β measured by the tractor | 3.83 ± 0.01 (true 3.84); 5.02 ± 0.01 (true 5.03) |
| Boulder capture (20 t) | lifted 4.6 m, grip slip 4 cm |
| Station-keeping | within 0.1 m; navigation errors inside 3σ 99.6% of the time |
| Keyhole, DART-class impactor | tow trims the shift to 1,100 km, 2 km from the aim point; nonlinear check within 0.07% |
| Sun around the barycenter | 0.02–1.97 solar radii; Jupiter 779,000 km, Saturn 430,000 km, Pluto 54 km |
| Keyhole scenario | 2135 flyby 186,945 km (0.49 LD) → Earth impact 2182-09-24; keyhole 3.2 km wide |
| Recommended DART (2030 pass) | 142 kg at 11.8 km/s with the full model (191 kg linear); 7 raising burns + injection; aim point 2.67× head-on |
| Live 2030 DART (β 3.84) | hit 15 m from the aim; 123 % of the target, 2135 flyby 1.43 LD; tractor launched after it |
| Undersized DART (30 %) | short; the next-DART recommendation closes the remaining shift to 2 % |
| 2182 impact without a deflection | 12.68 km/s at 76°, 1,408 Mt, 57.8° S 61.8° E (open ocean), breakup at 72 km, ~6.8 km cavity |

---

## 4. Problems found and fixed

<details><summary>Expand the full list</summary>

- **Step 1**
  - JPL's barycentric Bennu states are offset 116 km before 2031: fetched Sun-centered instead.
  - Mesh decimation broke the gravity model: rebuilt with a radial remesh.
  - Labels were stuck at the origin, small bodies were invisible, camera framing was off: labels re-added each frame, screen-space markers.
  - A VTK call needed a Python bool, not a NumPy bool.
- **Step 2**
  - Adding the large asteroids made the Bennu fit worse: Yarkovsky and solar-pressure parameters re-estimated by least squares.
  - Geocentric and barycentric runs diverged because of interpolation error: tested as an exact identity instead.
- **Step 3**
  - Lambert edge case (infinite flight time at the top of the grid).
  - Kepler Newton iteration diverged: safeguarded Newton.
  - Single-revolution Lambert gave silly low-thrust Δv: multi-revolution Lambert.
  - The impactor choice was counter-intuitive: now maximizes the shift within the Δv budget.
  - A UI method shadowed Qt's `show()`; a log-axis plot bug.
- **Step 4**
  - Brake-start and ZEM/ZEV guidance failed for low thrust: replaced by direct transcription.
  - Low-thrust hand-over happened while still Earth-bound: now requires C3 ≥ 0.
  - Receding-horizon solves failed: relaxed tolerances and ground re-plans.
  - A storm could make the arrival *earlier*: the schedule is now kept or slipped.
  - A deferred correction burn stalled the UI.
  - The navigation filter was overconfident: added dynamic model compensation.
- **Step 5**
  - The grip failed: microspine anchoring plus a contact check.
  - Finger servo transient at start-up.
  - The MPC wasn't in cvxpy's fast-recompile form (DPP) and was badly conditioned: rewritten and scaled.
  - The impact came before the tractor arrived: the plan is now timed against the optimized arrival.
  - Stand-off view framing.
- **Step 6**
  - `Queue.qsize()` is unsupported on macOS: replaced by a shared counter.
  - MuJoCo 3.16 API change.
  - `math.fmod` is not available in numba.
  - Threads slowed down pure-Python code: compiled it first.
  - A storm near Bennu made the filter diverge (grazing LIDAR returns): added gating and re-approach logic.
  - Facet-normal jitter in the impact normal: crater-scale averaging.
  - Ejecta looked like stars: recolored.
- **Step 7**
  - The planner's fixed-direction tow was underestimated about 20×.
  - The minimum-norm trim wasted thrust on directions the encounter barely responds to: steepest descent.
  - The trim toggled on and off: deadbeat tow timing.
  - The nonlinear check had 6.7% integration noise: tow pieces merged, 0.07%.
- **Mission goal change**
  - Integrating the STM through a deep Earth approach failed ("step size below spacing"): the scenario carries
    the STM only through the 2135 flyby (every push happens before it).
  - A grazing Earth impact was stepped over: the integration switches to 20 s steps inside 200,000 km and
    changes mode at the crossing events (testing the distance on the boundary looped).
  - Closest-approach sampling put the 2182 return inside Earth when its true asymptote missed by 22,000 km:
    candidates are refined on the asymptotic B-plane.
  - Re-integrating the undeflected orbit from 2026 cannot reproduce a 3 km keyhole: checks add the nonlinear
    difference of the pushes at 60 days before the flyby instead.
  - The terminal guidance only applied the aim offset in the last 17 s (138 m miss): it now aims from the start
    of the terminal phase (15 m).
  - The whole orbit raising ran inside one 30-day step: craft events are now step boundaries.
  - The deflection history was copied on every query (quadratic slowdown over decades of towing): growing arrays.
  - The tractor's parked pull kept changing the orbit, so the pre-flyby verification repeated daily: after the
    tow its pull is no longer counted, and the pre-flyby check runs once.
  - The clock ran on past a recommended DART's launch while it was being planned: the run pauses at every
    DART verdict, and a stale plan is refused.
  - The tractor pulled a DART overshoot back with the scenario orbit's (linear) sensitivity, and the full model
    ended at 0.40 LD: an overshoot is now held, and the tractor only steers outward to the aim distance.
  - The nonlinear check had a noise floor of 143 km at 2135, from adaptive steps differing between the
    with/without runs. Aligned restarts keep it at ~150 km (0.04 % of the goal). A fixed common grid reached
    0.8 km but took 10× longer.
- **Tractor launch after a re-fly**
  - A tractor-setting change re-planned DART #1 with the newer full-model mass: the flown DART is now kept.
  - A re-fly ending at a paused verdict left that verdict unprocessed (no tractor plan, button disabled): the
    re-fly now settles what is due at its end and before each replayed command.
- **DART window markers**
  - Colouring by "head-on gain per kg × slider mass" turned the recommended DART's own window red: the
    recommended mass is sized at the point of contact. The markers now use each window's planned arc and point of
    contact (your choice).
  - The point-of-contact search took 0.75 s per arc, scanning all 400k facets for every crater normal: a k-d
    tree finds the same facets, 12 ms per arc.
  - The first legend boxes inside the plot covered markers and the curve after 2135: the legend is now a strip
    under the title, outside the data.
- **October 17 fixes** (display sawtooth, weekly re-parking, slow DART arcs, unclear tractor launch): see
  "Fixes and features" in section 2.

</details>

---

## 5. Validation and figures

**Tests: 131** (`uv run pytest`, 6 parallel workers, several minutes). TESTS_RESULT

| Test file | Count |
|---|---|
| [tests/test_step1_data.py](tests/test_step1_data.py) | 9 |
| [tests/test_step2_physics.py](tests/test_step2_physics.py) | 19 |
| [tests/test_step3_planner.py](tests/test_step3_planner.py) | 18 |
| [tests/test_step4_live.py](tests/test_step4_live.py) | 6 |
| [tests/test_step5_prox.py](tests/test_step5_prox.py) | 10 |
| [tests/test_step6_impact.py](tests/test_step6_impact.py) | 7 |
| [tests/test_step7_tow.py](tests/test_step7_tow.py) | 8 |
| [tests/test_keyhole_mission.py](tests/test_keyhole_mission.py) | 9 |
| [tests/test_parallel.py](tests/test_parallel.py) | 4 |
| [tests/test_live_flow.py](tests/test_live_flow.py) | 7 |
| [tests/test_truth.py](tests/test_truth.py) | 2 |
| [tests/test_dart_windows.py](tests/test_dart_windows.py) | 3 |
| [tests/test_display_interp.py](tests/test_display_interp.py) | 4 |
| [tests/test_trim_hold.py](tests/test_trim_hold.py) | 2 |
| [tests/test_release.py](tests/test_release.py) | 3 |
| [tests/test_tractor_window.py](tests/test_tractor_window.py) | 3 |
| [tests/test_earliest_launch.py](tests/test_earliest_launch.py) | 7 |
| [tests/test_sky.py](tests/test_sky.py) | 5 |
| [tests/test_earth_impact.py](tests/test_earth_impact.py) | 5 |

**Report figures** (`uv run python -m scripts.validate`, in `outs/validation/`):

| Figure | Shows |
|---|---|
| `integrators` | RK4 against forward Euler |
| `bennu_jpl` | our Bennu propagation against JPL |
| `stm` | linear deflection against nonlinear runs |
| `gravity` (`bennu_surface_gravity.png`) | Bennu's surface gravity map |
| `checks` | assorted physics checks |
| `prox_hover` | LQR against MPC acquiring the station |
| `prox_ops` | the proximity timeline |
| `capture` | the MuJoCo boulder capture |
| `impact` | DART check, impact geometry, spin, ejecta |
| `tow` | Clohessy–Wiltshire check, steered tow, B-plane trim, nonlinear check (in the 2135 B-plane) |
| `sun_barycenter` | the Sun's motion and each planet's share |
| `keyhole` | resonant returns along the line of variations; Earth distance to the 2182 impact |
| `opportunities` | recommended DART mass at every close pass; speed–mass trade |
| `orbit_raising` | the raising ellipses, perigee speed and fuel per burn |

**App smoke tests** (screenshots in `outs/snapshots/`; last run in full on October 17):

| Command | Covers |
|---|---|
| `--smoke` | planning |
| `--smoke-live` | DART orbit raising → impact → result → tractor launch and cruise |
| `--smoke-prox` | … through the tractor's proximity operations, tow, release and verification |
| `--smoke-fail` | undersized DART falls short → recommendation → DART #2 |
| `--smoke-windows` | DART window markers shown and hidden; before and after a heavier DART |
| `--smoke-impact` | the 2182 Earth-impact page in planning playback |
| `python -m app.impact_show --smoke` | the Earth-impact page on its own |

---

## 6. How to run

```bash
cd project/bennu_defense
uv sync                                   # Python 3.12 + dependencies
uv run python -m scripts.fetch_data       # JPL ephemerides, SBDB, NASA 3D models, star sky, Milky Way (~285 MB, ~4 min)
uv run python -m scripts.fit_bennu        # Bennu's Yarkovsky / solar-pressure parameters (~45 s)
uv run python -m scripts.precompute       # Bennu nominal orbit + STM, 2026–2185 (~100 s)
uv run python -m scripts.make_scenario    # the keyhole scenario (~2 min)
uv run python -m app.main                 # the simulator (all 12 cores; --cores N for fewer)
uv run pytest                             # validation suite
uv run python -m scripts.validate         # report figures
```

The first four steps recreate `data/`, which is not in the repository; they are needed once on a fresh clone
(see the README's "Running on Windows or Linux").

Live demo: keep the recommended DART (2030 pass), press **🚀 Launch DART** at 1 year/s and use **⏭ Skip to the
next event**. Watch the orbit raising in the spacecraft view, the terminal approach, the impact on Bennu and the
verdict. After a success press **🚀 Launch gravity tractor**. For the failure path, untick "Use recommended
mass", set about a third of it, launch, and after the shortfall press **🚀 Launch the next DART** (the
recommendation fills the controls). ⏩ **Verify and run to the end** shows the 2135 flyby and the 2182 outcome.

---

## 7. Known limits

- **Cruise guidance:** minimum-energy, not strictly minimum-time.
- **Attitude:** full 6-DOF attitude runs only in the capture; elsewhere it is assumed to follow its command.
- **Boulder and arm:** the boulder is synthetic, and there is a single arm.
- **Bennu as a rigid body:** reshaping or disruption is not modeled, and the wobble after an off-center hit is reported but not simulated.
- **Ejecta:** the model is illustrative.
- **Long tows:** fast-forwarded tows assume perfect station-keeping. The closed loop runs at slower time warps.
- **Keyhole scenario:** the keyhole is found in our force model (fewer perturbers than JPL), so its location
  differs from JPL's; the scenario is a hypothetical virtual impactor, not a prediction.
- **Linearity:** the planner's DART sizing is linear (STM); mission-size shifts through the 2060 flyby carry a
  second-order part of tens of percent, corrected in the live run after each impact (not in the plan).
- **Orbit raising:** geocentric two-body (Moon and Sun neglected below 300,000 km apogee); the parking-orbit plane
  is assumed chosen for the departure; the injection is a single impulsive burn.
- **Trim precision:** before 2060 one day of towing moves the 2135 flyby by thousands of km, so the trim lands
  within ~0.5–1 % of the aim shift, and only along the line of variations. An overshoot is held, not pulled
  back (the sensitivity is linearized about the scenario orbit).
- **Verification noise:** the full-model check of a deflection has a ~150 km noise floor at 2135 (fine for a
  1 LD goal, too coarse to resolve the 3 km keyhole itself).
- **Earth impact:** the numbers are the Earth Impact Effects Program's first-order scalings; the plume and the
  flash are display models; no tsunami. The site's ocean/land is read from the Earth texture.
- **Star sky:** positions at J2000, without proper motion (stars move up to ~10″/yr, a fraction of a pixel by
  2182 for all but a handful).
- **Platform:** developed and tested on macOS (Apple Silicon) only. Windows and Linux are untested.

## 8. Next: Step 8

- **Visual polish:** lighting, models, labels, scale indicators.
- **Movie:** record and replay to MP4, which covers the professor's animation requirement.
- **Report:** final figures.
- **Cleanup:**
  - remove the unused `core/guidance/zemzev.py` and `core/guidance/rendezvous.py` and the dev scripts in `outs/dev/`;
  - remove what is left of the dropped 2060 "Earth impact" exercise: the `"maximize"` trim mode in `core/tow.py`,
    `earth_capture_radius` in `core/deflection.py` (only one test uses it), and the README paragraphs and tables that
    still describe it (Step 3, Step 4 and Step 7 sections).
- **Command API:** an `env.command()` / `env.step(u) → y` interface for the "send actions, receive sensor values" requirement.
- **AI-use record:** export the chat as a PDF.
