# References

Sources cited in this repository (the Bennu planetary-defense simulator) and in its design plan, [plan.md](plan.md).

How this list was built: I searched the whole repository (source code, docstrings, comments, tests, `README.md`, `done.md`, `plan.md`, data-file metadata) for every paper, book, dataset and tool the code names. Each entry says where the project uses it.

The **Check** column says how far each entry was checked:
- **Web** = title, authors, journal, volume, pages and DOI were checked against a web source on 2026-10-18.
- **Code** = the project names it, but I could not independently confirm all details. The details are as the project writes them.

---

## 1. Papers

### Deflection physics: kinetic impactor, gravity tractor

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 1 | Cheng, A. F., et al. (2023). Momentum transfer from the DART mission kinetic impact on asteroid Dimorphos. *Nature* 616, 457–460. doi:10.1038/s41586-023-05878-z | Momentum-transfer law M·Δv = m·U + m(β−1)(E·U)E (eq. 1); DART values (579.4 kg, 6.1447 km/s, β = 3.61, range 2.2–4.9); Δv_T = −2.70 mm/s | `core/deflection.py`, `core/impact.py`, `core/dart.py`, `core/constants.py` (`DART`), `core/proximity/ekf.py`, `core/proximity/ops.py`, `scripts/validate.py`, `tests/test_step3_planner.py`, `tests/test_step6_impact.py` | Web |
| 2 | Thomas, C. A., et al. (2023). Orbital period change of Dimorphos due to the DART kinetic impact. *Nature* 616, 448–451. doi:10.1038/s41586-023-05805-2. arXiv:2303.02077 | Measured period change of −33.0 ± 1.0 min (3σ), used as a validation target | `tests/test_step3_planner.py`, `plan.md` | Web |
| 3 | Lu, E. T., & Love, S. G. (2005). Gravitational tractor for towing asteroids. *Nature* 438, 177–178. doi:10.1038/438177a. arXiv:astro-ph/0509595 | Gravity-tractor hover: thrust canted outward by asin(R/d)+φ, a = Gm/d², the 20 t / 1 N worked example, plume-misses-surface constraint | `core/deflection.py`, `core/tow.py`, `core/control/prox.py`, `core/planner.py`, `tests/test_step3_planner.py` | Web |
| 4 | Mazanek, D. D., Reeves, D. M., Hopkins, J. B., Wade, D. W., Tantardini, M., & Shen, H. (2015). Enhanced Gravity Tractor Technique for Planetary Defense. *IAA Planetary Defense Conference*, Frascati, 13–17 April 2015, paper IAA-PDC-15-04-11. NASA NTRS 20150010968 | "Enhanced" tractor: spacecraft mass plus a collected boulder raises the gravitational pull. Basis of the boulder-capture step | `core/deflection.py`, `core/planner.py`, `core/proximity/capture.py` | Web (authors from NTRS record) |

### Guidance, navigation and control

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 5 | Guo, Y., Hawkins, M., & Wie, B. (2013). Applications of generalized zero-effort-miss/zero-effort-velocity feedback guidance algorithm. *Journal of Guidance, Control, and Dynamics* 36(3), 810–820 | ZEM/ZEV law a = 6·ZEM/t_go² − 2·ZEV/t_go for the tractor's rendezvous, approach and braking | `core/guidance/zemzev.py`, `core/guidance/rendezvous.py`, `core/proximity/ops.py` | Web |
| 6 | Clohessy, W. H., & Wiltshire, R. S. (1960). Terminal guidance system for satellite rendezvous. *Journal of the Aerospace Sciences* 27(9), 653–658 | Linear relative-motion model, used to check the along-track drift of the tractor tow | `core/tow.py`, `tests/test_step7_tow.py`, `scripts/validate.py` | Web (secondary sources only; the title wording varies between sources) |

### Bennu: orbit, mass, shape, spin

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 7 | Farnocchia, D., Chesley, S. R., et al. (2021). Ephemeris and hazard assessment for near-Earth asteroid (101955) Bennu based on OSIRIS-REx data. *Icarus* 369, 114594. doi:10.1016/j.icarus.2021.114594 | JPL orbit solution #118 behind the Horizons Bennu ephemeris; 2182-09-24 as the most likely impact date; impact probability about 1/2700 for that date and 5.7·10⁻⁴ cumulative through 2300 | `README.md`, `plan.md`, `done.md`, `core/constants.py`, `data/scenario_keyhole.json` | Web (the probabilities come from the project text and press reports, not from the paper's tables) |
| 8 | Chesley, S. R., et al. (2014). Orbit and bulk density of the OSIRIS-REx target Asteroid (101955) Bennu. *Icarus* 235, 5–22. doi:10.1016/j.icarus.2014.02.020. arXiv:1402.5573 | Published Yarkovsky parameter A2 = −4.618·10⁻¹⁴ au/d² (da/dt ≈ −284 m/yr), the reference for the project's own A2 fit | `core/dynamics/asteroid.py` (`LIT_A2`), `scripts/fit_bennu.py`, `scripts/validate.py`, `tests/test_step2_physics.py`, `data/bennu_nongrav_fit.json` | Web |
| 9 | Chesley, S. R., French, A. S., Davis, A. B., et al. (2020). Trajectory estimation for particles observed in the vicinity of (101955) Bennu. *Journal of Geophysical Research: Planets* 125(9), e2019JE006363. doi:10.1029/2019JE006363 | Cited for Bennu's GM = 4.8904 m³/s² | `core/constants.py` (`BENNU["gm"]`), `plan.md` | Web for the paper. **Caveat:** the search confirmed the paper exists but I could not confirm that this exact GM value is printed in it. The code comment says "SBDB, fetched 2026-10-15", so SBDB is the actual source of the number. |
| 10 | Daly, M. G., et al. (2020). Hemispherical differences in the shape and topography of asteroid (101955) Bennu. *Science Advances* 6(41), eabd3649. doi:10.1126/sciadv.abd3649 | Bulk density 1194 kg/m³, volume-equivalent diameter 484.44 m, extents 505 × 492 × 456.7 m | `core/constants.py` (`BENNU`), `plan.md` | Web for the paper. The numbers are as the project quotes them. |
| 11 | Hergenrother, C. W., et al. (2019). The operational environment and rotational acceleration of asteroid (101955) Bennu from OSIRIS-REx observations. *Nature Communications* 10, 1291. doi:10.1038/s41467-019-09213-x | Rotation period 4.296061 h | `core/constants.py` (`BENNU["rot_period_h"]`), `plan.md` | Web for the paper. The number is as the project quotes it. |
| 12 | Werner, R. A., & Scheeres, D. J. Exterior gravitation of a polyhedron derived and compared with harmonic and mascon gravitation representations of asteroid 4769 Castalia. *Celestial Mechanics and Dynamical Astronomy* 65(3), 313–344. doi:10.1007/BF00053511 | Closed-form gravity of a constant-density polyhedron: the 5,120-face Bennu physics mesh, the numba-compiled version, and the gravity-gradient used by the EKF and LQR | `core/dynamics/polyhedron.py`, `core/dynamics/fastgrav.py`, `plan.md` (E3) | Web. **Year:** the project writes 1997. Most sources, including the journal listing, give 1996 for this paper. |

### Ephemerides and Solar System

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 13 | Park, R. S., Folkner, W. M., Williams, J. G., & Boggs, D. H. (2021). The JPL Planetary and Lunar Ephemerides DE440 and DE441. *The Astronomical Journal* 161, 105. doi:10.3847/1538-3881/abd414 | Gravitational parameters of the Sun, planets and Moon; DE441 is the reference that the project's own N-body integration (Sun's motion, Bennu orbit) is compared against | `core/constants.py` (`GM`), `core/dynamics/solar_system.py`, `tests/test_step7_tow.py`, `app/plots.py` | Web |

### Earth impact (2182 scenario)

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 14 | Collins, G. S., Melosh, H. J., & Marcus, R. A. (2005). Earth Impact Effects Program: A web-based computer program for calculating the regional environmental consequences of a meteoroid impact on Earth. *Meteoritics & Planetary Science* 40(6), 817–840. doi:10.1111/j.1945-5100.2005.tb00157.x | Atmospheric entry (Eqs. 5, 8–12, 15–18), breakup altitude, pancake factor 7, transient crater (Eq. 21, 1.161 for rock and 1.365 for water). Eq. 32 (fireball) is deliberately not used: it applies only above about 15 km/s | `core/earth_impact.py`, `app/scenes/earth_impact.py`, `tests/test_earth_impact.py`, `README.md` | Web |
| 15 | Housen, K. R., & Holsapple, K. A. (2011). Ejecta from impact craters. *Icarus* 211, 856–875. doi:10.1016/j.icarus.2010.09.017 | Ejecta-velocity scaling M(>v) ∝ v^(−3μ), μ ≈ 0.41 for porous targets, used for the illustrative ejecta cloud after the DART impact | `core/impact.py` | Web |

### Boulders, the tractor's scan and the capture design range (added 2026-10-18)

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 23 | Burke, K. N., DellaGiustina, D. N., Bennett, C. A., Walsh, K. J., Pajola, M., Bierhaus, E. B., Nolan, M. C., et al. (2021). Particle size-frequency distributions of the OSIRIS-REx candidate sample sites on asteroid (101955) Bennu. *Remote Sensing* 13(7), 1315. doi:10.3390/rs13071315 | Power-law slopes −3.0 ± 0.2 … −2.3 ± 0.1 at the four candidate sites; the native boulders use −2.65, the middle of that range | `core/proximity/boulders.py` | Web (abstract) |
| 24 | Jewitt, D., Kim, Y., Li, J., & Mutchler, M. (2023). The Dimorphos boulder swarm. *The Astrophysical Journal Letters* 952, L12. doi:10.3847/2041-8213/ace1ec | Boulders ejected by the DART impact: differential size index q = −3.9 ± 1.5 (boulders > 4 m), velocity dispersion 0.30 m/s, close to the escape speed; used for the DART debris sizes (and the slow, falling-back fragments) | `core/proximity/boulders.py`, `core/impact.py` | Web (full text) |
| 25 | Lauretta, D. S., et al. (2017). OSIRIS-REx: Sample return from asteroid (101955) Bennu. *Space Science Reviews* 212, 925–984. doi:10.1007/s11214-017-0405-1 | OLA ranges to Bennu from beyond 7 km at 100 Hz (as close as 50 m at 10 kHz): the shot rate behind the scan's size accuracy | `core/proximity/boulders.py` | Web |
| 26 | Daly, M. G., et al. (2017). The OSIRIS-REx Laser Altimeter (OLA) investigation and instrument. *Space Science Reviews* 212, 899–924 | The laser altimeter the scan models | `core/proximity/boulders.py`, `core/proximity/sensors.py` | Web (title, journal, pages; I did not read the instrument specifications in the paper itself) |
| 27 | Gates, M., Stich, S., McDonald, M., Muirhead, B., Mazanek, D., Abell, P., & Lopez, P. (2015). The Asteroid Redirect Mission and sustainable human exploration. *Acta Astronautica* 111, 29–36. doi:10.1016/j.actaastro.2015.01.025 | The robotic boulder capture option: a boulder "typically 2–4 m"; the stones the arm can pick up | `core/proximity/boulders.py` | Web (bibliographic details; the 2–4 m wording is from summaries of the paper and of its IAC-14 version) |

Candidate sample-site coordinates (Nightingale 56°N 43°E, Kingfisher 11°N 56°E, Osprey 11°N 88°E, Sandpiper 47°S 322°E): NASA news release "NASA Mission Selects Final Four Site Candidates for Asteroid Sample Return" (2019-08-12) and J. Davis, "OSIRIS-REx Team Picks 4 Candidate Sample Sites on Asteroid Bennu", The Planetary Society (2019-08-15), https://www.planetary.org/articles/osiris-rex-4-sample-sites. Check: Web. Used in `core/proximity/boulders.py`.

Illustrative, not from a source: the 15 % (1σ) spread of the boulders' true density around the assumed 2,000 kg/m³, the 20° slope limit (set from MuJoCo captures: some 23–25° sites failed as the rock rolled away from the claws), and the 1.3 : 1 : 0.75 rock shape.

### Sky, stars and the Solar System's motion through the Galaxy (display)

| # | Reference | Used for | Where | Check |
|---|---|---|---|---|
| 16 | Anderson, E., & Francis, C. (2012). XHIP: An extended Hipparcos compilation. *Astronomy Letters* 38(5), 331–346. doi:10.1134/S1063773712050015. arXiv:1108.4971. VizieR V/137D | Source of the 41,411-star catalogue (to magnitude 8) via d3-celestial | `core/sky.py`, `README.md` | Web |
| 17 | Ballesteros, F. J. (2012). New insights into black bodies. *EPL* 97, 34008. doi:10.1209/0295-5075/97/34008. arXiv:1201.1809 | B−V colour index to effective temperature, for star colours | `core/sky.py`, `tests/test_sky.py` | Web |
| 18 | Wyman, C., Sloan, P.-P., & Shirley, P. (2013). Simple analytic approximations to the CIE XYZ color matching functions. *Journal of Computer Graphics Techniques* 2(2), 1–11 | Black-body temperature to sRGB colour | `core/sky.py` | Web |
| 19 | Drimmel, R., & Poggio, E. (2018). On the solar velocity. *Research Notes of the AAS* 2, 210 | Solar velocity relative to the Galactic centre (astropy `Galactocentric` "v4.0" parameters) | `core/galaxy.py`, `done.md`, `README.md` | Web (title and bibcode 2018RNAAS...2..210D) |
| 20 | GRAVITY Collaboration (2018). *Astronomy & Astrophysics* 615, L15 | Distance to the Galactic centre, part of the "v4.0" parameter set | `core/galaxy.py`, `README.md` | Code. The paper exists and is cited for the Sun–Galactic-centre distance. I did not confirm the title. |
| 21 | Reid, M. J., & Brunthaler, A. (2004). *The Astrophysical Journal* 616, 872 | Sgr A* proper motion, part of the "v4.0" parameter set | `core/galaxy.py`, `README.md` | Code. The web search did not return it. |
| 22 | Bennett, M., & Bovy, J. (2019). *Monthly Notices of the RAS* 482, 1417 | The Sun's height above the Galactic plane (20.8 pc) | `core/galaxy.py`, `README.md` | Web (appears in a reference list titled "Vertical waves in the solar neighbourhood in Gaia DR2"). I did not open the paper itself. |

---

## 2. Books and textbooks

Named in `plan.md` and in code comments. The project does not give editions or publishers, so the details below are the standard ones.

| Reference | Used for | Where |
|---|---|---|
| Curtis, H. D. *Orbital Mechanics for Engineering Students* | Universal-variable Kepler propagation (Alg. 3.4), Lambert's problem (Alg. 5.2), Example 5.2 (used as a test). The edition is not given in the project. | `core/guidance/kepler.py`, `tests/test_step3_planner.py` |
| Vallado, D. A. *Fundamentals of Astrodynamics and Applications* | Exponential atmosphere table (Table 8-4) for Earth-orbit drag. The edition is not given in the project. | `core/dynamics/perturbations.py` |
| Markley, F. L., & Crassidis, J. L. (2014). *Fundamentals of Spacecraft Attitude Determination and Control*. Springer | Quaternion attitude. Named in `plan.md` only. | `plan.md` |
| Zarchan, P. *Tactical and Strategic Missile Guidance* (AIAA) | Proportional navigation for the impactor's terminal guidance. Named in `plan.md` only. | `plan.md` (E7) |
| Battin, R. H. (title not given in the project; likely *An Introduction to the Mathematics and Methods of Astrodynamics*) | Velocity-to-be-gained steering for the first leg of the tractor rendezvous. The code comment says only "Battin". | `core/guidance/rendezvous.py` |

---

## 3. Named in the code without a full citation

Only the name appears in the project, so I did not invent a reference for these.

- **Öpik / Valsecchi**: B-plane axes (ζ = −Earth's heliocentric velocity projected on the plane, ξ = s × ζ) in `core/deflection.py`.
- **Farnocchia / Chesley A2 formulation**: the transverse Yarkovsky model a = A2 (1 au/r)² t̂ in `core/dynamics/perturbations.py`. The paper behind it is Chesley et al. (2014), entry 8.
- **Lecture 1 and lecture 3 of the course** (Prof. Benjamin Riviere, Autonomous Space Robotics, NYU, Fall 2026): N-body equations (L1 eqs. 1–2), Lagrange-form computed-torque control for the MuJoCo arm, LQR/DARE and Kalman/EKF (L3). `core/control/dare.py` is copied with attribution from the course repository (`fall-2026-space-robotics`, `lecture3/util/dare.py`). The course repository and lecture slides are not part of this repository.

---

## 4. Data sets and models

| Data | Source | Used for |
|---|---|---|
| Sun, planets, Moon, Pluto, Bennu and large-asteroid states, 2026–2185 | JPL Horizons API, https://ssd.jpl.nasa.gov/api/horizons.api (docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html). DE441, Bennu solution JPL#118 (Farnocchia et al. 2021) | `core/horizons.py`, `core/ephemeris.py`, `scripts/fetch_data.py` |
| Bennu physical parameters and Earth close approaches | JPL Small-Body Database API, https://ssd-api.jpl.nasa.gov/sbdb.api (fetched 2026-10-15) | `core/constants.py`, `data/sbdb_bennu.json` |
| Bennu shape model (3.37 M faces, 88 cm) and the 5 cm Nightingale sample-site patch | NASA SVS #5069, OSIRIS-REx OLA v20, https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/ | `core/assets.py`, `assets/bennu_*.npz`, `assets/tagsite.npz` |
| Bennu and OSIRIS-REx 3D display models (glTF/GLB) | NASA Science 3D Resources (https://assets.science.nasa.gov/content/dam/science/psd/solar/2023/09/b/Bennu_1_1.glb and `.../o/Osiris_Rex.glb`) | `scripts/fetch_data.py`, `assets/osiris_rex.glb` |
| Star catalogue and 88 constellation figures | d3-celestial by Olaf Frohn (BSD-3-Clause), https://github.com/ofrohn/d3-celestial. Stars from XHIP (entry 16); figures from the IAU constellation pages | `core/sky.py`, `assets/sky.npz` |
| Milky Way background (8192 × 4096) | NASA/GSFC Scientific Visualization Studio, "Deep Star Maps 2020" (SVS 4851), https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr. Built from Gaia DR2 (ESA/Gaia/DPAC), Hipparcos-2 and Tycho-2 | `core/sky.py`, `assets/milkyway_8k.jpg` |
| Earth, Moon and Sun textures | PyVista example datasets | `README.md` |

---

## 5. Software the physics depends on

These are tools, not papers. They are listed because the project's results rely on them (versions are in `pyproject.toml`). The project cites no papers for them.

- SciPy (`solve_ivp` with the DOP853 integrator, `solve_discrete_are`, `brentq`), NumPy, numba
- MuJoCo (rigid-body contact and arm dynamics for boulder capture)
- astropy (Galactocentric frame, time scales), trimesh, PyVista / PyVistaQt, PySide6, pyqtgraph, matplotlib
- cvxpy (MPC hover controller)

---

## Notes

1. **Not found in the project:** no `.bib` file, bibliography file or citation manager exists. Everything above comes from comments, docstrings and Markdown text.
2. **Already flagged in the project:** `done.md` says "cite only verified references". Entries 20 and 21 (GRAVITY 2018, Reid & Brunthaler 2004) and the GM value attributed to entry 9 are the ones I could not confirm. Check them before the report goes in.
3. **Werner & Scheeres** is dated 1997 in `plan.md` and the code. The journal listing gives 1996 for vol. 65(3). Use whichever year your report's style requires, but be consistent.
