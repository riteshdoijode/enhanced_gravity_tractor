# Hand-off: Bennu simulator fixes and features — DONE (2026-10-17)

Plan: `~/.cursor/plans/bennu_simulator_fixes_and_features_67e3c261.plan.md`. Everything in it is implemented,
tested and documented (README "Findings and fixes (2026-10-17)", done.md "Fixes and features"). Baseline: git tag
`checkpoint-2026-10-17`; the changes are in the commit after it.

| Plan § | Where |
|---|---|
| §1 display sawtooth | core/mission.py `state_at`, `_hermite`; tests/test_display_interp.py |
| §2 slow motion from 1,000 km | core/mission.py `SLOWMO_RANGE*`, `_range_crossing` |
| §3 re-parking loop, release / end of tow | core/tow.py `TrimGuidance`; core/proximity/ops.py `release`; core/mission.py `end_tow`; tests/test_trim_hold.py, test_release.py |
| §4 tractor status strip, plan ahead, blue window, warm start | app/main.py `_tractor_status`, `_tractor_planning`; core/planner.py `tractor_launch_window`; app/plots.py `set_tractor`; tests/test_tractor_window.py |
| §5 earliest launch | app/panels.py `launch_after`, `follow_slider`; core/planner.py anchored porkchop grid; tests/test_earliest_launch.py |
| §6 physics review, minimum impact speed | core/planner.py `MIN_IMPACT_SPEED`; README findings |
| §7 star sky, constellations | core/sky.py, app/scenes/sky.py, `scripts.fetch_data --only sky`; tests/test_sky.py |
| §8 Earth impact 2182 | core/earth_impact.py, app/scenes/earth_impact.py, app/impact_show.py, centre page 2; tests/test_earth_impact.py |

Smoke modes: `--smoke`, `--smoke-live`, `--smoke-prox`, `--smoke-fail`, `--smoke-windows`, `--smoke-impact` (new),
and `uv run python -m app.impact_show --smoke`.
