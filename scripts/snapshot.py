"""Render the 3D views off-screen to PNG (checks the scenes without opening the app).

    uv run python -m scripts.snapshot --view earth_moon --date "2135-09-25 07:23"
    uv run python -m scripts.snapshot --all
"""

import argparse

import pyvista as pv

from app.scenes import BennuScene, CraftScene, EarthMoonScene, SolarScene
from core.constants import ROOT
from core.ephemeris import Ephemeris
from core.scenario import KeyholeScenario
from core.timeutil import iso_to_t, t_to_utc_str

VIEWS = {"solar": SolarScene, "bennu": BennuScene, "craft": CraftScene, "earth_moon": EarthMoonScene}
PRESETS = [
    ("solar", "2026-10-15 00:00"),
    ("earth_moon", "2135-09-25 07:23"),
    ("earth_moon", "2182-09-24 18:00"),
    ("bennu", "2026-10-15 00:00"),
    ("craft", "2026-10-15 00:00"),
]


def render(eph, sc, view, date, size=(1400, 900), out_dir=ROOT / "outs" / "snapshots"):
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=size)
    scene = VIEWS[view](pl, eph, sc) if view == "earth_moon" else VIEWS[view](pl, eph)
    t = iso_to_t(date)
    scene.update(t)
    pl.add_text(f"{view}   {t_to_utc_str(t)} UTC", font_size=11, color="white", position="lower_left")
    path = out_dir / f"{view}_{date[:10]}.png"
    pl.screenshot(str(path))
    pl.close()
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=list(VIEWS), default="solar")
    p.add_argument("--date", default="2026-10-15 00:00")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    sc = KeyholeScenario.load()
    eph = Ephemeris().with_bennu(sc.nominal)
    for view, date in (PRESETS if args.all else [(args.view, args.date)]):
        print(render(eph, sc, view, date))


if __name__ == "__main__":
    main()
