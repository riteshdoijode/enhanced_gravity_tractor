"""Step 1 data fetch: JPL ephemerides, Bennu SBDB record, NASA 3D models, textures, the star sky.

    uv run python -m scripts.fetch_data            # everything (~155 MB download)
    uv run python -m scripts.fetch_data --skip-large   # skip the 77 MB TAG-site patch (Step 5)
    uv run python -m scripts.fetch_data --only sky     # stars and constellations (d3-celestial, 5.5 MB) -> assets/sky.npz
"""

import argparse
import json

import numpy as np
import requests
from astropy.time import Time
from tqdm import tqdm

from core import horizons
from core.assets import process_bennu, process_spacecraft, process_tagsite
from core.constants import (
    ASSET_DIR,
    BENNU_CLOSE_APPROACHES,
    BENNU_EPHEM_STOP,
    DATA_DIR,
    EPHEM_START,
    EPHEM_STOP,
    HORIZONS_BODIES,
)
from core.timeutil import jd_to_t

RAW = ASSET_DIR / "raw"
SVS = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/"
DOWNLOADS = {
    # NASA SVS #5069, OSIRIS-REx OLA global shape model v20, 88 cm resolution
    "bennu_global_v020.glb": SVS + "g_00880mm_alt_ptm_0000n00000_v020.glb",
    # NASA Science 3D Resources (VTAD): lightweight textured Bennu and OSIRIS-REx spacecraft
    "bennu_nasa3d.glb": "https://assets.science.nasa.gov/content/dam/science/psd/solar/2023/09/b/Bennu_1_1.glb",
    "osiris_rex.glb": "https://assets.science.nasa.gov/content/dam/science/psd/solar/2023/09/o/Osiris_Rex.glb",
}
LARGE_DOWNLOADS = {
    # 5 cm local model of the Nightingale TAG site (boulder-capture terrain, Step 5)
    "bennu_tagsite_v020.glb": SVS + "l_00050mm_alt_ptm_5595n04217_v020.glb",
}


def jd(iso):
    return Time(iso, scale="tdb").jd


def fetch_ephemerides(only=None):
    """All bodies, or only the named ones (added to the existing cache)."""
    out, centers = {}, {}
    if only:
        out = dict(np.load(DATA_DIR / "ephemeris.npz"))
        centers = json.loads((DATA_DIR / "ephemeris_meta.json").read_text())["centers"]
    start, stop = jd(EPHEM_START), jd(EPHEM_STOP)
    for body, (cmd, center) in HORIZONS_BODIES.items():
        if only and body not in only:
            continue
        if body == "moon":
            t_jd, r, v = horizons.vectors_chunked(cmd, center, start, stop, 0.25, "6 h")
        elif body == "bennu":
            t_jd, r, v = horizons.vectors_chunked(cmd, center, start, jd(BENNU_EPHEM_STOP), 1.0, "1 d")
            # hourly nodes within +-10 d of each Earth close approach (Earth bends the path in hours)
            for date, _, _ in BENNU_CLOSE_APPROACHES:
                c = jd(date)
                a, b = c - 10.0, min(c + 10.0, jd(BENNU_EPHEM_STOP))
                tj, rj, vj = horizons.vectors_chunked(cmd, center, a, b, 1 / 24, "1 h")
                t_jd, r, v = np.concatenate([t_jd, tj]), np.vstack([r, rj]), np.vstack([v, vj])
            t_jd, idx = np.unique(np.round(t_jd, 8), return_index=True)
            r, v = r[idx], v[idx]
        else:
            t_jd, r, v = horizons.vectors_chunked(cmd, center, start, stop, 1.0, "1 d")
        out[f"{body}_t"] = jd_to_t(t_jd)
        out[f"{body}_r"] = r * 1e3
        out[f"{body}_v"] = v * 1e3
        centers[body] = {"500@399": "earth", "500@10": "sun"}.get(center, "ssb")
        print(f"  {body:13s} {len(t_jd):7d} nodes")
    np.savez_compressed(DATA_DIR / "ephemeris.npz", **out)
    (DATA_DIR / "ephemeris_meta.json").write_text(
        json.dumps({"centers": centers, "frame": "ICRF", "units": "m, m/s, s past J2000 TDB",
                    "source": "JPL Horizons API", "fetched": Time.now().iso}, indent=2))


def fetch_check_points():
    """Off-node Horizons samples used by tests to measure interpolation error."""
    windows = {  # body -> (start, stop, step); steps chosen to fall between nodes
        "earth": ("2031-03-15", "2031-03-17", "37 m"),
        "jupiter_bary": ("2047-07-01", "2047-07-03", "37 m"),
        "moon": ("2029-06-01", "2029-06-02", "17 m"),
        "bennu": ("2033-01-10", "2033-01-12", "37 m"),
        "bennu_ca2135": ("2135-09-25 05:00", "2135-09-25 11:00", "7 m"),
    }
    out = {}
    for key, (s, e, step) in windows.items():
        body = key.split("_ca")[0]
        cmd, center = HORIZONS_BODIES[body]
        t_jd, r, v = horizons.vectors(cmd, center, s, e, step)
        out[f"{key}_t"], out[f"{key}_r"], out[f"{key}_v"] = jd_to_t(t_jd), r * 1e3, v * 1e3
    np.savez_compressed(DATA_DIR / "ephemeris_check.npz", **out)
    print(f"  {len(windows)} check windows")


def fetch_sbdb():
    rec = horizons.sbdb("101955")
    (DATA_DIR / "sbdb_bennu.json").write_text(json.dumps(rec, indent=2))
    print(f"  SBDB: {rec['object']['fullname']}, {len(rec.get('ca_data', []))} Earth approaches")


def download(name, url):
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {name}: cached")
        return dest
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"  {name}") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def cache_textures():
    from pyvista import examples
    from pyvista.examples import planets

    examples.load_globe_texture()
    planets.download_sun_surface(texture=True)
    planets.download_moon_surface(texture=True)
    print("  pyvista textures cached")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-large", action="store_true", help="skip the 77 MB TAG-site model")
    p.add_argument("--only", choices=["ephem", "sbdb", "assets", "sky"], default=None)
    p.add_argument("--bodies", nargs="*", default=None, help="with --only ephem: fetch just these bodies")
    args = p.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "ephem"):
        print("Ephemerides (JPL Horizons):")
        fetch_ephemerides(args.bodies)
        fetch_check_points()
    if args.only in (None, "ephem", "sbdb"):
        print("Bennu SBDB record:")
        fetch_sbdb()
    if args.only in (None, "assets"):
        print("3D models:")
        files = dict(DOWNLOADS)
        if not args.skip_large:
            files.update(LARGE_DOWNLOADS)
        paths = {name: download(name, url) for name, url in files.items()}
        cache_textures()
        print("Processing Bennu shape model:")
        props = process_bennu(paths["bennu_global_v020.glb"])
        print(json.dumps(props, indent=2))
        print("Processing spacecraft model:")
        print(json.dumps(process_spacecraft(paths["osiris_rex.glb"]), indent=2))
        if "bennu_tagsite_v020.glb" in paths:
            print("Processing the Nightingale TAG-site model (capture site):")
            print(json.dumps(process_tagsite(paths["bennu_tagsite_v020.glb"]), indent=2))
    if args.only in (None, "sky"):
        from core.sky import SKY_FILES, SKY_URL, process_sky

        print("Star sky (d3-celestial: XHIP stars to magnitude 8, IAU constellations):")
        for name in SKY_FILES:
            download(name, SKY_URL + name)
        print(json.dumps(process_sky(RAW), indent=2))


if __name__ == "__main__":
    main()
