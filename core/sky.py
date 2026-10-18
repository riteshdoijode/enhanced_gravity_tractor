"""The real star sky behind every view: stars to magnitude 8 with their colours, and the 88 constellations.

Data: d3-celestial (Olaf Frohn, BSD-3-Clause licence, https://github.com/ofrohn/d3-celestial), fetched by
`uv run python -m scripts.fetch_data --only sky` into assets/raw/ and processed here into assets/sky.npz:
  - stars.8.json: stars down to magnitude 8 from XHIP, An Extended Hipparcos Compilation (Anderson & Francis
    2012, VizieR V/137D): J2000 right ascension / declination, V magnitude, B-V colour index;
  - constellations.lines.json, constellations.json: the IAU constellation figures and name positions (the IAU
    constellation pages, with line modifications by the author of d3-celestial).
Positions are J2000 equatorial, i.e. ICRF to well under an arcsecond (far below a pixel). GeoJSON longitudes are
right ascensions folded into [-180, 180] degrees, which the unit vectors absorb.

Star colours: B-V -> effective temperature with Ballesteros (2012, EPL 97, 34008),
T = 4600 K (1 / (0.92 (B-V) + 1.7) + 1 / (0.92 (B-V) + 0.62)), then the colour of a black body at T: Planck's law
weighted with the CIE 1931 colour-matching functions (the analytic multi-lobe fit of Wyman, Sloan & Shirley 2013,
JCGT 2(2)), converted to sRGB (IEC 61966-2-1) and normalised to its brightest channel.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.constants import ASSET_DIR

SKY_URL = "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/"
SKY_FILES = ("stars.8.json", "constellations.lines.json", "constellations.json")
SKY_NPZ = ASSET_DIR / "sky.npz"
# The Milky Way background (the default sky): NASA/Goddard Space Flight Center Scientific Visualization Studio,
# "Deep Star Maps 2020" (SVS 4851; Gaia DR2: ESA/Gaia/DPAC; Hipparcos-2, Tycho-2): the whole sky in plate carree,
# ICRF right ascension / declination, centred on 0h with right ascension increasing to the left, north up; linear
# light. Converted once (fetch_data --only sky) to an sRGB JPEG.
MILKYWAY_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr"
MILKYWAY_RAW = "starmap_2020_8k.exr"
MILKYWAY_JPG = ASSET_DIR / "milkyway_8k.jpg"
MILKYWAY_GAIN = 3.0  # exposure for the screen: the Milky Way glow is 1-4 % of the brightest stars (clear, not washed out)


def process_milkyway(exr_path, out=MILKYWAY_JPG, gain=MILKYWAY_GAIN):
    """SVS star map (linear EXR) -> sRGB JPEG (8192 x 4096). Returns the image size."""
    import OpenEXR
    from PIL import Image

    px = OpenEXR.File(str(exr_path)).channels()["RGB"].pixels.astype(np.float32)
    x = np.clip(px * gain, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    Image.fromarray((srgb * 255 + 0.5).astype(np.uint8)).save(out, quality=92, optimize=True)
    return px.shape[1], px.shape[0]


@lru_cache(maxsize=2)
def load_milkyway(half=False):
    """The Milky Way background as uint8 (rows from the top = north), full or half resolution; None if missing."""
    if not MILKYWAY_JPG.exists():
        return None
    from PIL import Image

    img = Image.open(MILKYWAY_JPG).convert("RGB")
    if half:
        img = img.resize((img.width // 2, img.height // 2), Image.Resampling.BOX)
    return np.asarray(img)


def milkyway_pixel(ra_deg, dec_deg, shape):
    """(row from the top, column) of a sky direction in the Milky Way image (0h at the centre, RA to the left)."""
    h, w = shape[:2]
    col = int(((0.5 - np.asarray(ra_deg) / 360.0) % 1.0) * w) % w
    row = int(np.clip((90.0 - np.asarray(dec_deg)) / 180.0 * h, 0, h - 1))
    return row, col
BV_MISSING = 0.65  # stars without a B-V (52 of 41,411): a Sun-like colour


def unit_vectors(lon_deg, lat_deg):
    """ICRF unit vectors from right ascension (any 360-degree range) and declination in degrees."""
    a, d = np.radians(np.asarray(lon_deg, float)), np.radians(np.asarray(lat_deg, float))
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=-1)


def process_sky(raw_dir, out=SKY_NPZ):
    """d3-celestial GeoJSON (raw_dir) -> assets/sky.npz. Returns a short summary."""
    raw_dir = Path(raw_dir)
    stars = json.loads((raw_dir / "stars.8.json").read_text())["features"]
    lon = np.array([f["geometry"]["coordinates"][0] for f in stars], float)
    lat = np.array([f["geometry"]["coordinates"][1] for f in stars], float)
    mag = np.array([float(f["properties"]["mag"]) for f in stars], float)
    bv = np.array([float(f["properties"]["bv"]) if str(f["properties"]["bv"]).strip() else np.nan for f in stars])
    hip = np.array([int(f["id"]) for f in stars], np.int64)
    order = np.argsort(mag)  # brightest first
    seg, seg_con = [], []
    names = json.loads((raw_dir / "constellations.json").read_text())["features"]
    ids = sorted({f["id"] for f in names})
    for f in json.loads((raw_dir / "constellations.lines.json").read_text())["features"]:
        for poly in f["geometry"]["coordinates"]:
            p = unit_vectors(*np.asarray(poly, float).T)
            seg += [np.stack([p[:-1], p[1:]], axis=1)]
            seg_con += [np.full(len(p) - 1, ids.index(f["id"]))]
    lab_u = unit_vectors(*np.array([f["geometry"]["coordinates"] for f in names], float).T)
    np.savez_compressed(
        out, star_u=unit_vectors(lon, lat)[order].astype(np.float32), star_mag=mag[order].astype(np.float32),
        star_bv=bv[order].astype(np.float32), star_hip=hip[order], line_seg=np.concatenate(seg).astype(np.float32),
        line_con=np.concatenate(seg_con).astype(np.int16), con_ids=np.array(ids),
        label_u=lab_u.astype(np.float32), label_name=np.array([f["properties"]["name"] for f in names]),
        label_id=np.array([f["id"] for f in names]), label_rank=np.array([int(f["properties"]["rank"]) for f in names]))
    return {"stars": len(mag), "constellations": len(ids), "line segments": int(sum(len(s) for s in seg)),
            "labels": len(names), "brightest": float(mag.min()), "faintest": float(mag.max())}


@lru_cache(maxsize=1)
def load_sky():
    """The processed catalogue (arrays), with star colours (rgb, 0..1). None if assets/sky.npz is missing."""
    if not SKY_NPZ.exists():
        return None
    with np.load(SKY_NPZ) as z:
        sky = {k: z[k] for k in z.files}
    sky["star_rgb"] = bv_to_rgb(np.nan_to_num(sky["star_bv"], nan=BV_MISSING)).astype(np.float32)
    return sky


def constellations():
    """{id: {"names": [...], "label_u": (k, 3), "segments": (n, 2, 3)}} for the 88 constellations."""
    sky = load_sky()
    if sky is None:
        return {}
    out = {}
    for k, cid in enumerate(sky["con_ids"]):
        sel = sky["label_id"] == cid
        out[str(cid)] = {"names": [str(n) for n in sky["label_name"][sel]], "label_u": sky["label_u"][sel],
                         "segments": sky["line_seg"][sky["line_con"] == k]}
    return out


def bv_temperature(bv):
    """Effective temperature [K] from the B-V colour index (Ballesteros 2012)."""
    bv = np.clip(np.asarray(bv, float), -0.4, 2.0)
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def _lobe(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def _cmf(lam):
    """CIE 1931 2-degree colour-matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013); lam in nm."""
    x = 1.056 * _lobe(lam, 599.8, 37.9, 31.0) + 0.362 * _lobe(lam, 442.0, 16.0, 26.7) - 0.065 * _lobe(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _lobe(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _lobe(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def temperature_rgb(T):
    """sRGB colour (0..1, brightest channel 1) of a black body at temperature T [K]."""
    T = np.atleast_1d(np.asarray(T, float))
    lam = np.arange(380.0, 781.0, 5.0)
    c2 = 1.4387769e7  # second radiation constant hc/k [nm K]
    planck = lam[None, :] ** -5 / np.expm1(c2 / (lam[None, :] * T[:, None]))
    xyz = planck @ _cmf(lam).T
    rgb = np.clip(xyz @ XYZ_TO_SRGB.T, 0.0, None)
    rgb /= rgb.max(axis=1, keepdims=True)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)  # sRGB transfer function


def bv_to_rgb(bv):
    """Star colour from B-V (temperature via Ballesteros 2012, then the black-body colour), shape (n, 3)."""
    bv = np.asarray(bv, float)
    grid = np.linspace(-0.4, 2.0, 241)  # tabulated: 41,000 stars share 241 colours to 0.01 mag in B-V
    table = temperature_rgb(bv_temperature(grid))
    k = np.clip(np.round((np.clip(bv, -0.4, 2.0) + 0.4) / 0.01).astype(int), 0, len(grid) - 1)
    return table[k]
