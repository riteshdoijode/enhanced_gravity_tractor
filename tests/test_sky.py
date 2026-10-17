"""The star sky (core/sky.py, assets/sky.npz from d3-celestial): catalogue, positions, colours, constellations."""

import numpy as np
import pytest

from core.sky import bv_temperature, bv_to_rgb, constellations, load_sky


@pytest.fixture(scope="module")
def sky():
    s = load_sky()
    if s is None:
        pytest.skip("assets/sky.npz missing: uv run python -m scripts.fetch_data --only sky")
    return s


def test_catalogue(sky):
    u = sky["star_u"].astype(float)
    assert len(u) > 40_000
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)
    assert np.all(np.diff(sky["star_mag"]) >= 0)  # brightest first
    assert sky["star_mag"][0] == pytest.approx(-1.44, abs=0.05)  # Sirius
    assert sky["star_rgb"].shape == (len(u), 3) and sky["star_rgb"].min() >= 0 and sky["star_rgb"].max() <= 1


def test_known_stars(sky):
    """Polaris (HIP 11767) within 1 degree of the celestial pole; Sirius (HIP 32349) at RA 6h45m, Dec -16.7 deg."""
    def star(hip):
        return sky["star_u"][np.where(sky["star_hip"] == hip)[0][0]].astype(float)

    assert np.degrees(np.arccos(star(11767)[2])) < 1.0
    s = star(32349)
    assert np.degrees(np.arctan2(s[1], s[0])) % 360 == pytest.approx(101.29, abs=0.05)
    assert np.degrees(np.arcsin(s[2])) == pytest.approx(-16.72, abs=0.05)


def test_constellations(sky):
    c = constellations()
    assert len(c) == 88
    assert c["Ori"]["names"] == ["Orion"] and len(c["Ser"]["names"]) == 2  # Serpens Caput and Cauda
    seg = np.concatenate([v["segments"] for v in c.values()]).astype(float)
    assert np.allclose(np.linalg.norm(seg, axis=2), 1.0, atol=1e-6)
    # the figures join stars: a few segments end on the bright stars of Orion (Betelgeuse, HIP 27989)
    bet = sky["star_u"][np.where(sky["star_hip"] == 27989)[0][0]].astype(float)
    assert np.min(np.linalg.norm(c["Ori"]["segments"].astype(float).reshape(-1, 3) - bet, axis=1)) < 1e-3


def test_star_colours():
    """B-V -> temperature (Ballesteros 2012) -> black-body sRGB: the Sun (B-V 0.65) near 5,780 K and nearly white,
    hot stars blue-white, cool stars orange."""
    assert bv_temperature(0.65) == pytest.approx(5780, abs=30)
    sun, hot, cool = bv_to_rgb([0.65, -0.3, 1.6])
    assert sun.min() > 0.85
    assert hot[2] == pytest.approx(1.0) and hot[0] < 0.8
    assert cool[0] == pytest.approx(1.0) and cool[2] < 0.65
