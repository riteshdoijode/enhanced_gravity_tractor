"""The Solar System's motion through the Milky Way (it orients the Solar System view's home button).

The barycenter's velocity relative to the Galactic center comes from astropy's Galactocentric frame, parameter
set "v4.0": v_sun = (12.9, 245.6, 7.78) km/s along (toward the Galactic center, the direction of the Galaxy's
rotation, the north Galactic pole), from Drimmel & Poggio 2018 (RNAAS 2, 210), GRAVITY Collaboration 2018
(A&A 615, L15) and Reid & Brunthaler 2004 (ApJ 616, 872); the Sun sits 20.8 pc above the plane (Bennett & Bovy
2019, MNRAS 482, 1417). Rotated into ICRF axes this is 246 km/s toward l = 87.0 deg, b = +1.8 deg (RA 313.0 deg,
Dec +47.3 deg, in Cygnus), almost along the Galaxy's rotation. The planets' orbital plane (the J2000 ecliptic)
is tilted 60.2 deg to the Galactic plane.
"""

from functools import lru_cache

import numpy as np

from core.guidance.kepler import ecliptic_pole_icrf


@lru_cache(maxsize=1)
def solar_motion():
    """The Solar System's motion around the Galactic center, in ICRF axes:
        forward     unit vector of the barycenter's velocity relative to the Galactic center
        up          unit vector of the north Galactic pole (IAU Galactic frame)
        center      unit vector toward the Galactic center (l = 0, b = 0)
        speed_kms, l_deg, b_deg, ra_deg, dec_deg, constellation   (of the direction of motion)
        ecliptic_tilt_deg   angle between the planets' orbital plane and the Galactic plane."""
    import astropy.units as u
    from astropy.coordinates import ICRS, Galactocentric, SkyCoord, galactocentric_frame_defaults, get_constellation

    with galactocentric_frame_defaults.set("v4.0"):
        frame = Galactocentric()
    p0, v0 = 0 * u.pc, 0 * u.km / u.s
    center = SkyCoord(x=p0, y=p0, z=p0, v_x=v0, v_y=v0, v_z=v0, frame=frame)
    # the Galactic center, at rest in its own frame, seen from the barycenter moves at -v_sun
    v = -center.transform_to(ICRS()).velocity.d_xyz.to(u.km / u.s).value
    forward = v / np.linalg.norm(v)
    sky = SkyCoord(x=forward[0], y=forward[1], z=forward[2], representation_type="cartesian", frame="icrs")
    sph = sky.represent_as("spherical")
    radec = SkyCoord(ra=sph.lon, dec=sph.lat, frame="icrs")
    gal = radec.galactic
    up = SkyCoord(l=0 * u.deg, b=90 * u.deg, frame="galactic").icrs.cartesian.xyz.value
    center_dir = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic").icrs.cartesian.xyz.value
    return {"forward": forward, "up": up / np.linalg.norm(up), "center": center_dir / np.linalg.norm(center_dir),
            "speed_kms": float(np.linalg.norm(v)),
            "l_deg": float(gal.l.deg), "b_deg": float(gal.b.deg), "ra_deg": float(radec.ra.deg),
            "dec_deg": float(radec.dec.deg), "constellation": str(get_constellation(radec)),
            "ecliptic_tilt_deg": float(np.degrees(np.arccos(np.clip(up @ ecliptic_pole_icrf(), -1, 1))))}
