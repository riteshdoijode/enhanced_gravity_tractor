"""3D views of the simulator. Each scene owns one PyVista plotter (Qt widget or off-screen):

    SolarScene     Solar System, AU, ecliptic J2000 axes, origin = Solar System Barycenter (always in the center)
    BennuScene     Bennu at true scale, m, ICRF axes, origin = Bennu's center of mass (always top right)
    CraftScene     chase camera on the DART or the gravity tractor, craft-centered metres (below Bennu)
    EarthMoonScene Earth-Moon neighbourhood, km, origin = Earth (bottom tab: the 2135 keyhole flyby)
"""

from app.scenes.bennu import BennuScene
from app.scenes.craft import CraftScene
from app.scenes.earth_moon import EarthMoonScene
from app.scenes.solar import SolarScene

__all__ = ["SolarScene", "BennuScene", "CraftScene", "EarthMoonScene"]
