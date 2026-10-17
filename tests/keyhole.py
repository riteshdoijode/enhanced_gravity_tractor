"""Shared setup for the keyhole-scenario tests: the scenario, ephemerides with its Bennu, and the planner."""

from functools import lru_cache


@lru_cache(maxsize=1)
def setup():
    from core.ephemeris import Ephemeris
    from core.planner import Planner
    from core.scenario import KeyholeScenario

    sc = KeyholeScenario.load()
    eph = Ephemeris().with_bennu(sc.nominal)
    return sc, eph, Planner(eph, sc.nominal, sc)
