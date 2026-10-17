"""Precompute Bennu's nominal trajectory + state-transition matrix (2026-2140) -> data/bennu_nominal.npz.

    uv run python -m scripts.precompute
"""

import time

from core.dynamics.nominal import compute_nominal
from core.ephemeris import Ephemeris


def main():
    t = time.time()
    path = compute_nominal(Ephemeris())
    print(f"wrote {path} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
