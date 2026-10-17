"""Parallel test runs: `uv run pytest` spreads test files over 6 worker processes (pytest-xdist).
Each worker then uses one compute thread so 6 processes x numba threads do not oversubscribe the CPU."""

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("BENNU_THREADS", "1")
