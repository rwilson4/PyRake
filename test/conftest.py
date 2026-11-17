"""Pytest configuration for the PyRake suite.

Registers the ``slow`` marker. Without this pytest warns on every run that
the marker is unknown, and an unknown marker is also a silent typo: ``-m
"not slwo"`` would quietly deselect nothing.

Also arms Cvxium's solver-performance guard against ``perf_baseline.json``.

"""

import os
from pathlib import Path

import pytest

# Set before any test module imports cvxium, whose __init__ installs the guard when
# it sees this variable. setdefault, so a caller can point at another baseline or
# disable the guard with CVXIUM_PERF_BASELINE="".
os.environ.setdefault(
    "CVXIUM_PERF_BASELINE", str(Path(__file__).parent / "perf_baseline.json")
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: traces a real Pareto frontier, so it costs seconds rather "
        "than milliseconds. Deselect locally with -m 'not slow'; nothing "
        "passes -m in CI, so CI still runs these.",
    )
