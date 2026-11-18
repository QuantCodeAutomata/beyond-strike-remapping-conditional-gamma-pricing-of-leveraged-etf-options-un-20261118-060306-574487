"""Shared pytest fixtures and configuration for the test suite.

All PDE/Monte Carlo grids used by fixtures here are deliberately small
(n_x/n_i in the tens, steps_per_year ~ 50, n_paths in the low thousands) so
that the whole suite runs in seconds, not minutes, as instructed. They are
not meant to reproduce the paper's published numbers, only to exercise the
numerical code paths on fast, memory-light grids.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repository root (parent of tests/) is importable as `src`/`data`
# regardless of how pytest was invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dupire_surface import LocalVolSurface, build_surface  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: tests that call the live Massive/Dataloader REST API "
        "(require MASSIVE_API_KEY, skipped by default)",
    )


@pytest.fixture(scope="session")
def small_joint_law_grid() -> dict:
    """Small 2D grid configuration for src.joint_law_pde.solve_joint_law."""
    return {"n_x": 80, "n_i": 40, "steps_per_year": 50, "n_sigma": 6.0}


@pytest.fixture(scope="session")
def small_moment_grid() -> dict:
    """Small 1D grid configuration for src.moment_pde.solve_moments."""
    return {"n_x": 80, "steps_per_year": 50, "n_sigma": 6.0}


@pytest.fixture(scope="session")
def spx_surface() -> LocalVolSurface:
    return build_surface("SPX")


@pytest.fixture(scope="session")
def nky_surface() -> LocalVolSurface:
    return build_surface("NKY")
