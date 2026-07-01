"""Simulation state schema shared by eval, server, and visualizer."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


def build_sim_state(
    step: int,
    coverage_pct: float,
    grid: np.ndarray,
    agents: list[dict[str, Any]],
    comm_links: list[list[int]],
) -> dict[str, Any]:
    return {
        "step": step,
        "coverage_pct": round(float(coverage_pct), 4),
        "grid": base64.b64encode(grid.astype(np.uint8).tobytes()).decode("ascii"),
        "agents": agents,
        "comm_links": comm_links,
    }


def decode_grid(grid_b64: str, grid_size: int) -> np.ndarray:
    raw = base64.b64decode(grid_b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(grid_size, grid_size)
