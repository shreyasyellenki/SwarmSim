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
    obstacles: np.ndarray | None = None,
) -> dict[str, Any]:
    state = {
        "step": step,
        "coverage_pct": round(float(coverage_pct), 4),
        "grid": base64.b64encode(grid.astype(np.uint8).tobytes()).decode("ascii"),
        "agents": agents,
        "comm_links": comm_links,
    }
    if obstacles is not None:
        state["obstacles"] = base64.b64encode(obstacles.astype(np.uint8).tobytes()).decode("ascii")
    return state


def decode_grid(grid_b64: str, grid_size: int) -> np.ndarray:
    raw = base64.b64decode(grid_b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(grid_size, grid_size)
