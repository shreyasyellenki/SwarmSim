"""Procedural obstacle layout generation for swarm exploration."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors4(x: int, y: int, grid_size: int) -> list[tuple[int, int]]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            out.append((nx, ny))
    return out


def obstacles_to_array(obstacles: Iterable[tuple[int, int]], grid_size: int) -> np.ndarray:
    arr = np.zeros((grid_size, grid_size), dtype=bool)
    for x, y in obstacles:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            arr[x, y] = True
    return arr


def generate_scattered_obstacles(
    grid_size: int = 32,
    n_segments: int = 10,
    max_len: int = 4,
    min_segment_spacing: int = 2,
    rng: np.random.Generator | None = None,
) -> set[tuple[int, int]]:
    """Random short wall segments (Tier 1)."""
    rng = rng or np.random.default_rng()
    obstacles: set[tuple[int, int]] = set()
    margin = 2

    for _ in range(n_segments * 4):
        if len(obstacles) >= n_segments * max_len:
            break
        x = int(rng.integers(margin, grid_size - margin))
        y = int(rng.integers(margin, grid_size - margin))
        direction = tuple(rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)]))
        length = int(rng.integers(2, max_len + 1))
        segment: list[tuple[int, int]] = []
        ok = True
        for i in range(length):
            cx = x + direction[0] * i
            cy = y + direction[1] * i
            if not (margin <= cx < grid_size - margin and margin <= cy < grid_size - margin):
                ok = False
                break
            for ox, oy in obstacles:
                if _manhattan((cx, cy), (ox, oy)) < min_segment_spacing:
                    ok = False
                    break
            if not ok:
                break
            segment.append((cx, cy))
        if ok and len(segment) >= 2:
            obstacles.update(segment)

    return obstacles


def open_cells(obstacle_arr: np.ndarray) -> list[tuple[int, int]]:
    gs = obstacle_arr.shape[0]
    return [(x, y) for x in range(gs) for y in range(gs) if not obstacle_arr[x, y]]


def flood_fill_component(
    obstacle_arr: np.ndarray, start: tuple[int, int]
) -> set[tuple[int, int]]:
    gs = obstacle_arr.shape[0]
    if obstacle_arr[start]:
        return set()
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in _neighbors4(x, y, gs):
            if (nx, ny) in seen or obstacle_arr[nx, ny]:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return seen


def sample_spawn_positions(
    obstacle_arr: np.ndarray,
    n: int,
    min_spawn_dist: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    """Sample n open cells with minimum Manhattan spacing."""
    candidates = open_cells(obstacle_arr)
    if len(candidates) < n:
        return None
    rng.shuffle(candidates)
    positions: list[tuple[int, int]] = []
    for cell in candidates:
        if all(_manhattan(cell, q) >= min_spawn_dist for q in positions):
            positions.append(cell)
        if len(positions) == n:
            return positions
    return None


def validate_layout(
    obstacle_arr: np.ndarray,
    spawns: list[tuple[int, int]],
    min_reachable_fraction: float = 0.85,
) -> tuple[bool, float]:
    """Ensure spawns share one connected component and enough cells are reachable."""
    gs = obstacle_arr.shape[0]
    total_open = int((~obstacle_arr).sum())
    if total_open == 0:
        return False, 0.0

    component = flood_fill_component(obstacle_arr, spawns[0])
    if not all(s in component for s in spawns):
        return False, len(component) / total_open

    reachable_fraction = len(component) / total_open
    if reachable_fraction < min_reachable_fraction:
        return False, reachable_fraction
    return True, reachable_fraction


def generate_tier1_layout(
    grid_size: int,
    num_agents: int,
    n_segments: int = 10,
    max_segment_len: int = 4,
    min_segment_spacing: int = 2,
    min_spawn_dist: int = 6,
    min_reachable_fraction: float = 0.85,
    rng: np.random.Generator | None = None,
    max_attempts: int = 80,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """Generate scattered obstacles + spawns; retry until valid."""
    rng = rng or np.random.default_rng()
    last_fraction = 0.0
    for _ in range(max_attempts):
        obstacles = generate_scattered_obstacles(
            grid_size,
            n_segments=n_segments,
            max_len=max_segment_len,
            min_segment_spacing=min_segment_spacing,
            rng=rng,
        )
        obstacle_arr = obstacles_to_array(obstacles, grid_size)
        spawns = sample_spawn_positions(obstacle_arr, num_agents, min_spawn_dist, rng)
        if spawns is None:
            continue
        ok, frac = validate_layout(obstacle_arr, spawns, min_reachable_fraction)
        last_fraction = frac
        if ok:
            return obstacle_arr, spawns, frac
    raise RuntimeError(
        f"Failed to generate valid Tier-1 layout after {max_attempts} attempts "
        f"(last reachable fraction={last_fraction:.2f})"
    )
