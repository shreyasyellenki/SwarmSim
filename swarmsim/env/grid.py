"""Grid utilities for swarm exploration."""

from __future__ import annotations

import numpy as np


class ExplorationGrid:
    """Tracks explored cells on a 2D grid."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.explored = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.visit_count = np.zeros((grid_size, grid_size), dtype=np.int32)

    def reset(self) -> None:
        self.explored.fill(0)
        self.visit_count.fill(0)

    def world_to_cell(self, x: float, y: float, world_size: float) -> tuple[int, int]:
        half = world_size / 2.0
        cx = int(np.clip((x + half) / world_size * self.grid_size, 0, self.grid_size - 1))
        cy = int(np.clip((y + half) / world_size * self.grid_size, 0, self.grid_size - 1))
        return cx, cy

    def mark(self, x: float, y: float, world_size: float, agent_id: int) -> tuple[int, int, bool]:
        cx, cy = self.world_to_cell(x, y, world_size)
        self.visit_count[cx, cy] += 1
        is_new = self.explored[cx, cy] == 0
        if is_new:
            self.explored[cx, cy] = agent_id + 1
        return cx, cy, is_new

    def coverage(self) -> float:
        return float(np.count_nonzero(self.explored)) / float(self.grid_size * self.grid_size)

    def local_patch(self, x: float, y: float, world_size: float, window_k: int) -> np.ndarray:
        cx, cy = self.world_to_cell(x, y, world_size)
        half = window_k // 2
        patch = np.zeros((window_k, window_k), dtype=np.float32)
        for i in range(window_k):
            for j in range(window_k):
                gx = cx - half + i
                gy = cy - half + j
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    patch[i, j] = 1.0 if self.explored[gx, gy] > 0 else 0.0
        return patch.flatten()

    def downsample_mask(self, factor: int) -> np.ndarray:
        block = self.grid_size // factor
        mask = np.zeros((factor, factor), dtype=np.float32)
        for i in range(factor):
            for j in range(factor):
                region = self.explored[i * block : (i + 1) * block, j * block : (j + 1) * block]
                mask[i, j] = float(np.count_nonzero(region)) / float(region.size)
        return mask.flatten()

    def to_bytes(self) -> np.ndarray:
        return self.explored.astype(np.uint8)
