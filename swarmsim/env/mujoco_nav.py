"""Stage 1: single-agent MuJoCo waypoint navigation environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class MuJoCoNavEnv(gym.Env):
    """Quadrotor navigates to a random 3D waypoint using four motor thrusts."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or load_config()
        self.stage_cfg = self.cfg["stage1_mujoco"]
        self.render_mode = render_mode

        xml_rel = self.stage_cfg["xml_path"]
        xml_path = Path(__file__).resolve().parent / Path(xml_rel).name
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.episode_horizon = self.stage_cfg["episode_horizon"]
        self.success_radius = self.stage_cfg["success_radius"]
        drone_body_id = self.model.body("drone").id
        mass = float(self.model.body_mass[drone_body_id])
        self.hover_thrust = self.model.opt.gravity[2] * -1.0 * mass / 4.0

        obs_dim = self.stage_cfg["obs_dim"]
        action_dim = self.stage_cfg["action_dim"]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self.waypoint = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self._viewer = None

    def _sample_waypoint(self) -> np.ndarray:
        wr = self.stage_cfg["waypoint_range"]
        return np.array(
            [
                self.np_random.uniform(wr["x"][0], wr["x"][1]),
                self.np_random.uniform(wr["y"][0], wr["y"][1]),
                self.np_random.uniform(wr["z"][0], wr["z"][1]),
            ],
            dtype=np.float64,
        )

    def _sample_spawn(self) -> np.ndarray:
        sr = self.stage_cfg["spawn_range"]
        return np.array(
            [
                self.np_random.uniform(sr["x"][0], sr["x"][1]),
                self.np_random.uniform(sr["y"][0], sr["y"][1]),
                self.np_random.uniform(sr["z"][0], sr["z"][1]),
            ],
            dtype=np.float64,
        )

    def _set_waypoint_marker(self) -> None:
        self.model.site_pos[self.model.site("waypoint_marker").id] = self.waypoint

    def _get_obs(self) -> np.ndarray:
        pos = self.data.qpos[0:3].copy()
        vel = self.data.qvel[0:3].copy()
        rel = self.waypoint - pos
        dist = np.linalg.norm(rel)
        height = pos[2]
        yaw_rate = self.data.qvel[5]
        return np.array(
            [rel[0], rel[1], rel[2], vel[0], vel[1], vel[2], height, dist, yaw_rate],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        spawn = self._sample_spawn()
        self.data.qpos[0:3] = spawn
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0

        self.waypoint = self._sample_waypoint()
        self._set_waypoint_marker()
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        thrusts = self.hover_thrust * (1.0 + 0.5 * action)
        self.data.ctrl[:] = np.clip(thrusts, 0.0, 12.0)

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        pos = self.data.qpos[0:3]
        dist = float(np.linalg.norm(self.waypoint - pos))
        reached = dist < self.success_radius
        crashed = pos[2] < 0.05

        rw = self.stage_cfg["reward"]
        reward = -rw["distance_scale"] * dist + rw["alive_bonus"]
        if reached:
            reward += rw["success_bonus"]
        if crashed:
            reward -= rw["crash_penalty"]

        terminated = reached or crashed
        truncated = self.step_count >= self.episode_horizon

        info = {"distance": dist, "reached": reached, "crashed": crashed}
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
