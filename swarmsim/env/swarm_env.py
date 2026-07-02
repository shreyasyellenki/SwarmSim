"""Stage 2: VMAS multi-agent swarm exploration with learned communication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import vmas
import yaml
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from swarmsim.policy.network import swarm_global_dim, swarm_obs_dim


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class SwarmExplorationScenario(BaseScenario):
    """Cooperative grid exploration with optional inter-agent messaging."""

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        cfg = kwargs.get("config") or load_config()
        env_cfg = cfg["env"]
        comm_cfg = cfg["comm"]

        self.cfg = cfg
        self.num_agents = env_cfg["num_agents"]
        self.grid_size = env_cfg["grid_size"]
        self.local_k = env_cfg["local_window_k"]
        self.comm_radius = env_cfg["comm_radius"]
        self.max_neighbors = env_cfg["max_neighbors"]
        self.world_size = env_cfg["world_size"]
        self.message_dim = comm_cfg["message_dim"]
        self.comm_mode = comm_cfg.get("mode", "full")
        self.coverage_target = env_cfg["coverage_target"]
        self.reward_cfg = cfg["reward"]

        self.cell_size = self.world_size / self.grid_size
        self.comm_radius_world = self.comm_radius * self.cell_size
        self.obs_dim = swarm_obs_dim(self.local_k, self.max_neighbors, self.message_dim)

        world = World(
            batch_dim,
            device,
            dt=0.1,
            drag=0.15,
            x_semidim=self.world_size / 2.0,
            y_semidim=self.world_size / 2.0,
            dim_c=0,
        )
        self.plot_grid = True
        self.grid_spacing = self.cell_size

        action_size = 2 if self.comm_mode == "none" else 2 + self.message_dim
        colors = [Color.BLUE, Color.GREEN, Color.RED, Color.ORANGE, Color.PURPLE, Color.GRAY]
        for i in range(self.num_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=colors[i % len(colors)],
                shape=Sphere(radius=0.02),
                u_range=1.0,
                u_multiplier=0.4,
                max_speed=0.12,
                action_size=action_size,
            )
            world.add_agent(agent)

        self.explored = torch.zeros(
            batch_dim, self.grid_size, self.grid_size, device=device, dtype=torch.int32
        )
        self.visit_count = torch.zeros(
            batch_dim, self.grid_size, self.grid_size, device=device, dtype=torch.int32
        )
        self.incoming_messages = torch.zeros(
            batch_dim, self.num_agents, self.max_neighbors, self.message_dim, device=device
        )
        self.neighbor_rel_pos = torch.zeros(
            batch_dim, self.num_agents, self.max_neighbors, 2, device=device
        )
        self.outgoing_messages = torch.zeros(
            batch_dim, self.num_agents, self.message_dim, device=device
        )
        self.new_cells = torch.zeros(batch_dim, device=device)
        self.coverage = torch.zeros(batch_dim, device=device)
        self._step_count = torch.zeros(batch_dim, device=device, dtype=torch.int32)

        return world

    def _world_to_cell(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.world_size / 2.0
        cx = ((pos[..., 0] + half) / self.world_size * self.grid_size).long().clamp(0, self.grid_size - 1)
        cy = ((pos[..., 1] + half) / self.world_size * self.grid_size).long().clamp(0, self.grid_size - 1)
        return cx, cy

    def reset_world_at(self, env_index: int | None = None):
        if env_index is None:
            for idx in range(self.world.batch_dim):
                self.reset_world_at(idx)
            return

        for agent in self.world.agents:
            pos = torch.empty(self.world.dim_p, device=self.world.device)
            pos.uniform_(-self.world_size / 2.0 + 0.1, self.world_size / 2.0 - 0.1)
            agent.set_pos(pos, batch_index=env_index)
            agent.set_vel(torch.zeros(self.world.dim_p, device=self.world.device), batch_index=env_index)

        self.explored[env_index].zero_()
        self.visit_count[env_index].zero_()
        self.incoming_messages[env_index].zero_()
        self.neighbor_rel_pos[env_index].zero_()
        self.outgoing_messages[env_index].zero_()
        self.new_cells[env_index] = 0.0
        self.coverage[env_index] = 0.0
        self._step_count[env_index] = 0
        self._mark_all_agents(env_index)
        self._update_communication_for_env(env_index)

    def _mark_all_agents(self, env_index: int | None = None):
        indices = [env_index] if env_index is not None else list(range(self.world.batch_dim))
        for idx in indices:
            new_count = 0
            for agent_id, agent in enumerate(self.world.agents):
                cx, cy = self._world_to_cell(agent.state.pos)
                cx_i, cy_i = cx[idx].item(), cy[idx].item()
                self.visit_count[idx, cx_i, cy_i] += 1
                if self.explored[idx, cx_i, cy_i] == 0:
                    self.explored[idx, cx_i, cy_i] = agent_id + 1
                    new_count += 1
            self.new_cells[idx] = float(new_count)
            explored_cells = (self.explored[idx] > 0).sum().float()
            self.coverage[idx] = explored_cells / float(self.grid_size * self.grid_size)

    def process_action(self, agent: Agent):
        u = agent.action.u
        agent.action.u = u[..., :2]
        agent_index = self.world.agents.index(agent)
        if u.shape[-1] > 2:
            self.outgoing_messages[:, agent_index] = torch.tanh(u[..., 2:])

    def _update_communication_for_env(self, env_index: int | None = None):
        for receiver_id, receiver in enumerate(self.world.agents):
            neighbors = []
            for other_id, other in enumerate(self.world.agents):
                if other_id == receiver_id:
                    continue
                delta = other.state.pos - receiver.state.pos
                dist = torch.linalg.vector_norm(delta, dim=-1)
                neighbors.append((dist, delta, other_id))

            neighbors.sort(key=lambda x: x[0].mean().item())
            padded_rel = torch.zeros(
                self.world.batch_dim, self.max_neighbors, 2, device=self.world.device
            )
            padded_msg = torch.zeros(
                self.world.batch_dim,
                self.max_neighbors,
                self.message_dim,
                device=self.world.device,
            )

            for slot, (dist, delta, other_id) in enumerate(neighbors[: self.max_neighbors]):
                mask = (dist <= self.comm_radius_world).unsqueeze(-1).float()
                padded_rel[:, slot] = delta * mask
                if self.comm_mode == "full":
                    padded_msg[:, slot] = self.outgoing_messages[:, other_id] * mask

            self.neighbor_rel_pos[:, receiver_id] = padded_rel
            self.incoming_messages[:, receiver_id] = padded_msg

    def post_step(self):
        self._step_count += 1
        self._mark_all_agents()
        self._update_communication_for_env()

    def _local_patch(self, agent: Agent) -> torch.Tensor:
        cx, cy = self._world_to_cell(agent.state.pos)
        half = self.local_k // 2
        patch = torch.zeros(
            self.world.batch_dim, self.local_k, self.local_k, device=self.world.device
        )
        for i in range(self.local_k):
            for j in range(self.local_k):
                gx = (cx - half + i).clamp(0, self.grid_size - 1)
                gy = (cy - half + j).clamp(0, self.grid_size - 1)
                batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
                patch[:, i, j] = (self.explored[batch_idx, gx, gy] > 0).float()
        return patch.reshape(self.world.batch_dim, -1)

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        pos = agent.state.pos
        vel = agent.state.vel
        half = self.world_size / 2.0
        norm_pos = torch.stack(
            [(pos[:, 0] + half) / self.world_size, (pos[:, 1] + half) / self.world_size], dim=-1
        )
        norm_vel = torch.clamp(vel[:, :2] / 0.12, -1.0, 1.0)
        local = self._local_patch(agent)
        rel = self.neighbor_rel_pos[:, agent_index].reshape(self.world.batch_dim, -1)
        msgs = self.incoming_messages[:, agent_index].reshape(self.world.batch_dim, -1)
        return torch.cat([norm_pos, norm_vel, local, rel, msgs], dim=-1)

    def reward(self, agent: Agent):
        r = self.reward_cfg
        team_reward = r["alpha"] * self.new_cells
        cx, cy = self._world_to_cell(agent.state.pos)
        batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
        revisit = (self.visit_count[batch_idx, cx, cy] > 1).float()
        return team_reward - r["gamma"] * revisit

    def done(self):
        return self.coverage >= self.coverage_target

    def info(self, agent: Agent) -> dict[str, torch.Tensor]:
        return {
            "coverage": self.coverage,
            "new_cells": self.new_cells,
            "messages": self.outgoing_messages,
        }

    def build_global_state(self) -> torch.Tensor:
        factor = self.cfg["critic"]["grid_downsample"]
        block = self.grid_size // factor
        parts = []
        for agent in self.world.agents:
            parts.append(agent.state.pos[:, :2])
            parts.append(agent.state.vel[:, :2])
        agent_state = torch.cat(parts, dim=-1)

        down = torch.zeros(self.world.batch_dim, factor * factor, device=self.world.device)
        for i in range(factor):
            for j in range(factor):
                region = self.explored[
                    :, i * block : (i + 1) * block, j * block : (j + 1) * block
                ]
                idx = i * factor + j
                down[:, idx] = (region > 0).float().mean(dim=(-1, -2))
        return torch.cat([agent_state, down], dim=-1)

    def get_grid_numpy(self, env_index: int = 0):
        return self.explored[env_index].cpu().numpy().astype("uint8")

    def get_comm_links(self, env_index: int = 0) -> list[list[int]]:
        links = []
        agents = self.world.agents
        for i, a in enumerate(agents):
            for j in range(i + 1, len(agents)):
                delta = agents[j].state.pos[env_index] - a.state.pos[env_index]
                dist = torch.linalg.vector_norm(delta).item()
                if dist <= self.comm_radius_world:
                    links.append([i, j])
        return links


def make_swarm_env(
    config: dict[str, Any] | None = None,
    num_envs: int = 8,
    device: str = "cpu",
    max_steps: int | None = None,
):
    cfg = config or load_config()
    if max_steps is None:
        max_steps = cfg["env"]["episode_horizon"]

    comm_mode = cfg["comm"].get("mode", "full")
    message_dim = cfg["comm"]["message_dim"]
    action_dim = 2 + (0 if comm_mode == "none" else message_dim)

    env = vmas.make_env(
        scenario=SwarmExplorationScenario(),
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        max_steps=max_steps,
        config=cfg,
    )
    return env, action_dim
