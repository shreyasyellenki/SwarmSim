"""Stage 2: VMAS multi-agent swarm exploration with learned communication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import vmas
import yaml
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from swarmsim.env.maze_gen import generate_tier1_layout
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
        self.obstacle_mode = env_cfg.get("obstacle_mode", "none")
        self.include_obstacle_obs = self.obstacle_mode != "none"
        self.global_map_downsample = env_cfg.get("global_map_downsample", 0) or 0
        self.global_map_cells = self.global_map_downsample ** 2

        self.cell_size = self.world_size / self.grid_size
        self.comm_radius_world = self.comm_radius * self.cell_size
        self.obs_dim = swarm_obs_dim(
            self.local_k,
            self.max_neighbors,
            self.message_dim,
            self.global_map_cells,
            include_obstacles=self.include_obstacle_obs,
        )

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
        self.agent_new_cells = torch.zeros(batch_dim, self.num_agents, device=device)
        self.coverage = torch.zeros(batch_dim, device=device)
        self.coverage_delta = torch.zeros(batch_dim, device=device)
        self._step_count = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self._exploration_bonus = torch.zeros(batch_dim, device=device)
        self._repulsion_penalty = torch.zeros(batch_dim, device=device)
        self._frontier_bonus = torch.zeros(batch_dim, device=device)
        self._diversity_penalty = torch.zeros(batch_dim, device=device)
        self.obstacles = torch.zeros(
            batch_dim, self.grid_size, self.grid_size, device=device, dtype=torch.bool
        )
        self.reachable_cells = torch.full(
            (batch_dim,), self.grid_size * self.grid_size, device=device, dtype=torch.int32
        )
        self._wall_hit = torch.zeros(batch_dim, self.num_agents, device=device)
        self._prev_positions: list[torch.Tensor] = []
        self._layout_seed = torch.randint(0, 2**31, (batch_dim,), device=device)

        return world

    def _world_to_cell(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.world_size / 2.0
        cx = ((pos[..., 0] + half) / self.world_size * self.grid_size).long().clamp(0, self.grid_size - 1)
        cy = ((pos[..., 1] + half) / self.world_size * self.grid_size).long().clamp(0, self.grid_size - 1)
        return cx, cy

    def _cell_to_world(self, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """Cell centers in world coordinates [batch, 2]."""
        half = self.world_size / 2.0
        x = (cx.float() + 0.5) / self.grid_size * self.world_size - half
        y = (cy.float() + 0.5) / self.grid_size * self.world_size - half
        return torch.stack([x, y], dim=-1)

    def _generate_layout_for_env(self, env_index: int) -> list[tuple[int, int]] | None:
        """Populate obstacles for one env; return spawn cells or None if no obstacles."""
        env_cfg = self.cfg["env"]
        if self.obstacle_mode == "none":
            self.obstacles[env_index].zero_()
            self.reachable_cells[env_index] = self.grid_size * self.grid_size
            return None

        if self.obstacle_mode == "scattered":
            seed = int(self._layout_seed[env_index].item())
            obstacle_arr, spawns, _ = generate_tier1_layout(
                grid_size=self.grid_size,
                num_agents=self.num_agents,
                n_segments=int(env_cfg.get("n_obstacle_segments", 10)),
                max_segment_len=int(env_cfg.get("max_segment_len", 4)),
                min_segment_spacing=int(env_cfg.get("min_segment_spacing", 2)),
                min_spawn_dist=int(env_cfg.get("min_spawn_dist", 6)),
                min_reachable_fraction=float(env_cfg.get("min_reachable_fraction", 0.85)),
                rng=np.random.default_rng(seed),
            )
            self.obstacles[env_index] = torch.from_numpy(obstacle_arr).to(
                device=self.world.device, dtype=torch.bool
            )
            self.reachable_cells[env_index] = int((~self.obstacles[env_index]).sum().item())
            return spawns

        raise ValueError(f"Unsupported obstacle_mode: {self.obstacle_mode}")

    def _place_agents_at_spawns(self, env_index: int, spawns: list[tuple[int, int]]):
        cx = torch.tensor([s[0] for s in spawns], device=self.world.device, dtype=torch.long)
        cy = torch.tensor([s[1] for s in spawns], device=self.world.device, dtype=torch.long)
        pos = self._cell_to_world(cx, cy)
        for agent_idx, agent in enumerate(self.world.agents):
            agent.set_pos(pos[agent_idx], batch_index=env_index)
            agent.set_vel(torch.zeros(self.world.dim_p, device=self.world.device), batch_index=env_index)

    def reset_world_at(self, env_index: int | None = None):
        if env_index is None:
            for idx in range(self.world.batch_dim):
                self.reset_world_at(idx)
            return

        spawns = self._generate_layout_for_env(env_index)
        if spawns is not None:
            self._place_agents_at_spawns(env_index, spawns)
        else:
            for agent in self.world.agents:
                pos = torch.empty(self.world.dim_p, device=self.world.device)
                pos.uniform_(-self.world_size / 2.0 + 0.1, self.world_size / 2.0 - 0.1)
                agent.set_pos(pos, batch_index=env_index)
                agent.set_vel(
                    torch.zeros(self.world.dim_p, device=self.world.device), batch_index=env_index
                )

        self.explored[env_index].zero_()
        self.visit_count[env_index].zero_()
        self.incoming_messages[env_index].zero_()
        self.neighbor_rel_pos[env_index].zero_()
        self.outgoing_messages[env_index].zero_()
        self.new_cells[env_index] = 0.0
        self.agent_new_cells[env_index].zero_()
        self.coverage[env_index] = 0.0
        self.coverage_delta[env_index] = 0.0
        self._step_count[env_index] = 0
        self._wall_hit[env_index].zero_()
        self._layout_seed[env_index] = torch.randint(
            0, 2**31, (1,), device=self.world.device
        ).squeeze()
        self._mark_all_agents(env_index)
        self._update_communication_for_env(env_index)

    def _mark_all_agents(self, env_index: int | None = None):
        indices = [env_index] if env_index is not None else list(range(self.world.batch_dim))
        for idx in indices:
            new_count = 0
            if env_index is None or idx == env_index:
                self.agent_new_cells[idx].zero_()
            for agent_id, agent in enumerate(self.world.agents):
                cx, cy = self._world_to_cell(agent.state.pos)
                cx_i, cy_i = cx[idx].item(), cy[idx].item()
                if self.obstacle_mode != "none" and self.obstacles[idx, cx_i, cy_i]:
                    continue
                self.visit_count[idx, cx_i, cy_i] += 1
                if self.explored[idx, cx_i, cy_i] == 0:
                    self.explored[idx, cx_i, cy_i] = agent_id + 1
                    new_count += 1
                    self.agent_new_cells[idx, agent_id] += 1.0
            self.new_cells[idx] = float(new_count)
            explored_open = (self.explored[idx] > 0) & (~self.obstacles[idx])
            explored_cells = explored_open.sum().float()
            reachable = self.reachable_cells[idx].float().clamp(min=1.0)
            self.coverage[idx] = explored_cells / reachable

    def process_action(self, agent: Agent):
        u = agent.action.u
        agent.action.u = u[..., :2]
        agent_index = self.world.agents.index(agent)
        if agent_index == 0:
            self._prev_positions = [a.state.pos.clone() for a in self.world.agents]
            self._wall_hit.zero_()
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
        prev_coverage = self.coverage.clone()
        self._step_count += 1
        self._mark_all_agents()
        if self.obstacle_mode != "none":
            self._resolve_obstacle_collisions()
        self.coverage_delta = self.coverage - prev_coverage
        self._update_communication_for_env()

    def _resolve_obstacle_collisions(self):
        for agent_index, agent in enumerate(self.world.agents):
            cx, cy = self._world_to_cell(agent.state.pos)
            batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
            blocked = self.obstacles[batch_idx, cx, cy]
            if not blocked.any():
                continue
            prev = self._prev_positions[agent_index]
            blocked_mask = blocked.unsqueeze(-1)
            agent.state.pos = torch.where(blocked_mask, prev, agent.state.pos)
            agent.state.vel = torch.where(blocked_mask, torch.zeros_like(agent.state.vel), agent.state.vel)
            self._wall_hit[:, agent_index] = blocked.float()

    def _local_patch(self, agent: Agent, source: torch.Tensor) -> torch.Tensor:
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
                patch[:, i, j] = source[batch_idx, gx, gy].float()
        return patch.reshape(self.world.batch_dim, -1)

    def _local_explored_patch(self, agent: Agent) -> torch.Tensor:
        explored = (self.explored > 0).float()
        return self._local_patch(agent, explored)

    def _local_obstacle_patch(self, agent: Agent) -> torch.Tensor:
        return self._local_patch(agent, self.obstacles)

    def _downsampled_grid(self, grid: torch.Tensor, factor: int) -> torch.Tensor:
        """Coarse (factor x factor) mean map, flattened per env."""
        block = self.grid_size // factor
        down = torch.zeros(self.world.batch_dim, factor * factor, device=self.world.device)
        for i in range(factor):
            for j in range(factor):
                region = grid[:, i * block : (i + 1) * block, j * block : (j + 1) * block]
                down[:, i * factor + j] = region.float().mean(dim=(-1, -2))
        return down

    def _downsampled_explored(self, factor: int) -> torch.Tensor:
        """Coarse (factor x factor) fraction-explored map, flattened per env."""
        explored = (self.explored > 0).float()
        if self.obstacle_mode != "none":
            explored = explored * (~self.obstacles).float()
        return self._downsampled_grid(explored, factor)

    def _downsampled_obstacles(self, factor: int) -> torch.Tensor:
        return self._downsampled_grid(self.obstacles.float(), factor)

    def _exploration_bonus_at(self, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """Count-based intrinsic bonus: higher reward for rarely visited cells."""
        r = self.reward_cfg
        delta = float(r.get("delta", 0.0))
        if delta <= 0.0:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
        visit_freq = self.visit_count[batch_idx, cx, cy].float().clamp(min=1.0)
        if self.obstacle_mode != "none":
            on_wall = self.obstacles[batch_idx, cx, cy].float()
            if on_wall.any():
                return torch.zeros(self.world.batch_dim, device=self.world.device)
        horizon = float(self.cfg["env"]["episode_horizon"])
        time_decay = 1.0 - self._step_count.float().clamp(max=horizon) / horizon
        return delta * time_decay / visit_freq

    def _repulsion_penalty_for(self, agent_index: int) -> torch.Tensor:
        """Light penalty when another agent is within repulsion_radius grid cells."""
        r = self.reward_cfg
        weight = float(r.get("repulsion", 0.0))
        radius = float(r.get("repulsion_radius", 3))
        if weight <= 0.0:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        agent = self.world.agents[agent_index]
        cx, cy = self._world_to_cell(agent.state.pos)
        min_dist = torch.full(
            (self.world.batch_dim,), float("inf"), device=self.world.device
        )
        for other_id, other in enumerate(self.world.agents):
            if other_id == agent_index:
                continue
            ox, oy = self._world_to_cell(other.state.pos)
            dist = torch.sqrt((cx - ox).float().pow(2) + (cy - oy).float().pow(2))
            min_dist = torch.minimum(min_dist, dist)

        too_close = (min_dist < radius).float()
        return weight * too_close

    def _frontier_bonus_for(self, agent: Agent) -> torch.Tensor:
        """Reward for being near unexplored cells (fraction unexplored in local window)."""
        weight = float(self.reward_cfg.get("frontier", 0.0))
        if weight <= 0.0:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        local_explored = self._local_explored_patch(agent)
        local_obstacles = self._local_obstacle_patch(agent)
        open_cells = 1.0 - local_obstacles
        unexplored_open = (1.0 - local_explored) * open_cells
        open_count = open_cells.sum(dim=-1).clamp(min=1.0)
        frontier_frac = unexplored_open.sum(dim=-1) / open_count
        return weight * frontier_frac

    def _heading_diversity_penalty_for(self, agent_index: int) -> torch.Tensor:
        """Penalty when velocity aligns with in-range neighbors (encourages heading spread)."""
        weight = float(self.reward_cfg.get("diversity", 0.0))
        if weight <= 0.0:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        agent = self.world.agents[agent_index]
        vel_i = agent.state.vel[:, :2]
        speed_i = torch.linalg.vector_norm(vel_i, dim=-1)
        dir_i = vel_i / speed_i.unsqueeze(-1).clamp(min=1e-6)

        total_sim = torch.zeros(self.world.batch_dim, device=self.world.device)
        count = torch.zeros(self.world.batch_dim, device=self.world.device)
        for other_id, other in enumerate(self.world.agents):
            if other_id == agent_index:
                continue
            delta = other.state.pos - agent.state.pos
            dist = torch.linalg.vector_norm(delta, dim=-1)
            in_range = dist <= self.comm_radius_world
            vel_j = other.state.vel[:, :2]
            speed_j = torch.linalg.vector_norm(vel_j, dim=-1)
            moving = (speed_i > 1e-4) & (speed_j > 1e-4) & in_range
            dir_j = vel_j / speed_j.unsqueeze(-1).clamp(min=1e-6)
            similarity = (dir_i * dir_j).sum(dim=-1).clamp(-1.0, 1.0)
            total_sim = total_sim + similarity * moving.float()
            count = count + moving.float()

        mean_sim = total_sim / count.clamp(min=1.0)
        has_neighbor = count > 0
        penalty = weight * mean_sim * has_neighbor.float()
        return penalty

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        pos = agent.state.pos
        vel = agent.state.vel
        half = self.world_size / 2.0
        norm_pos = torch.stack(
            [(pos[:, 0] + half) / self.world_size, (pos[:, 1] + half) / self.world_size], dim=-1
        )
        norm_vel = torch.clamp(vel[:, :2] / 0.12, -1.0, 1.0)
        local = self._local_explored_patch(agent)
        rel = self.neighbor_rel_pos[:, agent_index].reshape(self.world.batch_dim, -1)
        msgs = self.incoming_messages[:, agent_index].reshape(self.world.batch_dim, -1)
        parts = [norm_pos, norm_vel, local, rel, msgs]
        if self.include_obstacle_obs:
            parts.insert(3, self._local_obstacle_patch(agent))
        if self.global_map_cells > 0:
            parts.append(self._downsampled_explored(self.global_map_downsample))
        return torch.cat(parts, dim=-1)

    def reward(self, agent: Agent):
        r = self.reward_cfg
        agent_index = self.world.agents.index(agent)
        cx, cy = self._world_to_cell(agent.state.pos)
        batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
        revisit = (self.visit_count[batch_idx, cx, cy] > 1).float()

        exploration = self._exploration_bonus_at(cx, cy)
        repulsion = self._repulsion_penalty_for(agent_index)
        frontier = self._frontier_bonus_for(agent)
        diversity = self._heading_diversity_penalty_for(agent_index)
        self._exploration_bonus = exploration
        self._repulsion_penalty = repulsion
        self._frontier_bonus = frontier
        self._diversity_penalty = diversity

        mode = r.get("mode", "team_new_cells")
        if mode == "spread":
            personal = self.agent_new_cells[:, agent_index]
            team_progress = self.coverage_delta * float(self.grid_size * self.grid_size)
            base = r["alpha"] * personal + r["beta"] * team_progress - r["gamma"] * revisit
        else:
            base = r["alpha"] * self.new_cells - r["gamma"] * revisit

        wall_penalty = float(r.get("wall_penalty", 0.0))
        wall = self._wall_hit[:, agent_index] if wall_penalty > 0.0 else 0.0

        return base + exploration + frontier - repulsion - diversity - wall_penalty * wall

    def done(self):
        return self.coverage >= self.coverage_target

    def info(self, agent: Agent) -> dict[str, torch.Tensor]:
        return {
            "coverage": self.coverage,
            "new_cells": self.new_cells,
            "messages": self.outgoing_messages,
            "exploration_bonus": self._exploration_bonus,
            "frontier_bonus": self._frontier_bonus,
            "repulsion_penalty": self._repulsion_penalty,
            "diversity_penalty": self._diversity_penalty,
        }

    def build_global_state(self) -> torch.Tensor:
        factor = self.cfg["critic"]["grid_downsample"]
        parts = []
        for agent in self.world.agents:
            parts.append(agent.state.pos[:, :2])
            parts.append(agent.state.vel[:, :2])
        agent_state = torch.cat(parts, dim=-1)
        down = self._downsampled_explored(factor)
        chunks = [agent_state, down]
        if self.include_obstacle_obs:
            chunks.append(self._downsampled_obstacles(factor))
        return torch.cat(chunks, dim=-1)

    def get_grid_numpy(self, env_index: int = 0):
        return self.explored[env_index].cpu().numpy().astype("uint8")

    def get_obstacle_numpy(self, env_index: int = 0):
        return self.obstacles[env_index].cpu().numpy().astype("uint8")

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
