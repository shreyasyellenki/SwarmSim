"""Policy networks for Stage 1 (single-agent) and Stage 2 (swarm + comm)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes: list[int], activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Single-agent actor-critic for Stage 1 MuJoCo training."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.actor_body = mlp([obs_dim, hidden, hidden])
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_body = mlp([obs_dim, hidden, hidden])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        body = self.actor_body(obs)
        value = self.value_head(self.critic_body(obs)).squeeze(-1)
        return self.mu_head(body), value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class SwarmActor(nn.Module):
    """Shared actor with movement + message heads for Stage 2."""

    def __init__(self, obs_dim: int, message_dim: int = 8, hidden: int = 256, comm_mode: str = "full"):
        super().__init__()
        self.comm_mode = comm_mode
        self.message_dim = message_dim
        self.body = mlp([obs_dim, hidden, hidden])
        self.movement_head = nn.Linear(hidden, 2)
        self.message_head = nn.Linear(hidden, message_dim) if comm_mode != "none" else None
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        hidden = self.body(obs)
        movement = torch.tanh(self.movement_head(hidden))
        if self.message_head is None:
            return movement, None
        message = torch.tanh(self.message_head(hidden))
        return movement, message

    def act_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mean action for inference / demo (no sampling noise)."""
        return self.forward(obs)

    def _movement_std(self, movement: torch.Tensor) -> torch.Tensor:
        movement = torch.nan_to_num(movement, nan=0.0, posinf=1.0, neginf=-1.0)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(movement)
        return movement, std

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        movement, message = self.forward(obs)
        movement, std = self._movement_std(movement)
        dist = Normal(movement, std)
        action_move = dist.sample()
        action_move = torch.clamp(action_move, -1.0, 1.0)
        log_prob = dist.log_prob(action_move).sum(-1)
        return action_move, message, log_prob, movement

    def evaluate(
        self, obs: torch.Tensor, action_move: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        movement, message = self.forward(obs)
        movement, std = self._movement_std(movement)
        dist = Normal(movement, std)
        log_prob = dist.log_prob(action_move).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, message


class CentralizedCritic(nn.Module):
    """MAPPO-style centralized critic for cooperative swarm training."""

    def __init__(self, global_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([global_dim, hidden, hidden, 1])

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        global_state = torch.nan_to_num(global_state, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.net(global_state).squeeze(-1)


def swarm_obs_dim(local_k: int, max_neighbors: int, message_dim: int) -> int:
    return 4 + local_k * local_k + max_neighbors * (2 + message_dim)


def swarm_global_dim(num_agents: int, grid_downsample: int) -> int:
    return num_agents * 4 + grid_downsample * grid_downsample


def swarm_action_dim(message_dim: int, comm_mode: str) -> int:
    if comm_mode == "none":
        return 2
    return 2 + message_dim
