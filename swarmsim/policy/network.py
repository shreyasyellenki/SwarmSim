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
    """Shared actor with movement + message heads for Stage 2.

    Optionally recurrent (GRU): when use_gru is True, the actor carries a hidden
    state across timesteps within an episode so a deterministic policy can
    condition on history (where it has been), not just the current observation.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int = 8,
        hidden: int = 256,
        comm_mode: str = "full",
        init_log_std: float = 0.0,
        use_gru: bool = False,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.comm_mode = comm_mode
        self.message_dim = message_dim
        self.use_gru = use_gru
        self.gru_hidden = gru_hidden if use_gru else 0
        self.body = mlp([obs_dim, hidden, hidden])
        self.gru = nn.GRUCell(hidden, gru_hidden) if use_gru else None
        head_in = gru_hidden if use_gru else hidden
        self.movement_head = nn.Linear(head_in, 2)
        self.message_head = nn.Linear(head_in, message_dim) if comm_mode != "none" else None
        self.log_std = nn.Parameter(torch.full((2,), float(init_log_std)))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru_hidden, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.body(obs)
        if self.use_gru:
            if hidden is None:
                hidden = torch.zeros(obs.shape[0], self.gru_hidden, device=obs.device)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
            new_hidden = self.gru(feat, hidden)
            head_in = new_hidden
        else:
            new_hidden = None
            head_in = feat
        movement = torch.tanh(self.movement_head(head_in))
        if self.message_head is None:
            return movement, None, new_hidden
        message = torch.tanh(self.message_head(head_in))
        return movement, message, new_hidden

    def act_deterministic(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Mean action for inference / demo (no sampling noise)."""
        return self.forward(obs, hidden)

    def _movement_std(self, movement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        movement = torch.nan_to_num(movement, nan=0.0, posinf=1.0, neginf=-1.0)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(movement)
        return movement, std

    def act(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        movement, message, new_hidden = self.forward(obs, hidden)
        movement, std = self._movement_std(movement)
        dist = Normal(movement, std)
        action_move = dist.sample()
        action_move = torch.clamp(action_move, -1.0, 1.0)
        log_prob = dist.log_prob(action_move).sum(-1)
        return action_move, message, log_prob, movement, new_hidden

    def evaluate(
        self, obs: torch.Tensor, action_move: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        movement, message, _ = self.forward(obs, hidden)
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


def swarm_obs_dim(local_k: int, max_neighbors: int, message_dim: int, global_map_cells: int = 0) -> int:
    return 4 + local_k * local_k + max_neighbors * (2 + message_dim) + global_map_cells


def swarm_global_dim(num_agents: int, grid_downsample: int) -> int:
    return num_agents * 4 + grid_downsample * grid_downsample


def swarm_action_dim(message_dim: int, comm_mode: str) -> int:
    if comm_mode == "none":
        return 2
    return 2 + message_dim
