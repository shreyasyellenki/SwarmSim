"""Custom PPO implementation for single-agent and multi-agent training."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False)
    if torch.isfinite(adv_std) and adv_std > 1e-6:
        normalized = (advantages - adv_mean) / (adv_std + 1e-8)
    else:
        normalized = advantages - adv_mean
    return torch.clamp(normalized, -10.0, 10.0)


def _params_finite(*modules: nn.Module) -> bool:
    for module in modules:
        for param in module.parameters():
            if not torch.isfinite(param).all():
                return False
    return True


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    num_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "PPOConfig":
        p = cfg["ppo"]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in p.items() if k in known})


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.obs = torch.zeros((size, obs_dim), device=device)
        self.actions = torch.zeros((size, action_dim), device=device)
        self.rewards = torch.zeros(size, device=device)
        self.dones = torch.zeros(size, device=device)
        self.values = torch.zeros(size, device=device)
        self.log_probs = torch.zeros(size, device=device)
        self.advantages = torch.zeros(size, device=device)
        self.returns = torch.zeros(size, device=device)

    def add(self, obs, action, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.size

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        n = self.ptr
        last_gae = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]


@dataclass
class FlatBatch:
    """Flattened view of the filled region of a SwarmRolloutBuffer."""

    obs: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    velocities: torch.Tensor
    hidden_in: torch.Tensor | None
    size: int


class SwarmRolloutBuffer:
    """Rollout storage for multi-agent VMAS training.

    Tensors are shaped [T, num_agents, num_envs, ...] so GAE can run along the
    time axis independently for every (agent, env) trajectory. Flattening the
    agent and env axes into the time axis would make consecutive entries
    neighbours in the batch rather than in time, and the GAE recursion would
    bootstrap across trajectories. The PPO update consumes `flat_batch()`.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_agents: int,
        num_envs: int,
        obs_dim: int,
        global_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 0,
    ):
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.ptr = 0

        shape = (rollout_steps, num_agents, num_envs)
        self.obs = torch.zeros((*shape, obs_dim), device=device)
        self.global_states = torch.zeros((*shape, global_dim), device=device)
        self.actions = torch.zeros((*shape, action_dim), device=device)
        self.velocities = torch.zeros((*shape, 2), device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.next_values = torch.zeros(shape, device=device)
        self.terminated = torch.zeros(shape, device=device)
        self.truncated = torch.zeros(shape, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.hidden_in = (
            torch.zeros((*shape, hidden_dim), device=device) if hidden_dim > 0 else None
        )

    def add_step(
        self,
        obs,
        global_state,
        actions,
        velocities,
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        log_probs,
        hidden_in=None,
    ):
        """Append one environment step. Leading dims are [num_agents, num_envs]."""
        t = self.ptr
        self.obs[t] = obs
        self.global_states[t] = global_state
        self.actions[t] = actions
        self.velocities[t] = velocities
        self.rewards[t] = rewards
        self.values[t] = values
        self.next_values[t] = next_values
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.log_probs[t] = log_probs
        if self.hidden_in is not None and hidden_in is not None:
            self.hidden_in[t] = hidden_in
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.rollout_steps

    def reset(self):
        self.ptr = 0

    def compute_gae(self, gamma: float, lam: float):
        """GAE(lambda) along the time axis, independently per (agent, env).

        `next_values[t]` is the critic's estimate for the state that actually
        followed step t, captured before any environment reset. Only a true
        terminal has no future, so a time-limit truncation still bootstraps;
        either kind of episode end stops advantage credit propagating further
        back.
        """
        n = self.ptr
        last_gae = torch.zeros(self.num_agents, self.num_envs, device=self.device)
        for t in reversed(range(n)):
            terminated = self.terminated[t]
            episode_end = torch.clamp(terminated + self.truncated[t], max=1.0)
            next_value = self.next_values[t] * (1.0 - terminated)
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * (1.0 - episode_end) * last_gae
            self.advantages[t] = last_gae
        self.advantages[:n].clamp_(-10.0, 10.0)
        self.returns[:n] = (self.advantages[:n] + self.values[:n]).clamp(-20.0, 20.0)

    def flat_batch(self) -> FlatBatch:
        n = self.ptr
        count = n * self.num_agents * self.num_envs
        return FlatBatch(
            obs=self.obs[:n].reshape(count, -1),
            global_states=self.global_states[:n].reshape(count, -1),
            actions=self.actions[:n].reshape(count, -1),
            log_probs=self.log_probs[:n].reshape(count),
            advantages=self.advantages[:n].reshape(count),
            returns=self.returns[:n].reshape(count),
            velocities=self.velocities[:n].reshape(count, 2),
            hidden_in=(
                self.hidden_in[:n].reshape(count, -1) if self.hidden_in is not None else None
            ),
            size=count,
        )


class PPOTrainer:
    def __init__(self, model: nn.Module, cfg: PPOConfig, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        obs = buffer.obs
        actions = buffer.actions
        old_log_probs = buffer.log_probs
        n = buffer.ptr
        advantages = _normalize_advantages(buffer.advantages[:n])
        returns = torch.clamp(buffer.returns[:n], -20.0, 20.0)

        indices = np.arange(n)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(cfg.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb = indices[start:end]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logp = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]

                log_prob, entropy, values = self.model.evaluate(mb_obs, mb_actions)
                if not torch.isfinite(log_prob).all() or not torch.isfinite(values).all():
                    continue
                ratio = torch.exp(torch.clamp(log_prob - mb_old_logp, -20.0, 20.0))
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if not _params_finite(self.model):
                    break

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }


class SwarmPPOTrainer:
    """PPO trainer for swarm actor + centralized critic."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        cfg: PPOConfig,
        device: torch.device,
        train_log_std: bool = True,
        message_heading_aux_coef: float = 0.0,
    ):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.message_heading_aux_coef = message_heading_aux_coef
        if train_log_std:
            actor_params = list(actor.parameters())
        else:
            actor_params = [p for n, p in actor.named_parameters() if n != "log_std"]
        params = actor_params + list(critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    def set_learning_rate(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, buffer: SwarmRolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        batch = buffer.flat_batch()
        obs = batch.obs
        global_states = batch.global_states
        actions = batch.actions
        old_log_probs = batch.log_probs
        n = batch.size
        advantages = _normalize_advantages(batch.advantages)
        returns = batch.returns

        indices = np.arange(n)
        policy_losses, value_losses, entropies, aux_losses = [], [], [], []

        for _ in range(cfg.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb = indices[start:end]
                mb_obs = obs[mb]
                mb_global = global_states[mb]
                mb_actions = actions[mb][:, :2]
                mb_old_logp = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]
                mb_hidden = batch.hidden_in[mb] if batch.hidden_in is not None else None
                mb_vel = batch.velocities[mb]

                log_prob, entropy, message = self.actor.evaluate(mb_obs, mb_actions, mb_hidden)
                values = self.critic(mb_global)
                if not torch.isfinite(log_prob).all() or not torch.isfinite(values).all():
                    continue
                ratio = torch.exp(torch.clamp(log_prob - mb_old_logp, -20.0, 20.0))
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                aux_loss_val = 0.0
                if (
                    self.message_heading_aux_coef > 0.0
                    and message is not None
                    and batch.actions.shape[-1] > 2
                ):
                    speed = torch.linalg.vector_norm(mb_vel, dim=-1)
                    moving = speed > 1e-4
                    if moving.any():
                        heading_target = mb_vel / speed.unsqueeze(-1).clamp(min=1e-6)
                        msg_heading = message[:, :2]
                        per_sample = (msg_heading - heading_target.detach()).pow(2).sum(dim=-1)
                        aux_loss = per_sample[moving].mean()
                        loss = loss + self.message_heading_aux_coef * aux_loss
                        aux_loss_val = aux_loss.item()

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()
                if not _params_finite(self.actor, self.critic):
                    break

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                if aux_loss_val:
                    aux_losses.append(aux_loss_val)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
        if aux_losses:
            metrics["message_heading_aux_loss"] = float(np.mean(aux_losses))
        return metrics
