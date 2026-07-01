"""Custom PPO implementation for single-agent and multi-agent training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


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
        return cls(**p)


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


class SwarmRolloutBuffer:
  """Rollout buffer for multi-agent VMAS training (flattened agent steps)."""

  def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device):
      self.size = size
      self.device = device
      self.ptr = 0
      self.obs = torch.zeros((size, obs_dim), device=device)
      self.global_states = torch.zeros((size, 1), device=device)  # resized on first add
      self.actions = torch.zeros((size, action_dim), device=device)
      self.rewards = torch.zeros(size, device=device)
      self.dones = torch.zeros(size, device=device)
      self.values = torch.zeros(size, device=device)
      self.log_probs = torch.zeros(size, device=device)
      self.advantages = torch.zeros(size, device=device)
      self.returns = torch.zeros(size, device=device)
      self._global_dim: int | None = None

  def add(self, obs, global_state, action, reward, done, value, log_prob):
      if self._global_dim is None:
          self._global_dim = global_state.shape[-1]
          self.global_states = torch.zeros((self.size, self._global_dim), device=self.device)
      i = self.ptr
      self.obs[i] = obs
      self.global_states[i] = global_state
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
        advantages = buffer.advantages[:n]
        returns = buffer.returns[:n]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                ratio = torch.exp(log_prob - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }


class SwarmPPOTrainer:
    """PPO trainer for swarm actor + centralized critic."""

    def __init__(self, actor: nn.Module, critic: nn.Module, cfg: PPOConfig, device: torch.device):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device
        params = list(actor.parameters()) + list(critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    def update(self, buffer: SwarmRolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        obs = buffer.obs
        global_states = buffer.global_states
        actions = buffer.actions
        old_log_probs = buffer.log_probs
        n = buffer.ptr
        advantages = buffer.advantages[:n]
        returns = buffer.returns[:n]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(n)
        policy_losses, value_losses, entropies = [], [], []

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

                log_prob, entropy, _ = self.actor.evaluate(mb_obs, mb_actions)
                values = self.critic(mb_global)
                ratio = torch.exp(log_prob - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }
