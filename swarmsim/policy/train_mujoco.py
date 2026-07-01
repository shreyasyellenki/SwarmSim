"""Stage 1: train single-agent MuJoCo waypoint navigation with custom PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from swarmsim.env.mujoco_nav import MuJoCoNavEnv
from swarmsim.policy.network import ActorCritic
from swarmsim.policy.ppo import PPOConfig, PPOTrainer, RolloutBuffer


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(total_timesteps: int | None = None) -> Path:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_cfg = PPOConfig.from_config(cfg)
    train_cfg = cfg["training"]
    stage_cfg = cfg["stage1_mujoco"]

    if total_timesteps is None:
        total_timesteps = train_cfg["total_timesteps"]

    env = MuJoCoNavEnv(cfg)
    obs_dim = stage_cfg["obs_dim"]
    action_dim = stage_cfg["action_dim"]

    model = ActorCritic(obs_dim, action_dim).to(device)
    trainer = PPOTrainer(model, ppo_cfg, device)
    buffer = RolloutBuffer(ppo_cfg.rollout_steps, obs_dim, action_dim, device)

    weights_dir = Path(__file__).resolve().parents[2] / train_cfg["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / train_cfg["mujoco_policy"]

    log_dir = Path(__file__).resolve().parents[2] / train_cfg["tensorboard_dir"] / "mujoco"
    writer = SummaryWriter(log_dir=str(log_dir))

    obs, _ = env.reset()
    global_step = 0
    update_idx = 0
    episode_reward = 0.0
    episode_count = 0

    while global_step < total_timesteps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, log_prob, value = model.act(obs_t)

        action_np = action.cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        episode_reward += reward

        buffer.add(
            obs_t,
            action,
            torch.tensor(reward, device=device),
            torch.tensor(float(done), device=device),
            value,
            log_prob,
        )
        global_step += 1
        obs = next_obs

        if done:
            writer.add_scalar("train/episode_reward", episode_reward, episode_count)
            writer.add_scalar("train/distance", info.get("distance", 0.0), episode_count)
            episode_reward = 0.0
            episode_count += 1
            obs, _ = env.reset()

        if buffer.full():
            with torch.no_grad():
                last_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, last_value = model(last_obs)
            buffer.compute_gae(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
            metrics = trainer.update(buffer)
            buffer.reset()
            update_idx += 1

            if update_idx % train_cfg["log_interval"] == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, update_idx)

            if update_idx % train_cfg["save_interval"] == 0:
                torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), save_path)
    writer.close()
    env.close()
    print(f"Training complete. Weights saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    train(args.timesteps)
