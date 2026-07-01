"""Stage 2: train multi-agent VMAS swarm with custom PPO and communication."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from swarmsim.env.swarm_env import load_config, make_swarm_env
from swarmsim.policy.network import CentralizedCritic, SwarmActor, swarm_global_dim, swarm_obs_dim
from swarmsim.policy.ppo import PPOConfig, SwarmPPOTrainer, SwarmRolloutBuffer


def set_comm_mode(cfg: dict, mode: str) -> dict:
    updated = copy.deepcopy(cfg)
    updated["comm"]["mode"] = mode
    return updated


def train(
    comm_mode: str = "full",
    total_timesteps: int | None = None,
    num_envs: int = 8,
    rollout_steps: int | None = None,
) -> Path:
    cfg = set_comm_mode(load_config(), comm_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_cfg = PPOConfig.from_config(cfg)
    if rollout_steps is not None:
        ppo_cfg.rollout_steps = rollout_steps
    train_cfg = cfg["training"]
    env_cfg = cfg["env"]
    comm_cfg = cfg["comm"]

    if total_timesteps is None:
        total_timesteps = train_cfg["swarm_total_timesteps"]

    env, action_dim = make_swarm_env(cfg, num_envs=num_envs, device=str(device))
    scenario = env.scenario
    num_agents = env_cfg["num_agents"]
    obs_dim = swarm_obs_dim(env_cfg["local_window_k"], env_cfg["max_neighbors"], comm_cfg["message_dim"])
    global_dim = swarm_global_dim(num_agents, cfg["critic"]["grid_downsample"])

    actor = SwarmActor(obs_dim, comm_cfg["message_dim"], comm_mode=comm_cfg["mode"]).to(device)
    critic = CentralizedCritic(global_dim).to(device)
    trainer = SwarmPPOTrainer(actor, critic, ppo_cfg, device)

    steps_per_rollout = ppo_cfg.rollout_steps
    buffer = SwarmRolloutBuffer(steps_per_rollout * num_agents * num_envs, obs_dim, action_dim, device)

    weights_dir = Path(__file__).resolve().parents[2] / train_cfg["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"swarm_policy_{comm_mode}.pt"

    log_dir = Path(__file__).resolve().parents[2] / train_cfg["tensorboard_dir"] / f"swarm_{comm_mode}"
    writer = SummaryWriter(log_dir=str(log_dir))

    obs = env.reset()
    global_step = 0
    update_idx = 0
    episode_returns = torch.zeros(num_envs, device=device)
    episode_count = 0

    while global_step < total_timesteps:
        while not buffer.full() and global_step < total_timesteps:
            global_state = scenario.build_global_state()
            actions_to_env = []
            step_records = []

            for agent_idx in range(num_agents):
                agent_obs = obs[agent_idx].to(device)
                with torch.no_grad():
                    move, message, log_prob, _ = actor.act(agent_obs)
                    value = critic(global_state)

                if comm_cfg["mode"] == "none" or message is None:
                    full_action = move
                else:
                    full_action = torch.cat([move, message], dim=-1)

                actions_to_env.append(full_action)
                step_records.append((agent_obs, global_state, full_action, log_prob, value))

            next_obs, rews, dones, _ = env.step(actions_to_env)

            team_reward = sum(rews) / float(num_agents)
            done_flag = dones[0] if isinstance(dones, list) else dones
            if done_flag.ndim > 1:
                done_flag = done_flag.any(dim=-1)

            for agent_obs, gs, action, log_prob, value in step_records:
                for env_i in range(num_envs):
                    buffer.add(
                        agent_obs[env_i],
                        gs[env_i],
                        action[env_i],
                        team_reward[env_i],
                        done_flag[env_i].float(),
                        value[env_i],
                        log_prob[env_i],
                    )

            global_step += num_envs
            episode_returns += team_reward
            if done_flag.any():
                finished = int(done_flag.sum().item())
                mean_return = episode_returns[done_flag].mean().item() if finished > 0 else 0.0
                writer.add_scalar("train/episode_return", mean_return, episode_count)
                writer.add_scalar("train/coverage", scenario.coverage.mean().item(), episode_count)
                writer.add_scalar(
                    "train/message_l2",
                    scenario.outgoing_messages.norm(dim=-1).mean().item(),
                    episode_count,
                )
                episode_returns[done_flag] = 0.0
                episode_count += finished

            obs = next_obs

        if buffer.ptr == 0:
            break

        with torch.no_grad():
            last_global = scenario.build_global_state()
            last_value = critic(last_global).mean()
        buffer.compute_gae(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        metrics = trainer.update(buffer)
        buffer.reset()
        update_idx += 1

        if update_idx % train_cfg["log_interval"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, update_idx)

        if update_idx % train_cfg["save_interval"] == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "comm_mode": comm_mode}, save_path)

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "comm_mode": comm_mode}, save_path)
    writer.close()
    print(f"Swarm training complete ({comm_mode}). Weights saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm-mode", choices=["full", "null", "none"], default="full")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=None)
    args = parser.parse_args()
    train(args.comm_mode, args.timesteps, args.num_envs, args.rollout_steps)
