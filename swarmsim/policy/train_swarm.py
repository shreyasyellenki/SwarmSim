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
from swarmsim.policy.ppo import PPOConfig, SwarmPPOTrainer, SwarmRolloutBuffer, _params_finite


def set_comm_mode(cfg: dict, mode: str) -> dict:
    updated = copy.deepcopy(cfg)
    updated["comm"]["mode"] = mode
    return updated


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return start + (end - start) * progress


def apply_training_schedules(
    actor: SwarmActor,
    trainer: SwarmPPOTrainer,
    cfg: dict,
    global_step: int,
    total_timesteps: int,
) -> dict[str, float]:
    """Update log_std and entropy_coef from config schedules. Returns logged values."""
    policy_cfg = cfg.get("policy", {})
    progress = global_step / max(total_timesteps, 1)
    logged: dict[str, float] = {}

    std_sched = policy_cfg.get("std_schedule", {})
    if std_sched.get("enabled"):
        log_std = linear_schedule(
            float(std_sched.get("start_log_std", 0.0)),
            float(std_sched.get("end_log_std", -1.6)),
            progress,
        )
        actor.log_std.data.fill_(log_std)
        logged["log_std"] = log_std
        logged["action_std"] = float(torch.exp(actor.log_std).mean().item())

    ent_sched = policy_cfg.get("entropy_schedule", {})
    if ent_sched.get("enabled"):
        entropy_coef = linear_schedule(
            float(ent_sched.get("start", 0.01)),
            float(ent_sched.get("end", 0.001)),
            progress,
        )
        trainer.cfg.entropy_coef = entropy_coef
        logged["entropy_coef"] = entropy_coef

    return logged


def train(
    comm_mode: str = "full",
    total_timesteps: int | None = None,
    num_envs: int = 8,
    rollout_steps: int | None = None,
    revisit_gamma: float | None = None,
    save_name: str | None = None,
    init_log_std: float | None = None,
    std_anneal: bool = False,
    entropy_anneal: bool = False,
    use_gru: bool | None = None,
    reward_mode: str | None = None,
) -> Path:
    cfg = set_comm_mode(load_config(), comm_mode)
    if revisit_gamma is not None:
        cfg["reward"]["gamma"] = revisit_gamma
    if reward_mode is not None:
        cfg["reward"]["mode"] = reward_mode
    if init_log_std is not None:
        cfg.setdefault("policy", {})["init_log_std"] = init_log_std
    policy_cfg = cfg.setdefault("policy", {})
    if use_gru is not None:
        policy_cfg["use_gru"] = use_gru
    if std_anneal:
        policy_cfg.setdefault("std_schedule", {})["enabled"] = True
    if entropy_anneal:
        policy_cfg.setdefault("entropy_schedule", {})["enabled"] = True
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
    global_map_cells = (env_cfg.get("global_map_downsample", 0) or 0) ** 2
    obs_dim = swarm_obs_dim(
        env_cfg["local_window_k"], env_cfg["max_neighbors"], comm_cfg["message_dim"], global_map_cells
    )
    global_dim = swarm_global_dim(num_agents, cfg["critic"]["grid_downsample"])

    actor_log_std = cfg.get("policy", {}).get("init_log_std", 0.0)
    use_gru_flag = bool(policy_cfg.get("use_gru", False))
    gru_hidden = int(policy_cfg.get("gru_hidden", 128))
    actor = SwarmActor(
        obs_dim,
        comm_cfg["message_dim"],
        comm_mode=comm_cfg["mode"],
        init_log_std=actor_log_std,
        use_gru=use_gru_flag,
        gru_hidden=gru_hidden,
    ).to(device)
    critic = CentralizedCritic(global_dim).to(device)
    std_schedule_on = policy_cfg.get("std_schedule", {}).get("enabled", False)
    trainer = SwarmPPOTrainer(actor, critic, ppo_cfg, device, train_log_std=not std_schedule_on)

    steps_per_rollout = ppo_cfg.rollout_steps
    hidden_dim = actor.gru_hidden
    buffer = SwarmRolloutBuffer(
        steps_per_rollout * num_agents * num_envs, obs_dim, action_dim, device, hidden_dim=hidden_dim
    )

    weights_dir = Path(__file__).resolve().parents[2] / train_cfg["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_stem = save_name or f"swarm_policy_{comm_mode}"
    save_path = weights_dir / f"{weight_stem}.pt"

    run_name = weight_stem.removeprefix("swarm_policy_")
    log_dir = Path(__file__).resolve().parents[2] / train_cfg["tensorboard_dir"] / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    obs = env.reset()
    global_step = 0
    update_idx = 0
    episode_returns = torch.zeros(num_envs, device=device)
    episode_count = 0
    nan_recoveries = 0
    last_good: dict[str, dict] | None = None

    def build_checkpoint() -> dict:
        ckpt = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "comm_mode": comm_mode,
            "revisit_gamma": cfg["reward"]["gamma"],
            "init_log_std": cfg.get("policy", {}).get("init_log_std", 0.0),
            "use_gru": use_gru_flag,
            "gru_hidden": gru_hidden,
            "reward_mode": cfg["reward"].get("mode", "team_new_cells"),
        }
        return ckpt

    hidden_states = [actor.initial_hidden(num_envs, device) for _ in range(num_agents)]

    while global_step < total_timesteps:
        schedule_vals = apply_training_schedules(actor, trainer, cfg, global_step, total_timesteps)
        while not buffer.full() and global_step < total_timesteps:
            global_state = scenario.build_global_state()
            actions_to_env = []
            step_records = []

            for agent_idx in range(num_agents):
                agent_obs = torch.nan_to_num(obs[agent_idx].to(device), nan=0.0, posinf=1.0, neginf=-1.0)
                h_in = hidden_states[agent_idx]
                with torch.no_grad():
                    move, message, log_prob, _, h_out = actor.act(agent_obs, h_in)
                    value = critic(global_state)
                hidden_states[agent_idx] = h_out

                if comm_cfg["mode"] == "none" or message is None:
                    full_action = move
                else:
                    full_action = torch.cat([move, message], dim=-1)

                actions_to_env.append(full_action)
                step_records.append((agent_obs, global_state, full_action, log_prob, value, h_in))

            next_obs, rews, dones, _ = env.step(actions_to_env)

            team_reward = sum(rews) / float(num_agents)
            done_flag = dones[0] if isinstance(dones, list) else dones
            if done_flag.ndim > 1:
                done_flag = done_flag.any(dim=-1)

            for agent_obs, gs, action, log_prob, value, h_in in step_records:
                for env_i in range(num_envs):
                    buffer.add(
                        agent_obs[env_i],
                        gs[env_i],
                        action[env_i],
                        team_reward[env_i],
                        done_flag[env_i].float(),
                        value[env_i],
                        log_prob[env_i],
                        hidden_in=h_in[env_i] if h_in is not None else None,
                    )

            if actor.use_gru and done_flag.any():
                for agent_idx in range(num_agents):
                    hidden_states[agent_idx][done_flag] = 0.0

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

        if not _params_finite(actor, critic):
            if last_good is not None:
                actor.load_state_dict(last_good["actor"])
                critic.load_state_dict(last_good["critic"])
            else:
                raise RuntimeError("Policy parameters became non-finite and no checkpoint is available.")

        with torch.no_grad():
            last_global = scenario.build_global_state()
            last_value = critic(last_global).mean()
        buffer.compute_gae(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        buffer.advantages[: buffer.ptr] = torch.clamp(buffer.advantages[: buffer.ptr], -10.0, 10.0)
        buffer.returns[: buffer.ptr] = torch.clamp(buffer.returns[: buffer.ptr], -20.0, 20.0)

        pre_update = {
            "actor": {k: v.clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in critic.state_dict().items()},
        }
        try:
            metrics = trainer.update(buffer)
        except (ValueError, RuntimeError) as exc:
            nan_recoveries += 1
            restore = last_good or pre_update
            actor.load_state_dict(restore["actor"])
            critic.load_state_dict(restore["critic"])
            writer.add_scalar("train/nan_recoveries", nan_recoveries, update_idx)
            print(f"Warning: PPO update failed ({exc}); restored last checkpoint.")
            buffer.reset()
            continue
        if not _params_finite(actor, critic):
            nan_recoveries += 1
            restore = last_good or pre_update
            actor.load_state_dict(restore["actor"])
            critic.load_state_dict(restore["critic"])
            writer.add_scalar("train/nan_recoveries", nan_recoveries, update_idx)
            buffer.reset()
            continue

        last_good = {
            "actor": {k: v.clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in critic.state_dict().items()},
        }
        buffer.reset()
        update_idx += 1

        if update_idx % train_cfg["log_interval"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, update_idx)
            if revisit_gamma is not None:
                writer.add_scalar("train/revisit_gamma", revisit_gamma, update_idx)
            for k, v in schedule_vals.items():
                writer.add_scalar(f"train/{k}", v, update_idx)

        if update_idx % train_cfg["save_interval"] == 0:
            torch.save(build_checkpoint(), save_path)

    checkpoint = build_checkpoint()

    apply_training_schedules(actor, trainer, cfg, total_timesteps, total_timesteps)
    checkpoint = build_checkpoint()

    if not _params_finite(actor, critic) and last_good is not None:
        actor.load_state_dict(last_good["actor"])
        critic.load_state_dict(last_good["critic"])

    torch.save(checkpoint, save_path)
    writer.close()
    print(f"Swarm training complete ({comm_mode}, revisit_gamma={cfg['reward']['gamma']}). Weights saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm-mode", choices=["full", "null", "none"], default="full")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Revisit penalty (reward.gamma)")
    parser.add_argument("--save-name", type=str, default=None, help="Weight filename stem, e.g. swarm_policy_full_gamma03")
    parser.add_argument("--init-log-std", type=float, default=None, help="Initial log std of movement Gaussian")
    parser.add_argument("--std-anneal", action="store_true", help="Linearly anneal log_std per policy.std_schedule")
    parser.add_argument("--entropy-anneal", action="store_true", help="Linearly decay entropy_coef per policy.entropy_schedule")
    parser.add_argument("--use-gru", action="store_true", help="Use a recurrent (GRU) actor with per-episode hidden state")
    parser.add_argument("--reward-mode", choices=["team_new_cells", "spread"], default=None)
    args = parser.parse_args()
    train(
        args.comm_mode,
        args.timesteps,
        args.num_envs,
        args.rollout_steps,
        revisit_gamma=args.gamma,
        save_name=args.save_name,
        init_log_std=args.init_log_std,
        std_anneal=args.std_anneal,
        entropy_anneal=args.entropy_anneal,
        use_gru=True if args.use_gru else None,
        reward_mode=args.reward_mode,
    )
