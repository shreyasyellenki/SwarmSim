"""Evaluate trained swarm policies and export rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from swarmsim.env.swarm_env import load_config, make_swarm_env
from swarmsim.policy.network import CentralizedCritic, SwarmActor, swarm_global_dim, swarm_obs_dim
from swarmsim.sim.state import build_sim_state


def apply_checkpoint_config(cfg: dict, checkpoint: dict) -> dict:
    """Merge env/reward settings stored in a training checkpoint into cfg."""
    updated = copy.deepcopy(cfg)
    if "global_map_downsample" in checkpoint:
        updated["env"]["global_map_downsample"] = checkpoint["global_map_downsample"]
    if "curiosity_delta" in checkpoint:
        updated["reward"]["delta"] = checkpoint["curiosity_delta"]
    if "frontier" in checkpoint:
        updated["reward"]["frontier"] = checkpoint["frontier"]
    if "repulsion" in checkpoint:
        updated["reward"]["repulsion"] = checkpoint["repulsion"]
    if "repulsion_radius" in checkpoint:
        updated["reward"]["repulsion_radius"] = checkpoint["repulsion_radius"]
    if "diversity" in checkpoint:
        updated["reward"]["diversity"] = checkpoint["diversity"]
    if "message_heading_aux" in checkpoint:
        updated["reward"]["message_heading_aux"] = checkpoint["message_heading_aux"]
    return updated


def infer_global_map_downsample(checkpoint: dict, cfg: dict) -> int:
    """Infer actor global-map downsample from checkpoint weight shapes."""
    if "global_map_downsample" in checkpoint:
        return int(checkpoint["global_map_downsample"])

    env_cfg = cfg["env"]
    comm_cfg = cfg["comm"]
    obs_dim = int(checkpoint["actor"]["body.0.weight"].shape[1])
    base_dim = swarm_obs_dim(
        env_cfg["local_window_k"], env_cfg["max_neighbors"], comm_cfg["message_dim"], 0
    )
    extra = obs_dim - base_dim
    if extra == 0:
        return 0
    side = int(round(math.sqrt(extra)))
    if side * side != extra:
        raise ValueError(
            f"Cannot infer global_map_downsample from obs_dim={obs_dim} (base={base_dim})"
        )
    return side


def cfg_for_checkpoint(cfg: dict, checkpoint: dict) -> dict:
    """Return cfg aligned with checkpoint env/reward layout (incl. legacy weights)."""
    updated = apply_checkpoint_config(cfg, checkpoint)
    updated["env"]["global_map_downsample"] = infer_global_map_downsample(checkpoint, cfg)
    if "use_gru" in checkpoint:
        updated.setdefault("policy", {})["use_gru"] = checkpoint["use_gru"]
    if "comm_mode" in checkpoint:
        updated.setdefault("comm", {})["mode"] = checkpoint["comm_mode"]
    return updated


def load_policy(weights_path: Path, cfg: dict, device: torch.device):
    comm_cfg = cfg["comm"]
    env_cfg = cfg["env"]
    policy_cfg = cfg.get("policy", {})
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    map_downsample = infer_global_map_downsample(checkpoint, cfg)
    global_map_cells = map_downsample ** 2
    comm_mode = checkpoint.get("comm_mode", comm_cfg.get("mode", "full"))
    use_gru = checkpoint.get("use_gru", policy_cfg.get("use_gru", False))
    gru_hidden = checkpoint.get("gru_hidden", policy_cfg.get("gru_hidden", 128))

    obs_dim = swarm_obs_dim(
        env_cfg["local_window_k"], env_cfg["max_neighbors"], comm_cfg["message_dim"], global_map_cells
    )
    global_dim = swarm_global_dim(env_cfg["num_agents"], cfg["critic"]["grid_downsample"])

    actor = SwarmActor(
        obs_dim,
        comm_cfg["message_dim"],
        comm_mode=comm_mode,
        use_gru=use_gru,
        gru_hidden=gru_hidden,
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, comm_mode


def run_episode(
    env,
    actor,
    scenario,
    cfg: dict,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = True,
):
    env_cfg = cfg["env"]
    num_agents = env_cfg["num_agents"]
    obs = env.reset(seed=seed)
    step = 0
    max_steps = env_cfg["episode_horizon"]
    coverage_threshold = cfg["eval"]["coverage_threshold"]
    time_to_threshold = max_steps

    trajectory = []
    hidden_states = [actor.initial_hidden(1, device) for _ in range(num_agents)]
    while step < max_steps:
        actions = []
        with torch.no_grad():
            for agent_idx in range(num_agents):
                agent_obs = obs[agent_idx].to(device)
                h_in = hidden_states[agent_idx]
                if deterministic:
                    move, message, h_out = actor.act_deterministic(agent_obs, h_in)
                else:
                    move, message, _, _, h_out = actor.act(agent_obs, h_in)
                hidden_states[agent_idx] = h_out
                if message is None:
                    actions.append(move)
                else:
                    actions.append(torch.cat([move, message], dim=-1))

        obs, _, dones, _ = env.step(actions)
        coverage = float(scenario.coverage[0].item())
        if time_to_threshold == max_steps and coverage >= coverage_threshold:
            time_to_threshold = step

        agents = []
        for i, agent in enumerate(scenario.world.agents):
            pos = agent.state.pos[0].cpu().numpy()
            vel = agent.state.vel[0].cpu().numpy()
            heading = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-4 else 0.0
            msg_mag = float(scenario.outgoing_messages[0, i].norm().item())
            half = env_cfg["world_size"] / 2.0
            agents.append(
                {
                    "id": i,
                    "x": float((pos[0] + half) / env_cfg["world_size"]),
                    "y": float((pos[1] + half) / env_cfg["world_size"]),
                    "heading": heading,
                    "msg_magnitude": msg_mag,
                }
            )

        state = build_sim_state(
            step=step,
            coverage_pct=coverage,
            grid=scenario.get_grid_numpy(0),
            agents=agents,
            comm_links=scenario.get_comm_links(0),
        )
        trajectory.append(state)
        step += 1

        done = bool(dones.any().item()) if hasattr(dones, "any") else bool(dones)
        if done or coverage >= env_cfg["coverage_target"]:
            break

    return time_to_threshold, coverage, trajectory


def evaluate(
    weights_path: Path,
    comm_mode: str | None = None,
    episodes: int | None = None,
    deterministic: bool = True,
) -> dict:
    cfg = load_config()
    if comm_mode:
        cfg["comm"]["mode"] = comm_mode

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = cfg_for_checkpoint(cfg, checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, _ = make_swarm_env(cfg, num_envs=1, device=str(device))
    scenario = env.scenario
    actor, _ = load_policy(weights_path, cfg, device)

    eval_cfg = cfg["eval"]
    seeds = eval_cfg["seeds"][: eval_cfg["num_seeds"]]
    episodes_per_seed = episodes or eval_cfg["episodes_per_seed"]

    times = []
    coverages = []
    for seed in seeds:
        for ep in range(episodes_per_seed):
            episode_seed = seed * 1000 + ep
            t, cov, _ = run_episode(
                env, actor, scenario, cfg, device, seed=episode_seed, deterministic=deterministic
            )
            times.append(t)
            coverages.append(cov)

    results = {
        "metric": eval_cfg["metric"],
        "mean_time_to_threshold": float(np.mean(times)),
        "std_time_to_threshold": float(np.std(times)),
        "mean_final_coverage": float(np.mean(coverages)),
        "num_episodes": len(times),
        "comm_mode": cfg["comm"]["mode"],
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--comm-mode", choices=["full", "null", "none"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--export", type=Path, default=None, help="Export one rollout JSON")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling")
    args = parser.parse_args()

    deterministic = not args.stochastic

    if args.export:
        cfg = load_config()
        if args.comm_mode:
            cfg["comm"]["mode"] = args.comm_mode
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        cfg = cfg_for_checkpoint(cfg, checkpoint)
        device = torch.device("cpu")
        env, _ = make_swarm_env(cfg, num_envs=1, device="cpu")
        actor, _ = load_policy(args.weights, cfg, device)
        _, _, traj = run_episode(
            env, actor, env.scenario, cfg, device, seed=0, deterministic=deterministic
        )
        args.export.write_text(json.dumps(traj))
        print(f"Exported rollout to {args.export}")
    else:
        results = evaluate(args.weights, args.comm_mode, args.episodes, deterministic=deterministic)
        print(json.dumps(results, indent=2))
