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
    if "episode_horizon" in checkpoint:
        updated["env"]["episode_horizon"] = checkpoint["episode_horizon"]
    if "local_window_k" in checkpoint:
        updated["env"]["local_window_k"] = checkpoint["local_window_k"]
    if "obstacle_mode" in checkpoint:
        updated["env"]["obstacle_mode"] = checkpoint["obstacle_mode"]
    return updated


def _obstacles_enabled(cfg: dict, checkpoint: dict | None = None) -> bool:
    mode = "none"
    if checkpoint and "obstacle_mode" in checkpoint:
        mode = checkpoint["obstacle_mode"]
    else:
        mode = cfg.get("env", {}).get("obstacle_mode", "none")
    return mode != "none"


def infer_global_map_downsample(checkpoint: dict, cfg: dict) -> int:
    """Infer actor global-map downsample from checkpoint weight shapes."""
    if "global_map_downsample" in checkpoint:
        return int(checkpoint["global_map_downsample"])

    env_cfg = cfg["env"]
    comm_cfg = cfg["comm"]
    obs_dim = int(checkpoint["actor"]["body.0.weight"].shape[1])
    include_obstacles = _obstacles_enabled(cfg, checkpoint)
    base_dim = swarm_obs_dim(
        env_cfg["local_window_k"],
        env_cfg["max_neighbors"],
        comm_cfg["message_dim"],
        0,
        include_obstacles=include_obstacles,
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
    include_obstacles = _obstacles_enabled(cfg, checkpoint)

    obs_dim = swarm_obs_dim(
        env_cfg["local_window_k"],
        env_cfg["max_neighbors"],
        comm_cfg["message_dim"],
        global_map_cells,
        include_obstacles=include_obstacles,
    )
    global_dim = swarm_global_dim(
        env_cfg["num_agents"],
        cfg["critic"]["grid_downsample"],
        include_obstacles=include_obstacles,
    )

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
    collect_trajectory: bool = True,
):
    """Roll out one episode.

    Returns ``(stats, trajectory)``. ``stats["coverage_curve"]`` holds coverage
    after every step, which is what the uncensored metrics are derived from.
    Set ``collect_trajectory=False`` to skip the per-step visualiser payload,
    which dominates runtime in large evaluations.
    """
    env_cfg = cfg["env"]
    num_agents = env_cfg["num_agents"]
    obs = env.reset(seed=seed)
    step = 0
    max_steps = env_cfg["episode_horizon"]
    coverage_threshold = cfg["eval"]["coverage_threshold"]
    time_to_threshold = max_steps
    coverage = 0.0

    coverage_curve: list[float] = []
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
        coverage_curve.append(coverage)
        if time_to_threshold == max_steps and coverage >= coverage_threshold:
            time_to_threshold = step

        if collect_trajectory:
            agents = []
            for i, agent in enumerate(scenario.world.agents):
                pos = agent.state.pos[0].cpu().numpy()
                vel = agent.state.vel[0].cpu().numpy()
                heading = (
                    float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-4 else 0.0
                )
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

            trajectory.append(
                build_sim_state(
                    step=step,
                    coverage_pct=coverage,
                    grid=scenario.get_grid_numpy(0),
                    agents=agents,
                    comm_links=scenario.get_comm_links(0),
                )
            )
        step += 1

        done = bool(dones.any().item()) if hasattr(dones, "any") else bool(dones)
        if done or coverage >= env_cfg["coverage_target"]:
            break

    stats = {
        "final_coverage": coverage,
        "steps": step,
        "time_to_threshold": time_to_threshold,
        "reached_threshold": coverage >= coverage_threshold,
        "coverage_curve": coverage_curve,
        "max_steps": max_steps,
    }
    return stats, trajectory


COVERAGE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90)


def _summary(values) -> dict:
    """Mean with dispersion and a 95% CI, for comparing runs that overlap."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "sem": 0.0, "ci95": 0.0, "n": 0}
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    return {
        "mean": float(arr.mean()),
        "std": std,
        "sem": sem,
        "ci95": 1.96 * sem,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": n,
    }


def _coverage_auc(curve: list[float], max_steps: int) -> float:
    """Mean coverage over the full horizon, padding early finishes with the final value.

    Without the padding an episode that terminates early by hitting the
    coverage target would score lower than one that crawls to the horizon.
    """
    if not curve:
        return 0.0
    padded = curve + [curve[-1]] * (max_steps - len(curve))
    return float(np.mean(padded[:max_steps]))


def _threshold_stats(curves: list[list[float]], max_steps: int) -> dict:
    """Per-threshold reach rate and time-to-reach among episodes that got there.

    Reported separately because a mean over only the successful episodes is
    meaningless without knowing how many succeeded.
    """
    out = {}
    for thresh in COVERAGE_THRESHOLDS:
        steps = []
        for curve in curves:
            hit = next((i for i, c in enumerate(curve) if c >= thresh), None)
            if hit is not None:
                steps.append(hit)
        rate = len(steps) / len(curves) if curves else 0.0
        out[f"{thresh:.2f}"] = {
            "reach_rate": rate,
            "censored_fraction": 1.0 - rate,
            "mean_steps_if_reached": float(np.mean(steps)) if steps else None,
            "median_steps_if_reached": float(np.median(steps)) if steps else None,
        }
    return out


def resolve_comm_modes(cfg: dict, checkpoint: dict, requested: str | None) -> tuple[str, str]:
    """Split the requested comm mode into (policy mode, channel delivery mode).

    The policy mode is fixed by the checkpoint -- it decides whether a message
    head exists and how wide the action is -- so only delivery can be
    overridden at evaluation time.
    """
    policy_mode = checkpoint.get("comm_mode", cfg["comm"].get("mode", "full"))
    if requested is None:
        return policy_mode, policy_mode
    if (requested == "none") != (policy_mode == "none"):
        raise ValueError(
            f"Cannot evaluate a '{policy_mode}' checkpoint with comm mode '{requested}': "
            "'none' changes the action width and removes the message head, so it needs a "
            "separately trained checkpoint. Use 'null' to ablate the channel instead."
        )
    return policy_mode, requested


def evaluate(
    weights_path: Path,
    comm_mode: str | None = None,
    episodes: int | None = None,
    deterministic: bool = True,
    seeds: list[int] | None = None,
) -> dict:
    cfg = load_config()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = cfg_for_checkpoint(cfg, checkpoint)

    policy_mode, delivery = resolve_comm_modes(cfg, checkpoint, comm_mode)
    cfg["comm"]["mode"] = policy_mode
    cfg["comm"]["delivery"] = delivery

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, _ = make_swarm_env(cfg, num_envs=1, device=str(device))
    scenario = env.scenario
    actor, _ = load_policy(weights_path, cfg, device)

    eval_cfg = cfg["eval"]
    if seeds is None:
        seeds = eval_cfg["seeds"][: eval_cfg["num_seeds"]]
    episodes_per_seed = episodes or eval_cfg["episodes_per_seed"]

    times, coverages, aucs, lengths, curves = [], [], [], [], []
    for seed in seeds:
        for ep in range(episodes_per_seed):
            stats, _ = run_episode(
                env,
                actor,
                scenario,
                cfg,
                device,
                seed=seed * 1000 + ep,
                deterministic=deterministic,
                collect_trajectory=False,
            )
            times.append(stats["time_to_threshold"])
            coverages.append(stats["final_coverage"])
            lengths.append(stats["steps"])
            curves.append(stats["coverage_curve"])
            aucs.append(_coverage_auc(stats["coverage_curve"], stats["max_steps"]))

    max_steps = cfg["env"]["episode_horizon"]
    results = {
        "metric": "final_coverage",
        "num_episodes": len(times),
        "comm_mode": policy_mode,
        "comm_delivery": delivery,
        "deterministic": deterministic,
        "final_coverage": _summary(coverages),
        "coverage_auc": _summary(aucs),
        "episode_steps": _summary(lengths),
        "thresholds": _threshold_stats(curves, max_steps),
        # Retained so existing bundle scripts keep working. This metric is
        # censored at the horizon whenever the policy never reaches 90%.
        "mean_time_to_threshold": float(np.mean(times)),
        "std_time_to_threshold": float(np.std(times)),
        "mean_final_coverage": float(np.mean(coverages)),
        "legacy_metric": cfg["eval"]["metric"],
        "legacy_censored_fraction": float(np.mean([t == max_steps for t in times])),
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
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        cfg = cfg_for_checkpoint(cfg, checkpoint)
        policy_mode, delivery = resolve_comm_modes(cfg, checkpoint, args.comm_mode)
        cfg["comm"]["mode"] = policy_mode
        cfg["comm"]["delivery"] = delivery
        device = torch.device("cpu")
        env, _ = make_swarm_env(cfg, num_envs=1, device="cpu")
        actor, _ = load_policy(args.weights, cfg, device)
        _, traj = run_episode(
            env, actor, env.scenario, cfg, device, seed=0, deterministic=deterministic
        )
        args.export.write_text(json.dumps(traj))
        print(f"Exported rollout to {args.export}")
    else:
        results = evaluate(args.weights, args.comm_mode, args.episodes, deterministic=deterministic)
        print(json.dumps(results, indent=2))
