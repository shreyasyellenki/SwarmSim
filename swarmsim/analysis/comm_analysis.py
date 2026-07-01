"""Correlate learned message components with agent state variables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from swarmsim.env.swarm_env import load_config, make_swarm_env
from swarmsim.policy.eval import load_policy, run_episode


def collect_message_data(cfg: dict, weights_path: Path, num_episodes: int = 10) -> dict:
    device = torch.device("cpu")
    env, _ = make_swarm_env(cfg, num_envs=1, device="cpu")
    scenario = env.scenario
    actor, _ = load_policy(weights_path, cfg, device)

    message_dim = cfg["comm"]["message_dim"]
    records = {f"msg_{i}": [] for i in range(message_dim)}
    records.update({"heading": [], "speed": [], "local_unexplored": [], "new_cell": []})

    for ep in range(num_episodes):
        obs_list = env.reset(seed=ep)
        for step in range(cfg["env"]["episode_horizon"]):
            actions = []
            with torch.no_grad():
                for agent_idx in range(cfg["env"]["num_agents"]):
                    move, message, _, _ = actor.act(obs_list[agent_idx].to(device))
                    if message is None:
                        actions.append(move)
                    else:
                        actions.append(torch.cat([move, message], dim=-1))

            obs_list, _, dones, _ = env.step(actions)

            for i, agent in enumerate(scenario.world.agents):
                vel = agent.state.vel[0].cpu().numpy()
                heading = float(np.arctan2(vel[1], vel[0]))
                speed = float(np.linalg.norm(vel[:2]))
                msg = scenario.outgoing_messages[0, i].cpu().numpy()
                local_patch = scenario._local_patch(agent)[0].cpu().numpy()
                local_unexplored = float(1.0 - local_patch.mean())
                new_cell = float(scenario.new_cells[0].item())

                records["heading"].append(heading)
                records["speed"].append(speed)
                records["local_unexplored"].append(local_unexplored)
                records["new_cell"].append(new_cell)
                for d in range(message_dim):
                    records[f"msg_{d}"].append(float(msg[d]))

            if bool(dones[0].any().item() if isinstance(dones, list) else dones.any().item()):
                break

    return records


def compute_correlations(records: dict) -> dict:
    state_keys = ["heading", "speed", "local_unexplored", "new_cell"]
    message_keys = [k for k in records if k.startswith("msg_")]
    corr = {}
    for mk in message_keys:
        corr[mk] = {}
        for sk in state_keys:
            a = np.array(records[mk])
            b = np.array(records[sk])
            if a.std() < 1e-8 or b.std() < 1e-8:
                corr[mk][sk] = 0.0
            else:
                corr[mk][sk] = float(np.corrcoef(a, b)[0, 1])
    return corr


def plot_correlations(corr: dict, output_path: Path):
    message_keys = sorted(corr.keys())
    state_keys = ["heading", "speed", "local_unexplored", "new_cell"]
    matrix = np.array([[corr[mk][sk] for sk in state_keys] for mk in message_keys])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(state_keys)))
    ax.set_xticklabels(state_keys, rotation=30, ha="right")
    ax.set_yticks(range(len(message_keys)))
    ax.set_yticklabels(message_keys)
    ax.set_title("Message Component vs Agent State Correlations")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/swarm_policy_full.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    args = parser.parse_args()

    cfg = load_config()
    if not args.weights.exists():
        print(f"Weights not found: {args.weights}. Run train_swarm.py first.")
        return

    records = collect_message_data(cfg, args.weights, args.episodes)
    corr = compute_correlations(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "correlations.json").write_text(json.dumps(corr, indent=2))
    plot_correlations(corr, args.output_dir / "correlation_heatmap.png")
    print(f"Analysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
