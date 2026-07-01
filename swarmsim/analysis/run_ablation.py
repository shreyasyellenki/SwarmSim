"""Run communication ablation evaluation across all three conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from swarmsim.policy.eval import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed (default from config)")
    parser.add_argument("--output", type=Path, default=Path("weights/ablation_results.json"))
    args = parser.parse_args()

    results = {}
    for mode in ("none", "null", "full"):
        weights = args.weights_dir / f"swarm_policy_{mode}.pt"
        if not weights.exists():
            weights = args.weights_dir / "swarm_policy.pt"
        if not weights.exists():
            print(f"Skipping {mode}: weights not found at {weights}")
            continue
        results[mode] = evaluate(weights, comm_mode=mode, episodes=args.episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Ablation results written to {args.output}")
    for mode, res in results.items():
        print(
            f"  {mode}: mean_time={res['mean_time_to_threshold']:.1f} "
            f"(±{res['std_time_to_threshold']:.1f}), coverage={res['mean_final_coverage']:.2%}"
        )


if __name__ == "__main__":
    main()
