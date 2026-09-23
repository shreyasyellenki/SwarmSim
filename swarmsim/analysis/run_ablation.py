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
        # No fallback to a generic checkpoint: evaluating one policy under
        # three labels silently fabricates an ablation.
        if not weights.exists():
            print(f"Skipping {mode}: no checkpoint at {weights}")
            continue
        results[mode] = evaluate(weights, comm_mode=mode, episodes=args.episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Ablation results written to {args.output}")
    for mode, res in results.items():
        cov = res["final_coverage"]
        auc = res["coverage_auc"]
        print(
            f"  {mode}: coverage={cov['mean']:.2%} ±{cov['ci95']:.2%} (95% CI, n={cov['n']}), "
            f"auc={auc['mean']:.3f}"
        )
    if len(results) > 1:
        print(
            "\nNote: these are single-training-run conditions. Differences smaller than the "
            "seed-to-seed spread are not evidence. See scripts/run_multiseed.py."
        )


if __name__ == "__main__":
    main()
