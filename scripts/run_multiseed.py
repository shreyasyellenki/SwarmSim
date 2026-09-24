#!/usr/bin/env python
"""Train one checkpoint per seed, evaluate each, and aggregate across seeds.

A single training run tells you almost nothing: seed-to-seed spread in this
project has been comparable to the effect sizes being claimed. This script
makes the seed the unit of analysis, which is the level at which two
configurations can actually be compared.

Usage:
    python scripts/run_multiseed.py --name bundle_f --seeds 0 1 2 3 4 --parallel 4
    python scripts/run_multiseed.py --name bundle_f --seeds 0 1 2 --eval-only

Training flags default to the Bundle F recipe; override by passing them after
``--``:
    python scripts/run_multiseed.py --name exp --seeds 0 1 -- --frontier 0.4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BUNDLE_F_ARGS = [
    "--comm-mode", "full",
    "--use-gru",
    "--no-global-map",
    "--curiosity", "0.3",
    "--frontier", "0.2",
    "--repulsion", "0.05",
    "--timesteps", "250000",
    "--num-envs", "4",
    "--rollout-steps", "256",
]


def train_one(args_tuple) -> tuple[int, str, bool, str]:
    seed, name, train_args, threads = args_tuple
    stem = f"{name}_s{seed}"
    cmd = [sys.executable, "-m", "swarmsim.policy.train_swarm",
           *train_args, "--seed", str(seed), "--save-name", stem]
    env = dict(os.environ)
    # Parallel runs oversubscribe the machine unless each is single-threaded.
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    log = REPO / "logs" / f"{stem}.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return seed, stem, proc.returncode == 0, str(log)


# Two-sided 95% t critical values by degrees of freedom. A normal 1.96 would
# badly understate the interval at the handful of seeds we can afford.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def _t95(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T95:
        return _T95[df]
    return min((v for k, v in _T95.items() if k >= df), default=1.96)


def across_seed_summary(per_seed: list[float]) -> dict:
    """Mean and CI over seeds -- the correct unit for comparing configurations."""
    arr = np.asarray(per_seed, dtype=float)
    n = arr.size
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n else 0.0
    return {
        "mean": float(arr.mean()) if n else float("nan"),
        "std": std,
        "sem": sem,
        "ci95": _t95(n - 1) * sem if n > 1 else float("inf"),
        "t_crit": _t95(n - 1) if n > 1 else None,
        "min": float(arr.min()) if n else float("nan"),
        "max": float(arr.max()) if n else float("nan"),
        "n_seeds": int(n),
        "per_seed": [float(v) for v in arr],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Checkpoint stem; files become <name>_s<seed>.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent training runs")
    parser.add_argument("--threads", type=int, default=1, help="Torch threads per run")
    parser.add_argument("--eval-seeds", type=int, default=10, help="Eval seeds per checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per eval seed")
    parser.add_argument("--delivery", nargs="+", default=["full", "null"],
                        help="Channel delivery modes to evaluate per checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_args = [a for a in args.train_args if a != "--"] or BUNDLE_F_ARGS
    weights_dir = REPO / "weights"
    out_path = args.output or weights_dir / f"{args.name}_multiseed.json"

    if not args.eval_only:
        jobs = [(s, args.name, train_args, args.threads) for s in args.seeds]
        started = time.time()
        print(f"Training {len(jobs)} seeds, {args.parallel} at a time "
              f"({args.threads} thread(s) each)...", flush=True)
        results = []
        if args.parallel > 1:
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = [pool.submit(train_one, j) for j in jobs]
                for fut in as_completed(futures):
                    results.append(fut.result())
                    seed, stem, ok, log = results[-1]
                    print(f"  [{time.time() - started:7.0f}s] seed {seed}: "
                          f"{'ok' if ok else 'FAILED'}  ({stem}.pt)", flush=True)
        else:
            for j in jobs:
                results.append(train_one(j))
                seed, stem, ok, log = results[-1]
                print(f"  [{time.time() - started:7.0f}s] seed {seed}: "
                      f"{'ok' if ok else 'FAILED'}  ({stem}.pt)", flush=True)
        if not all(ok for _, _, ok, _ in results):
            for seed, stem, ok, log in results:
                if not ok:
                    print(f"  seed {seed} log: {log}")
            print("At least one run failed; aborting before aggregation.")
            return 1

    from swarmsim.policy.eval import evaluate

    eval_seeds = list(range(args.eval_seeds))
    per_mode: dict[str, dict[int, dict]] = {m: {} for m in args.delivery}
    for seed in args.seeds:
        ckpt = weights_dir / f"{args.name}_s{seed}.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: missing {ckpt}, skipping")
            continue
        for mode in args.delivery:
            res = evaluate(ckpt, comm_mode=mode, episodes=args.episodes,
                           deterministic=True, seeds=eval_seeds)
            per_mode[mode][seed] = res
            print(f"  seed {seed} delivery={mode}: "
                  f"coverage={res['final_coverage']['mean']:.4f} "
                  f"auc={res['coverage_auc']['mean']:.4f}", flush=True)

    report = {"name": args.name, "train_args": train_args, "train_seeds": args.seeds,
              "eval_seeds": eval_seeds, "episodes_per_eval_seed": args.episodes,
              "conditions": {}}
    for mode, by_seed in per_mode.items():
        if not by_seed:
            continue
        report["conditions"][mode] = {
            "final_coverage": across_seed_summary([r["final_coverage"]["mean"] for r in by_seed.values()]),
            "coverage_auc": across_seed_summary([r["coverage_auc"]["mean"] for r in by_seed.values()]),
            "per_seed_detail": {str(s): r for s, r in by_seed.items()},
        }

    # Paired difference: same seed under both delivery modes removes the
    # seed-to-seed variance that otherwise swamps the comparison.
    if len(args.delivery) == 2:
        a, b = args.delivery
        shared = sorted(set(per_mode[a]) & set(per_mode[b]))
        if shared:
            diffs = [per_mode[a][s]["final_coverage"]["mean"] - per_mode[b][s]["final_coverage"]["mean"]
                     for s in shared]
            report["paired_difference"] = {
                "comparison": f"{a} - {b}",
                "final_coverage": across_seed_summary(diffs),
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nWrote {out_path}\n")
    print(f"{'condition':<12} {'coverage (mean over seeds)':<32} {'auc':<12}")
    for mode, block in report["conditions"].items():
        c, u = block["final_coverage"], block["coverage_auc"]
        print(f"{mode:<12} {c['mean']:.4f} +-{c['ci95']:.4f} (n={c['n_seeds']} seeds)   {u['mean']:.4f}")
    if "paired_difference" in report:
        d = report["paired_difference"]["final_coverage"]
        sig = d["n_seeds"] > 1 and abs(d["mean"]) > d["ci95"]
        print(f"\npaired {report['paired_difference']['comparison']}: "
              f"{d['mean']:+.4f} +-{d['ci95']:.4f}  "
              f"-> {'distinguishable from zero' if sig else 'NOT distinguishable from zero'}")
    print("\nQuote the across-seed CI, not the per-episode CI: episodes within a seed "
          "share one policy and understate uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
