#!/usr/bin/env bash
# Experiment 4: training-pressure fix — anneal log_std (1.0 -> 0.2) and decay entropy.
# Holds gamma=0.3 and global map ON. Compare vs gamma03 baseline (14.9%). See STORYLINE.md.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=300000
NUM_ENVS=4
ROLLOUT=128
GAMMA=0.3
NAME=swarm_policy_full_stdanneal

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Training full, gamma=$GAMMA, std+entropy anneal -> ${NAME}.pt ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --gamma "$GAMMA" \
  --std-anneal \
  --entropy-anneal \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating (deterministic) ==="
$PY - <<PY
import json
import torch
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
ckpt = torch.load(weights, map_location="cpu", weights_only=False)
log_std = ckpt["actor"]["log_std"].tolist()
std = torch.exp(torch.tensor(log_std)).tolist()
res = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/stdanneal_experiment_results.json")
payload = {
    "std_anneal": {
        "revisit_gamma": ${GAMMA},
        "weights": str(weights),
        "final_log_std": log_std,
        "final_std": std,
        "eval": res,
    }
}
out.write_text(json.dumps(payload, indent=2))
print(f"Results written to {out}")
print(f"  final_log_std={log_std}, final_std={[round(s,3) for s in std]}")
print(f"  coverage={res['mean_final_coverage']:.2%}, time_to_90%={res['mean_time_to_threshold']:.0f}")
PY

log "=== DONE ==="
