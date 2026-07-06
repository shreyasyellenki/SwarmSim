#!/usr/bin/env bash
# Bundle D: Bundle A + gentle delayed std anneal (start 150k, final std 0.7), 250k steps.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=250000
NUM_ENVS=4
ROLLOUT=256
GAMMA=0.3
CURIOSITY=0.3
REPULSION=0.05
ANNEAL_START=150000
STD_FINAL=0.7
NAME=swarm_policy_full_bundle_d

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle D: Bundle A + std anneal (start=${ANNEAL_START}, final=${STD_FINAL}), ${STEPS} steps ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --gamma "$GAMMA" \
  --use-gru \
  --no-global-map \
  --curiosity "$CURIOSITY" \
  --repulsion "$REPULSION" \
  --std-anneal \
  --std-anneal-start "$ANNEAL_START" \
  --std-final "$STD_FINAL" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating deterministic + stochastic ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
det = evaluate(weights, comm_mode="full", deterministic=True)
stoch = evaluate(weights, comm_mode="full", deterministic=False)
out = Path("weights/bundle_d_experiment_results.json")
out.write_text(json.dumps({
    "bundle_d": {
        "revisit_gamma": ${GAMMA},
        "curiosity": ${CURIOSITY},
        "repulsion": ${REPULSION},
        "std_anneal_start": ${ANNEAL_START},
        "std_final": ${STD_FINAL},
        "global_map": False,
        "weights": str(weights),
        "eval_deterministic": det,
        "eval_stochastic": stoch,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  deterministic: {det['mean_final_coverage']:.2%}")
print(f"  stochastic:    {stoch['mean_final_coverage']:.2%}")
print(f"  gap:           {stoch['mean_final_coverage'] - det['mean_final_coverage']:.2%}")
PY

log "=== DONE ==="
