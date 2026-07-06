#!/usr/bin/env bash
# Bundle F: frontier reward + curiosity + repulsion, no anneal, 250k steps.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=250000
NUM_ENVS=4
ROLLOUT=256
GAMMA=0.3
CURIOSITY=0.3
FRONTIER=0.2
REPULSION=0.05
NAME=swarm_policy_full_bundle_f

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle F: curiosity=${CURIOSITY} frontier=${FRONTIER} repulsion=${REPULSION}, ${STEPS} steps ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --gamma "$GAMMA" \
  --use-gru \
  --no-global-map \
  --curiosity "$CURIOSITY" \
  --frontier "$FRONTIER" \
  --repulsion "$REPULSION" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating deterministic ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
det = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/bundle_f_experiment_results.json")
out.write_text(json.dumps({
    "bundle_f": {
        "revisit_gamma": ${GAMMA},
        "curiosity": ${CURIOSITY},
        "frontier": ${FRONTIER},
        "repulsion": ${REPULSION},
        "global_map": False,
        "weights": str(weights),
        "eval_deterministic": det,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  deterministic: {det['mean_final_coverage']:.2%}")
PY

log "=== DONE ==="
