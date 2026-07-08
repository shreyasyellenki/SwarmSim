#!/usr/bin/env bash
# Bundle K: Bundle F + linear LR decay (3e-4 -> 3e-5), 250k steps.
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
LR_FINAL=0.00003
NAME=swarm_policy_full_bundle_k

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle K: F config + lr anneal -> ${LR_FINAL}, ${STEPS} steps ==="
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
  --lr-anneal \
  --lr-final "$LR_FINAL" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating deterministic ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
det = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/bundle_k_experiment_results.json")
out.write_text(json.dumps({
    "bundle_k": {
        "revisit_gamma": ${GAMMA},
        "curiosity": ${CURIOSITY},
        "frontier": ${FRONTIER},
        "repulsion": ${REPULSION},
        "lr_anneal": True,
        "lr_final": ${LR_FINAL},
        "global_map": False,
        "weights": str(weights),
        "eval_deterministic": det,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  deterministic: {det['mean_final_coverage']:.2%}")
PY

log "=== DONE ==="
