#!/usr/bin/env bash
# Bundle N: F config + 7x7 local observation window, 250k steps.
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
REPULSION_RADIUS=3
LOCAL_WINDOW=7
NAME=swarm_policy_full_bundle_n

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle N: F config + local window ${LOCAL_WINDOW}x${LOCAL_WINDOW}, ${STEPS} steps ==="
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
  --repulsion-radius "$REPULSION_RADIUS" \
  --local-window "$LOCAL_WINDOW" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating deterministic ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
det = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/bundle_n_experiment_results.json")
out.write_text(json.dumps({
    "bundle_n": {
        "revisit_gamma": ${GAMMA},
        "curiosity": ${CURIOSITY},
        "frontier": ${FRONTIER},
        "repulsion": ${REPULSION},
        "repulsion_radius": ${REPULSION_RADIUS},
        "local_window_k": ${LOCAL_WINDOW},
        "global_map": False,
        "weights": str(weights),
        "eval_deterministic": det,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  deterministic: {det['mean_final_coverage']:.2%}")
PY

log "=== DONE ==="
