#!/usr/bin/env bash
# Bundle A: local-only obs + count-based curiosity + inter-agent repulsion.
# GRU actor, no anneal, 500k steps. Success gate: >40% deterministic coverage.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=500000
NUM_ENVS=4
ROLLOUT=128
GAMMA=0.3
CURIOSITY=0.3
REPULSION=0.05
NAME=swarm_policy_full_bundle_a

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle A: local obs + curiosity=$CURIOSITY + repulsion=$REPULSION, GRU, ${STEPS} steps ==="
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
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating (deterministic) ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
res = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/bundle_a_experiment_results.json")
out.write_text(json.dumps({
    "bundle_a": {
        "revisit_gamma": ${GAMMA},
        "curiosity": ${CURIOSITY},
        "repulsion": ${REPULSION},
        "global_map": False,
        "weights": str(weights),
        "eval": res,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  bundle_a: coverage={res['mean_final_coverage']:.2%}, time_to_90%={res['mean_time_to_threshold']:.0f}")
PY

log "=== DONE ==="
