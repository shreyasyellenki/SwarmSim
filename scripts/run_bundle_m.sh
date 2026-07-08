#!/usr/bin/env bash
# Bundle M: F config + LR decay (3e-4 -> 0) + revisit gamma curriculum (0.01 -> 0.5).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=250000
NUM_ENVS=4
ROLLOUT=256
CURIOSITY=0.3
FRONTIER=0.2
REPULSION=0.05
LR_FINAL=0
GAMMA_START=0.01
GAMMA_END=0.5
NAME=swarm_policy_full_bundle_m

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Bundle M: F + lr anneal -> ${LR_FINAL}, gamma ${GAMMA_START}->${GAMMA_END}, ${STEPS} steps ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --use-gru \
  --no-global-map \
  --curiosity "$CURIOSITY" \
  --frontier "$FRONTIER" \
  --repulsion "$REPULSION" \
  --lr-anneal \
  --lr-final "$LR_FINAL" \
  --gamma-anneal \
  --gamma-start "$GAMMA_START" \
  --gamma-end "$GAMMA_END" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating deterministic ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
det = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/bundle_m_experiment_results.json")
out.write_text(json.dumps({
    "bundle_m": {
        "curiosity": ${CURIOSITY},
        "frontier": ${FRONTIER},
        "repulsion": ${REPULSION},
        "lr_anneal": True,
        "lr_final": ${LR_FINAL},
        "gamma_anneal": True,
        "gamma_start": ${GAMMA_START},
        "gamma_end": ${GAMMA_END},
        "eval_revisit_gamma": ${GAMMA_END},
        "global_map": False,
        "weights": str(weights),
        "eval_deterministic": det,
    }
}, indent=2))
print(f"Results written to {out}")
print(f"  deterministic: {det['mean_final_coverage']:.2%}")
PY

log "=== DONE ==="
