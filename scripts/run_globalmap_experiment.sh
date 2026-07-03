#!/usr/bin/env bash
# Experiment 3 (Path A): give the actor a coarse global coverage map (8x8) in its
# observation, holding gamma=0.3. Train full-comm at 300k, then deterministic eval.
# Compare against gamma=0.3 baseline (14.9% coverage, local-obs only). See STORYLINE.md.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=300000
NUM_ENVS=4
ROLLOUT=128
GAMMA=0.3
NAME=swarm_policy_full_globalmap

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Training full, gamma=$GAMMA, global map ON -> ${NAME}.pt ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --gamma "$GAMMA" \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating (deterministic) ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
res = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/globalmap_experiment_results.json")
out.write_text(json.dumps({"global_map": {"revisit_gamma": ${GAMMA}, "weights": str(weights), "eval": res}}, indent=2))
print(f"Results written to {out}")
print(f"  global_map: coverage={res['mean_final_coverage']:.2%}, time_to_90%={res['mean_time_to_threshold']:.0f}")
PY

log "=== DONE ==="
