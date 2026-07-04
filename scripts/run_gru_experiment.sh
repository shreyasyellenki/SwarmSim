#!/usr/bin/env bash
# Experiment 5: recurrent (GRU) actor with per-episode hidden state, so a
# deterministic policy can condition on history (where it's been), not just the
# current obs. Isolates GRU as the single variable vs the global-map run (12.8%):
# both have gamma=0.3 + global map ON; this adds GRU. See STORYLINE.md.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=300000
NUM_ENVS=4
ROLLOUT=128
GAMMA=0.3
NAME=swarm_policy_full_gru

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Training full, gamma=$GAMMA, GRU actor ON -> ${NAME}.pt ==="
$PY -m swarmsim.policy.train_swarm \
  --comm-mode full \
  --timesteps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT" \
  --gamma "$GAMMA" \
  --use-gru \
  --save-name "$NAME"
log "=== Finished training ==="

log "=== Evaluating (deterministic) ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

weights = Path("weights/${NAME}.pt")
res = evaluate(weights, comm_mode="full", deterministic=True)
out = Path("weights/gru_experiment_results.json")
out.write_text(json.dumps({"gru": {"revisit_gamma": ${GAMMA}, "weights": str(weights), "eval": res}}, indent=2))
print(f"Results written to {out}")
print(f"  gru: coverage={res['mean_final_coverage']:.2%}, time_to_90%={res['mean_time_to_threshold']:.0f}")
PY

log "=== DONE ==="
