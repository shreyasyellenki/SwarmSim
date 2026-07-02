#!/usr/bin/env bash
# Communication ablation: train none/null/full back-to-back, then eval.
# Default 300k steps — long enough for fair comparison and avoids late-training NaNs
# seen on 1M none runs. Override with ABLATION_STEPS=1000000 if desired.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
STEPS="${ABLATION_STEPS:-300000}"
NUM_ENVS=4
ROLLOUT=128

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Preserve the current 300k full weights before the 1M run overwrites them.
if [ -f weights/swarm_policy_full.pt ]; then
  cp weights/swarm_policy_full.pt weights/swarm_policy_full_300k.pt
  log "Backed up existing full weights -> weights/swarm_policy_full_300k.pt"
fi

for mode in none null full; do
  weights="weights/swarm_policy_${mode}.pt"
  if [ -f "$weights" ]; then
    cp "$weights" "weights/swarm_policy_${mode}_partial.pt"
    rm -f "$weights"
    log "Removed partial checkpoint for '$mode' (backed up as *_partial.pt)"
  fi
  log "=== Training '$mode' for $STEPS steps ==="
  $PY -m swarmsim.policy.train_swarm \
    --comm-mode "$mode" \
    --timesteps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT"
  log "=== Finished training '$mode' ==="
done

log "=== Running ablation eval (50 seeds x 20 episodes) ==="
$PY -m swarmsim.analysis.run_ablation --weights-dir weights --output weights/ablation_results.json

log "=== ALL DONE. Results in weights/ablation_results.json ==="
