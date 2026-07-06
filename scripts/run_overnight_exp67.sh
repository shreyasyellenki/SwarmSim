#!/usr/bin/env bash
# Overnight Experiments 6 + 7. See STORYLINE.md for results.
#
# Exp 6: GRU + global map + gamma=0.3 + std/entropy anneal @ 1M steps
# Exp 7: same stack + spread reward (per-agent discovery credit) + gamma=0.5 @ 1M steps
#
# NaN safeguards (prior 1M runs crashed ~470k without anneal):
#   - std/entropy anneal forces late training to optimize the mean policy
#   - PPO advantage std-floor, non-finite guards, checkpoint rollback (ppo.py)
#   - Periodic checkpoint saves every save_interval updates
#
# Override: OVERNIGHT_STEPS=500000 bash scripts/run_overnight_exp67.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS="${OVERNIGHT_STEPS:-1000000}"
NUM_ENVS=4
ROLLOUT=128
OUT=weights/overnight_exp67_results.json

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

COMMON=(
  --comm-mode full
  --timesteps "$STEPS"
  --num-envs "$NUM_ENVS"
  --rollout-steps "$ROLLOUT"
  --use-gru
  --std-anneal
  --entropy-anneal
)

log "=== Exp 6: GRU + anneal, gamma=0.3, team reward, ${STEPS} steps ==="
$PY -m swarmsim.policy.train_swarm \
  "${COMMON[@]}" \
  --gamma 0.3 \
  --save-name swarm_policy_full_gru_anneal_1m
log "=== Exp 6 training done ==="

log "=== Exp 7: GRU + anneal + spread reward, gamma=0.5, ${STEPS} steps ==="
$PY -m swarmsim.policy.train_swarm \
  "${COMMON[@]}" \
  --gamma 0.5 \
  --reward-mode spread \
  --save-name swarm_policy_full_gru_spread_1m
log "=== Exp 7 training done ==="

log "=== Evaluating both (deterministic, full protocol) ==="
$PY - <<PY
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

results = {}
for key, weights, gamma in [
    ("exp6_gru_anneal", "weights/swarm_policy_full_gru_anneal_1m.pt", 0.3),
    ("exp7_gru_spread", "weights/swarm_policy_full_gru_spread_1m.pt", 0.5),
]:
    w = Path(weights)
    print(f"Evaluating {key} ...", flush=True)
    results[key] = {
        "revisit_gamma": gamma,
        "timesteps": ${STEPS},
        "weights": str(w),
        "eval": evaluate(w, comm_mode="full", deterministic=True),
    }

out = Path("${OUT}")
out.write_text(json.dumps(results, indent=2))
print(f"Results written to {out}")
for key, res in results.items():
    ev = res["eval"]
    print(f"  {key}: coverage={ev['mean_final_coverage']:.2%}, time_to_90%={ev['mean_time_to_threshold']:.0f}")
PY

log "=== ALL DONE ==="
