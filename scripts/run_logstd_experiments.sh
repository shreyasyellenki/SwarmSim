#!/usr/bin/env bash
# Experiment 2: hold revisit_gamma=0.3, vary initial movement log_std to reduce
# reliance on action noise. Compare deterministic eval against the gamma=0.3
# baseline (init_log_std=0.0). See STORYLINE.md.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=300000
NUM_ENVS=4
ROLLOUT=128
GAMMA=0.3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# init_log_std -0.7 => std~0.50, -1.6 => std~0.20 (baseline 0.0 => std=1.0)
for ils in -0.7 -1.6; do
  tag=$(echo "$ils" | tr -d '.-')
  name="swarm_policy_full_ils${tag}"
  log "=== Training full, gamma=$GAMMA, init_log_std=$ils -> ${name}.pt ==="
  $PY -m swarmsim.policy.train_swarm \
    --comm-mode full \
    --timesteps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT" \
    --gamma "$GAMMA" \
    --init-log-std "$ils" \
    --save-name "$name"
  log "=== Finished init_log_std=$ils ==="
done

log "=== Evaluating both checkpoints (deterministic) ==="
$PY - <<'PY'
import json
from pathlib import Path

from swarmsim.policy.eval import evaluate

results = {}
for ils in (-0.7, -1.6):
    tag = str(ils).replace(".", "").replace("-", "")
    weights = Path(f"weights/swarm_policy_full_ils{tag}.pt")
    results[f"init_log_std_{ils}"] = {
        "init_log_std": ils,
        "revisit_gamma": 0.3,
        "weights": str(weights),
        "eval": evaluate(weights, comm_mode="full", deterministic=True),
    }

out = Path("weights/logstd_experiment_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"Results written to {out}")
for key, res in results.items():
    ev = res["eval"]
    print(
        f"  {key}: coverage={ev['mean_final_coverage']:.2%}, "
        f"time_to_90%={ev['mean_time_to_threshold']:.0f} steps"
    )
PY

log "=== DONE ==="
