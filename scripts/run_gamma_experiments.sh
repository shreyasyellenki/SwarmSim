#!/usr/bin/env bash
# Train full-comm policies with revisit_gamma=0.3 vs 0.5, then eval and compare.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=300000
NUM_ENVS=4
ROLLOUT=128
OUT=weights/gamma_experiment_results.json

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for gamma in 0.3 0.5; do
  tag=$(echo "$gamma" | tr -d '.')
  name="swarm_policy_full_gamma${tag}"
  log "=== Training full, revisit_gamma=$gamma -> ${name}.pt ==="
  $PY -m swarmsim.policy.train_swarm \
    --comm-mode full \
    --timesteps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT" \
    --gamma "$gamma" \
    --save-name "$name"
  log "=== Finished training gamma=$gamma ==="
done

log "=== Evaluating both checkpoints (deterministic) ==="
$PY - <<'PY'
import json
from pathlib import Path

from swarmsim.policy.eval import evaluate

results = {}
for gamma in (0.3, 0.5):
    tag = str(gamma).replace(".", "")
    weights = Path(f"weights/swarm_policy_full_gamma{tag}.pt")
    results[f"gamma_{gamma}"] = {
        "revisit_gamma": gamma,
        "weights": str(weights),
        "eval": evaluate(weights, comm_mode="full", deterministic=True),
    }

out = Path("weights/gamma_experiment_results.json")
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
