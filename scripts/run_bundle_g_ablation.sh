#!/usr/bin/env bash
# Comm ablation at Bundle G config + comm_analysis on full weights.
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
DIVERSITY=0.1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

FULL_WEIGHTS=weights/swarm_policy_full_bundle_g.pt
if [ ! -f "$FULL_WEIGHTS" ]; then
  echo "Missing $FULL_WEIGHTS — run scripts/run_bundle_g.sh first."
  exit 1
fi

for MODE in null none; do
  NAME="swarm_policy_${MODE}_bundle_g"
  log "=== Training comm-mode=${MODE} (Bundle G config) -> ${NAME}.pt ==="
  $PY -m swarmsim.policy.train_swarm \
    --comm-mode "$MODE" \
    --timesteps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT" \
    --gamma "$GAMMA" \
    --use-gru \
    --no-global-map \
    --curiosity "$CURIOSITY" \
    --frontier "$FRONTIER" \
    --repulsion "$REPULSION" \
    --diversity "$DIVERSITY" \
    --save-name "$NAME"
done

log "=== Evaluating full / null / none ==="
$PY - <<'PY'
import json
from pathlib import Path
from swarmsim.policy.eval import evaluate

modes = {
    "full": "weights/swarm_policy_full_bundle_g.pt",
    "null": "weights/swarm_policy_null_bundle_g.pt",
    "none": "weights/swarm_policy_none_bundle_g.pt",
}
results = {}
for mode, path in modes.items():
    w = Path(path)
    if not w.exists():
        print(f"Skip {mode}: {w} missing")
        continue
    results[mode] = evaluate(w, comm_mode=mode, deterministic=True)
    print(f"  {mode}: {results[mode]['mean_final_coverage']:.2%}")

out = Path("weights/bundle_g_ablation_results.json")
out.write_text(json.dumps({"bundle_g_ablation": results}, indent=2))
print(f"Written {out}")
PY

log "=== comm_analysis on full Bundle G weights ==="
$PY -m swarmsim.analysis.comm_analysis \
  --weights "$FULL_WEIGHTS" \
  --episodes 10 \
  --output-dir weights/bundle_g_comm_analysis

log "=== Ablation DONE ==="
