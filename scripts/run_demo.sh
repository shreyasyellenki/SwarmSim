#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Optional: SWARMSIM_WEIGHTS=swarm_policy_full_gru.pt ./scripts/run_demo.sh
export SWARMSIM_WEIGHTS="${SWARMSIM_WEIGHTS:-}"

uvicorn swarmsim.server.main:app --host 0.0.0.0 --port 8000 --reload
