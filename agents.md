# SwarmSim — Agent Context (start here)

> Copy-paste the session prompt at the bottom into new Cursor chats.

**Design doc:** [`SwarmSim_Altered_Design_Doc.md`](SwarmSim_Altered_Design_Doc.md)  
**Private experiment log:** `STORYLINE.md` (gitignored — coverage numbers, ablation results, resume notes)  
**Stage 1 (optional / sidelined):** [`agents_stage_1.md`](agents_stage_1.md)

---

## What this project is

Multi-agent RL swarm that explores a 2D grid with **learned inter-agent communication**, visualized live in the browser.

- **Stage 1 (sidelined):** MuJoCo single-agent waypoint nav — learning only, not used in demo
- **Stage 2 (active):** VMAS 6-agent swarm + custom PPO + 8-dim message vectors
- **Demo:** VMAS rollout → FastAPI WebSocket → Three.js (`DataTexture` grid)

---

## Setup

```bash
cd SwarmSim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All hyperparameters: [`swarmsim/config.yaml`](swarmsim/config.yaml)

---

## Work completed

### Infrastructure
- [x] Repo scaffold, `requirements.txt`, `.venv` workflow, `README.md`
- [x] `config.yaml` — Stage 1 + Stage 2 + PPO + eval + demo settings
- [x] `Dockerfile`, `scripts/run_demo.sh` (executable in git)
- [x] `agents_stage_1.md` — MuJoCo context for optional future sessions

### Stage 1 — MuJoCo (sidelined, code kept)
- [x] `swarmsim/env/mujoco_drone.xml` — MJCF quadrotor
- [x] `swarmsim/env/mujoco_nav.py` — Gymnasium waypoint env
- [x] `swarmsim/policy/train_mujoco.py` — single-agent PPO training

### Stage 2 — VMAS swarm
- [x] `swarmsim/env/swarm_env.py` — grid exploration, comm, MAPPO global state, reward shaping
- [x] `swarmsim/policy/network.py` — `SwarmActor` (optional GRU), `CentralizedCritic`
- [x] `swarmsim/policy/ppo.py` — custom PPO + `SwarmRolloutBuffer`
- [x] `swarmsim/policy/train_swarm.py` — multi-agent training CLI
- [x] `swarmsim/policy/eval.py` — rollout eval (deterministic by default)

### Demo / visualizer
- [x] `swarmsim/server/main.py` — decoupled sim loop + WebSocket
- [x] `swarmsim/visualizer/` — Three.js grid, agents, comm links
- [x] Checkpoint-aware demo loading (`demo.weights`, `SWARMSIM_WEIGHTS`)

### Reward & policy features
- [x] Count-based curiosity (`--curiosity`), frontier bonus (`--frontier`), repulsion (`--repulsion`)
- [x] Heading diversity penalty (`--diversity`), message heading aux loss (`--message-heading-aux`)
- [x] Delayed std anneal (`--std-anneal`, `--std-anneal-start`, `--std-final`)
- [x] Local-only actor obs (`--no-global-map`), GRU actor (`--use-gru`)
- [x] Experiment scripts: `scripts/run_bundle_*.sh`

### Analysis
- [x] `swarmsim/analysis/run_ablation.py` — none/null/full comm comparison
- [x] `swarmsim/analysis/comm_analysis.py` — message–state correlation heatmap

---

## After each experiment run (dev workflow)

**Do not put coverage numbers or ablation statistics in committed docs** (`README.md`, `agents.md`, design doc). Record those in **`STORYLINE.md`** (local, gitignored).

1. Update **`STORYLINE.md`** — hypothesis, metrics, lesson learned, demo weights if changed.
2. Commit code/scripts only; experiment result JSON stays local (see `.gitignore`).
3. Push when appropriate.

```bash
git add swarmsim/ scripts/ agents.md README.md
git commit -m "Add frontier reward and bundle F training script"
git push origin main
```

---

## Work remaining (directional — no metrics here)

- [ ] Comm ablation at current best reward config
- [ ] Coordination levers (diversity tuning, message aux loss, init heading) on best base config
- [ ] Optional gentle std anneal after movement policy stabilizes
- [ ] README polish (concept + how-to only; no results tables)

---

## Key files map

| Path | Role |
|------|------|
| `swarmsim/config.yaml` | All hyperparameters |
| `swarmsim/env/swarm_env.py` | VMAS scenario + reward |
| `swarmsim/policy/train_swarm.py` | Stage 2 training |
| `swarmsim/policy/ppo.py` | PPO implementation |
| `swarmsim/server/main.py` | Live demo server |
| `swarmsim/visualizer/scene.js` | Three.js rendering |
| `weights/*.pt` | Trained checkpoints (gitignored) |

---

## Architecture

```
VMAS train → swarm_policy_{full|null|none}.pt
                 ↓
          server/main.py (12 Hz sim)
                 ↓ WebSocket 30 Hz
          visualizer/ (Three.js)
```

---

## Communication ablation

| Mode | Config flag | Meaning |
|------|-------------|---------|
| `none` | `comm.mode: none` | No message head |
| `null` | `comm.mode: null` | Messages emitted but zeroed in obs |
| `full` | `comm.mode: full` | Full emergent comm |

---

## Actor observation

**59 dims (local-only):** `[norm_pos(2), norm_vel(2), local_5x5(25), neighbor_rel(6), incoming_msgs(24)]`

**123 dims (with global map):** above + `global_map(64)` when `env.global_map_downsample: 8`

Use `--no-global-map` for local-only obs. Demo/eval infer layout from checkpoint metadata.

---

## Reward (default `team_new_cells`)

```
reward = alpha * new_cells - gamma * revisit + delta * curiosity + frontier * frontier_frac
         - repulsion * proximity - diversity * heading_alignment
```

Optional PPO aux loss: `--message-heading-aux` supervises `msg[:2]` ≈ velocity heading (full comm only).

Team rewards are averaged across agents in `train_swarm.py`.

---

## Common commands

```bash
source .venv/bin/activate

# Train (example — see config.yaml and scripts/run_bundle_*.sh)
python -m swarmsim.policy.train_swarm --comm-mode full --use-gru --no-global-map \
  --curiosity 0.3 --frontier 0.2 --repulsion 0.05 \
  --timesteps 250000 --num-envs 4 --rollout-steps 256

# Demo
bash scripts/run_demo.sh
SWARMSIM_WEIGHTS=swarm_policy_full_bundle_f.pt bash scripts/run_demo.sh

# Eval
python -m swarmsim.policy.eval --weights weights/swarm_policy_full_bundle_f.pt

# Ablation
python -m swarmsim.analysis.run_ablation --weights-dir weights

# Comm analysis
python -m swarmsim.analysis.comm_analysis --weights weights/swarm_policy_full.pt
```

---

## Gotchas

1. **`.gitignore`:** `ENV/` matches `swarmsim/env/` on macOS — exceptions added as `!swarmsim/env/**`
2. **Demo weights:** `demo.weights` in config or `SWARMSIM_WEIGHTS` env; server matches checkpoint obs dim
3. **Grid bytes:** NumPy `explored[cx, cy]` C-order; frontend must flip Y to match `normToWorld()`
4. **Experiment metrics:** keep in `STORYLINE.md` / local `weights/*_results.json`, not in public docs

---

## Cursor session prompt

> We are building SwarmSim — multi-agent VMAS swarm with custom PPO and learned communication, live Three.js demo. Read `agents.md` and `SwarmSim_Altered_Design_Doc.md`. Stage 1 MuJoCo is sidelined (`agents_stage_1.md`). Private experiment history: `STORYLINE.md`. Active focus: [describe task]. Use `.venv`.
