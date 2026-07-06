# SwarmSim — Agent Context (start here)

> Copy-paste the session prompt at the bottom into new Cursor chats.

**Design doc:** [`SwarmSim_Altered_Design_Doc.md`](SwarmSim_Altered_Design_Doc.md)  
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
- **Not required for demo.** Weights: `weights/mujoco_policy.pt` (gitignored)

### Stage 2 — VMAS swarm (headline deliverable)
- [x] `swarmsim/env/swarm_env.py` — custom `SwarmExplorationScenario` (grid exploration, comm, MAPPO global state)
- [x] `swarmsim/env/grid.py` — exploration grid utilities
- [x] `swarmsim/policy/network.py` — `SwarmActor` (movement + message head), `CentralizedCritic`, `ActorCritic` (Stage 1)
- [x] `swarmsim/policy/ppo.py` — custom PPO + `SwarmRolloutBuffer` (~240 lines)
- [x] `swarmsim/policy/train_swarm.py` — multi-agent training (`--comm-mode full|null|none`, `--rollout-steps`, `--num-envs`)
- [x] `swarmsim/policy/eval.py` — rollout eval, time-to-90%-coverage metric
- [x] Short smoke training run → `weights/swarm_policy_full.pt` (~4k steps; undertrained)

### Demo / visualizer
- [x] `swarmsim/server/main.py` — decoupled sim loop + WebSocket at 30 Hz
- [x] `swarmsim/sim/state.py` — JSON state schema (base64 grid)
- [x] `swarmsim/visualizer/` — Three.js `DataTexture` grid, agent cones, comm links, stats panel
- [x] Demo tuning: `demo.sim_hz: 12`, deterministic actions, `max_speed` cap
- [x] **Bug fix:** grid texture Y-flip / axis mapping so explored cells align with agent positions (`scene.js`)

### Analysis
- [x] `swarmsim/analysis/run_ablation.py` — compare no/null/full comm
- [x] `swarmsim/analysis/comm_analysis.py` — message–state correlation heatmap
- [ ] Comm ablation at Bundle F config (null/none baselines) — pending

### Reward shaping + training experiments (2026-07-05/06)
- [x] **Count-based curiosity** (`reward.delta`, `--curiosity`) + **inter-agent repulsion** (`--repulsion`)
- [x] **Frontier reward** (`reward.frontier`, `--frontier`) — bonus for unexplored cells in local 5×5
- [x] **Delayed std anneal** (`--std-anneal`, `--std-anneal-start`, `--std-final`)
- [x] **Local-only actor obs** (`--no-global-map`, 59-dim) for load-bearing comm
- [x] Demo server: checkpoint-aware obs dim + `SWARMSIM_WEIGHTS` / `demo.weights` in config
- [x] Experiment scripts: `run_bundle_a.sh` … `run_bundle_f.sh`

### Best eval result (deterministic, 50 seeds × 20 episodes)
- **Bundle F** (`swarm_policy_full_bundle_f.pt`): **35.9%** mean final coverage
- Prior best: Bundle A 21.9%, Exp 5 GRU 19.5%

---

## Work remaining (priority order)

### 1. Training (most important)
- [ ] **Comm ablation** at Bundle F config (`null` / `none`, same hyperparams as `full`)
- [ ] If deterministic >40% and comm wins: record ablation in README + resume bullets
- [ ] **Bundle F + gentle anneal** (after visual confirms sweeping, not streaking): anneal start ~220k, std-final ~0.8
- [ ] Target: 75%+ deterministic coverage (may need task scaling: smaller grid / more agents)

### 2. Analysis + README
- [ ] Run `comm_analysis.py` on trained `full` weights; add correlation plot to README
- [ ] Ablation results table in README
- [ ] Replace `demorecording.mov` after retraining with better behavior

### 3. Visualizer polish
- [ ] Verify grid/agent alignment after `scene.js` fix (restart demo)
- [ ] Optional: `?speed=` query param for 1x/4x playback
- [ ] Optional: flash on new cell discovery
- [ ] Pass `GRID_SIZE` from server instead of hardcoded 32 in `scene.js`

### 4. Stage 1 (optional, user-driven)
- [ ] MuJoCo waypoint training for learning — see `agents_stage_1.md`
- [ ] Do **not** block Stage 2 on this

### 5. Nice-to-have
- [ ] Colab notebook `notebooks/train_colab.ipynb`
- [ ] `scripts/run_demo.sh` open browser automatically
- [ ] Collision penalty in reward (if agents pile up after training)
- [ ] Git LFS or release artifact for trained `swarm_policy_full.pt`

---

## Key files map

| Path | Role |
|------|------|
| `swarmsim/config.yaml` | All hyperparameters |
| `swarmsim/env/swarm_env.py` | VMAS scenario |
| `swarmsim/policy/train_swarm.py` | Stage 2 training |
| `swarmsim/policy/ppo.py` | PPO implementation |
| `swarmsim/server/main.py` | Live demo server |
| `swarmsim/visualizer/scene.js` | Three.js rendering |
| `weights/swarm_policy_full_bundle_f.pt` | **Best demo weights** (35.9% det. eval) |

---

## Architecture

```
VMAS train (Colab/local) → swarm_policy_{full|null|none}.pt
                              ↓
                    server/main.py (12 Hz sim)
                              ↓ WebSocket 30 Hz
                    visualizer/ (Three.js)
```

Stage 1 MuJoCo is a separate track; weights do not transfer.

---

## Communication ablation

| Mode | Config flag | Meaning |
|------|-------------|---------|
| `none` | `comm.mode: none` | No message head |
| `null` | `comm.mode: null` | Messages emitted but zeroed in obs |
| `full` | `comm.mode: full` | Full emergent comm |

---

## Actor observation

**59 dims (local-only, Bundle A/F default):** `[norm_pos(2), norm_vel(2), local_5x5(25), neighbor_rel(6), incoming_msgs(24)]`

**123 dims (with global map):** above + `global_map(64)` when `env.global_map_downsample: 8`

- Use `--no-global-map` for 59-dim obs (comm becomes load-bearing).
- Demo/eval infer obs layout from checkpoint metadata + weight shapes.

---

## Reward (default `team_new_cells`)

```
reward = alpha * new_cells - gamma * revisit + delta * curiosity + frontier * frontier_frac - repulsion * proximity
```

| Key | Default | Bundle F |
|-----|---------|----------|
| `alpha` | 1.0 | team new cells per step |
| `gamma` | 0.3 | revisit penalty |
| `delta` (`--curiosity`) | 0 | **0.3** count-based exploration bonus |
| `frontier` (`--frontier`) | 0 | **0.2** unexplored fraction in 5×5 window |
| `repulsion` (`--repulsion`) | 0 | **0.05** when agents within 3 cells |

Team rewards are averaged across agents in `train_swarm.py` for PPO.

---

## Experiment results summary (deterministic eval)

| Run | Coverage | Notes |
|-----|----------|-------|
| γ=0.3 baseline | 14.9% | |
| Exp 3 global map | 12.8% | observability alone insufficient |
| Exp 4 std anneal → 0.20 | 13.2% | |
| **Exp 5 GRU @ 300k** | **19.5%** | memory helps |
| GRU @ 500k (no reward change) | 13.0% | longer ≠ better |
| Overnight Exp 6+7 @ 1M | 13.5% / 14.7% | anneal + spread hurt |
| **Bundle A** (curiosity + repulsion, GRU 500k) | **21.9%** | repulsion fixes corner clustering |
| Bundle D (anneal @ 150k, std 0.7, 250k) | 13.1% det ≈ stoch | gap closed, bad mean committed |
| **Bundle F** (frontier 0.2, 250k, no anneal) | **35.9%** | **current best** |

**Train/eval gap diagnosis:** stochastic train coverage → ~100% while deterministic eval ~22% (Bundle A) because mean policy draws thin diagonal streaks; noise accidentally fills gaps. Frontier reward addresses sweeping behavior.

---

## Reward / policy tuning history (see STORYLINE.md for narrative)

- `reward.gamma` (revisit penalty): **0.3** (was 0.01).
- **Exp 5 GRU:** `--use-gru` → **19.5%**.
- **Bundle A:** `--no-global-map --curiosity 0.3 --repulsion 0.05 --use-gru` @ 500k → **21.9%**.
- **Bundle F:** + `--frontier 0.2` @ 250k → **35.9%** (demo default).
- **Std anneal CLI:** `--std-anneal --std-anneal-start N --std-final 0.7` (delayed; don't start during breakthrough ~110k–200k).
- PPO NaN guards in `ppo.py` / `train_swarm.py` for long runs.

---

## Common commands

```bash
source .venv/bin/activate

# Train (Bundle F — current best config)
python -m swarmsim.policy.train_swarm --comm-mode full --timesteps 250000 \
  --num-envs 4 --rollout-steps 256 --gamma 0.3 --use-gru --no-global-map \
  --curiosity 0.3 --frontier 0.2 --repulsion 0.05 --save-name swarm_policy_full_bundle_f

# Or use script
bash scripts/run_bundle_f.sh

# Demo (default: bundle_f weights in config.yaml)
bash scripts/run_demo.sh
SWARMSIM_WEIGHTS=swarm_policy_full_bundle_a.pt bash scripts/run_demo.sh

# Eval
python -m swarmsim.policy.eval --weights weights/swarm_policy_full_bundle_f.pt

# Ablation
python -m swarmsim.analysis.run_ablation --weights-dir weights
```

---

## Gotchas

1. **`.gitignore`:** `ENV/` matches `swarmsim/env/` on macOS — exceptions added as `!swarmsim/env/**`
2. **Demo weights:** `demo.weights` in config or `SWARMSIM_WEIGHTS` env; server matches checkpoint obs dim
3. **Grid bytes:** NumPy `explored[cx, cy]` C-order; frontend must flip Y to match `normToWorld()`
4. **Demo speed:** `demo.sim_hz` in config (default 12)
5. **Stage 1:** do not modify MuJoCo files when working on Stage 2 unless fixing bugs

---

## Cursor session prompt

> We are building SwarmSim — multi-agent VMAS swarm with custom PPO and learned communication, live Three.js demo. Read `agents.md` and `SwarmSim_Altered_Design_Doc.md`. Stage 1 MuJoCo is sidelined (`agents_stage_1.md`). Active focus: [describe task]. Do not edit the plan file. Use `.venv`.
