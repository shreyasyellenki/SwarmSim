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

### Analysis (scripts exist, full runs pending)
- [x] `swarmsim/analysis/run_ablation.py` — compare no/null/full comm
- [x] `swarmsim/analysis/comm_analysis.py` — message–state correlation heatmap

### Known eval result (smoke weights)
- Mean final coverage ~19%, never hits 90% in 500 steps — expected for ~4k-step training

---

## Work remaining (priority order)

### 1. Training (most important)
- [ ] **Long `full` training run** on Colab or local GPU:
  ```bash
  python -m swarmsim.policy.train_swarm --comm-mode full --timesteps 500000 --num-envs 8 --rollout-steps 2048
  ```
- [ ] **Ablation baselines:** `null` and `none` with same timesteps
- [ ] **Eval all three** and record results:
  ```bash
  python -m swarmsim.analysis.run_ablation --weights-dir weights
  ```
- [ ] Target: full-comm beats null-comm beats no-comm on `time_to_90pct_coverage` (50 seeds × 20 eps)

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
| `weights/swarm_policy_full.pt` | Demo weights (gitignored) |

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

## Actor observation (as of Exp 3)

Each agent's observation = `[norm_pos(2), norm_vel(2), local_5x5(25), neighbor_rel(6), incoming_msgs(24), global_map(64)]` = **123 dims** with the global map on.

- **`env.global_map_downsample: 8`** — actor sees an 8×8 coarse "fraction explored" map (shared belief state). Set to `0` to disable (reverts to 59-dim local-only obs).
- Added in Experiment 3 (Path A) to fix corner-clustering: with only a 5×5 local window, a deterministic policy can't perceive where unexplored space is, so it collapses to a constant heading. The critic already used this downsampled map; now the actor does too (shared helper `_downsampled_explored`).
- **Tradeoff:** less "purely decentralized" — frame as a shared belief map. On an open grid this also makes comm largely redundant (agents see the same holes); comm becomes load-bearing once observability is tightened (obstacles / private maps).

---

## Reward / policy tuning history (see STORYLINE.md for full log)

- `reward.gamma` (revisit penalty): **0.3** (was 0.01). Sweep 0.01→0.3→0.5 gave 13%→14.9%→13% deterministic coverage.
- `policy.init_log_std`: **0.0** default; sweep to -0.7/-1.6 did not help (~13%).
- **Learned log_std without anneal:** policies end at **~+2.0 (std≈7.4)** — noise grows; see Exp 4.
- **Exp 4 std anneal:** `--std-anneal --entropy-anneal` → final std≈0.20, deterministic coverage **13.2%** (vs 14.9% no anneal) — diagnostic confirmed, eval gain did not materialize at 300k.
- **Exp 5 GRU actor:** `--use-gru` → deterministic coverage **19.5%** (best so far). Weights: `swarm_policy_full_gru.pt`.
- **Overnight Exp 6+7:** `bash scripts/run_overnight_exp67.sh` — GRU+anneal @ 1M (Exp 6), then GRU+anneal+spread reward @ 1M (Exp 7). Results: `weights/overnight_exp67_results.json`.
- Ablation (none/null/full) at 300k, γ=0.01: 18.3% / 13.1% / 13.0% — comm did **not** help on open grid.
- **Exp 3 global map in actor obs** (300k, γ=0.3): **12.8%** deterministic coverage vs **14.9%** without map — did not fix corner-clustering hypothesis at this budget.
- Runs use **300k steps** (1M crashes ~470k on reward saturation → advantage collapse → NaN; guards added in ppo.py/train_swarm.py).

---

## Common commands

```bash
source .venv/bin/activate

# Train
python -m swarmsim.policy.train_swarm --comm-mode full --timesteps 200000 --num-envs 4 --rollout-steps 256

# Demo
bash scripts/run_demo.sh
# → http://localhost:8000

# Eval
python -m swarmsim.policy.eval --weights weights/swarm_policy_full.pt --episodes 5

# Ablation
python -m swarmsim.analysis.run_ablation --weights-dir weights

# Comm analysis
python -m swarmsim.analysis.comm_analysis --weights weights/swarm_policy_full.pt
```

---

## Gotchas

1. **`.gitignore`:** `ENV/` matches `swarmsim/env/` on macOS — exceptions added as `!swarmsim/env/**`
2. **Undertrained demo:** corner-clustering is policy quality, not just viz bugs
3. **Grid bytes:** NumPy `explored[cx, cy]` C-order; frontend must flip Y to match `normToWorld()`
4. **Demo speed:** `demo.sim_hz` in config (default 12)
5. **Stage 1:** do not modify MuJoCo files when working on Stage 2 unless fixing bugs

---

## Cursor session prompt

> We are building SwarmSim — multi-agent VMAS swarm with custom PPO and learned communication, live Three.js demo. Read `agents.md` and `SwarmSim_Altered_Design_Doc.md`. Stage 1 MuJoCo is sidelined (`agents_stage_1.md`). Active focus: [describe task]. Do not edit the plan file. Use `.venv`.
