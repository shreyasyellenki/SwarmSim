# SwarmSim

Multi-agent RL drone swarm with emergent communication and real-time visualization.

**Stage 1:** Single-agent waypoint navigation in MuJoCo (MJCF quadrotor + custom PPO)  
**Stage 2:** Multi-agent grid exploration in VMAS with learned 8-dim message vectors  
**Demo:** VMAS rollout streamed via FastAPI WebSocket to a Three.js dashboard

Design doc: [`SwarmSim_Altered_Design_Doc.md`](SwarmSim_Altered_Design_Doc.md)  
Stage 1 (optional, sidelined): [`agents_stage_1.md`](agents_stage_1.md)

## Setup

```bash
cd SwarmSim
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the venv is activated (`source .venv/bin/activate`).

## Stage 1 — MuJoCo training

```bash
python -m swarmsim.policy.train_mujoco --timesteps 50000
```

## Stage 2 — VMAS swarm training

```bash
# Full communication (short smoke test)
python -m swarmsim.policy.train_swarm --comm-mode full --timesteps 10000 --num-envs 2 --rollout-steps 128

# Full training run
python -m swarmsim.policy.train_swarm --comm-mode full

# Ablation baselines
python -m swarmsim.policy.train_swarm --comm-mode null --timesteps 100000
python -m swarmsim.policy.train_swarm --comm-mode none --timesteps 100000
```

## Evaluation and ablation

```bash
python -m swarmsim.policy.eval --weights weights/swarm_policy_full.pt
python -m swarmsim.analysis.run_ablation --weights-dir weights
python -m swarmsim.analysis.comm_analysis --weights weights/swarm_policy_full.pt
```

## Live demo

```bash
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh
# Open http://localhost:8000
```

## Project structure

```
swarmsim/
├── config.yaml           # All hyperparameters
├── env/                  # MuJoCo + VMAS environments
├── policy/               # PPO, networks, training scripts
├── server/               # FastAPI WebSocket server
├── visualizer/           # Three.js frontend
├── analysis/             # Comm analysis + ablation runner
└── sim/                  # Shared state schema
```

## Key hyperparameters (locked in config.yaml)

| Parameter | Value |
|-----------|-------|
| Agents | 6 |
| Grid | 32×32 |
| Comm radius | 8 cells |
| Message dim | 8 |
| Eval metric | time to 90% coverage (50 seeds × 20 episodes) |

## Communication ablation

| Condition | Description |
|-----------|-------------|
| `none` | No message head; zero-padded obs |
| `null` | Messages emitted but ignored by neighbors |
| `full` | Full emergent communication |
