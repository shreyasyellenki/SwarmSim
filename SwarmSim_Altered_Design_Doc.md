# SwarmSim — Design Document
### RL-Trained Drone Swarm with Emergent Communication and Real-Time Visualization

---

## 1. What This Project Is

A multi-agent drone swarm simulation where agents are trained with reinforcement learning to explore an unknown 2D grid environment. Agents communicate with neighbors via small learned message vectors — nothing is hardcoded about what they say. The output is a collaboratively built map that grows in real-time, visualized in a web interface.

The project has two distinct stages: a single-agent navigation policy trained in MuJoCo (physics-grounded, industry-standard), then scaled to multi-agent swarm coordination with emergent communication in VMAS (built for fast multi-agent training).

**In one sentence for your resume:** *Single-agent navigation policy trained in MuJoCo; scaled to a multi-agent swarm with emergent inter-agent communication in VMAS, visualized via a FastAPI + Three.js real-time dashboard.*

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│             Stage 1 — Single Agent (MuJoCo)          │
│                                                      │
│   MuJoCo env (MJCF quadrotor) ──► Custom PPO        │
│   - Waypoint navigation task        │                │
│   - Physics-grounded dynamics  Saved weights         │
│   - Runs locally on Mac             │                │
│                                     ▼                │
│                          Validated single-agent      │
│                          policy + MuJoCo familiarity │
└──────────────────────────┬──────────────────────────┘
                           │ concepts carry forward
                           ▼
┌─────────────────────────────────────────────────────┐
│          Stage 2 — Multi-Agent Swarm (VMAS)          │
│                                                      │
│   VMAS swarm env ──► Custom PPO + Comm Head          │
│   - 6 agents, 32×32 grid               │            │
│   - Learned message vectors       Saved weights      │
│   - Trains on Colab Pro                │            │
└──────────────────────────┬─────────────────────────-┘
                           │ export policy.pt
                           ▼
┌─────────────────────────────────────────────────────┐
│              Demo / Inference (Local Mac)            │
│                                                      │
│   VMAS rollout ──► FastAPI WebSocket ──► Three.js   │
│   (run trained policy)   (stream state)  (visualize)│
└─────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack and Why Each Choice

| Component | Choice | Why |
|---|---|---|
| Stage 1 simulation | MuJoCo | Industry standard in robotics research (DeepMind, OpenAI). MJCF model teaches you how physics environments are defined. Free, works natively on M1 Mac. |
| Stage 2 training env | VMAS | Built specifically for multi-agent scenarios. Vectorized — fast iteration on Colab GPU. Has existing navigation/flocking scenarios to reference. SB3 does not support VMAS natively, so custom PPO is required either way. |
| RL algorithm | Custom PPO (~250 lines) | Fewer hyperparameters than QMIX or MADDPG. You write GAE, rollout buffer, and clipping yourself — fully explainable in an interview. Industry-standard policy gradient method. |
| Communication | Learned message head on shared policy network | Agents emit an 8-float vector per timestep. No hardcoding of what it means — the network learns what's useful to say. |
| Critic | Centralized (MAPPO-style) | Critic sees global state during training; actors stay decentralized at inference. Better sample efficiency than per-agent local critics for cooperative tasks. |
| Training hardware | Google Colab Pro | No local GPU needed. Handles VMAS vectorized training in reasonable wall-clock time. |
| Backend | FastAPI + WebSocket (decoupled) | Sim runs in background task at full speed; broadcaster sends latest state at 30 Hz. Dropping frames is fine — frontend always shows current state, never queued old state. |
| Frontend | Three.js with DataTexture | Single plane mesh with a texture updated as a byte array each frame. Avoids 4,096 draw calls from per-cell meshes that would stutter at 30fps. |
| Dependency management | conda (arm64) | Guarantees native M1 environment. Avoids Rosetta Python issues. |

---

## 4. Hyperparameter Config

Lock these before writing any training code. Put them in `config.yaml` and never change two at once.

```yaml
# config.yaml
env:
  num_agents: 6
  grid_size: 32          # start here; upgrade to 64 after baseline works
  local_window_k: 5      # 5×5 cell observation window around each agent
  comm_radius: 8         # cells (~25% of 32×32 diagonal)
  max_neighbors: 3       # fixed obs padding; zero-pad if fewer in range
  episode_horizon: 500   # steps per episode
  coverage_target: 0.85  # episode terminates early if hit

reward:
  alpha: 1.0             # new cells explored
  beta: 0.1              # collision penalty
  gamma: 0.01            # revisit penalty

comm:
  message_dim: 8         # floats per message vector

eval:
  num_seeds: 50
  episodes_per_seed: 20
  metric: time_to_90pct_coverage   # mean ± std across seeds
```

**Why locking matters:** RL training is sensitive to hyperparameters. If you change grid_size and comm_radius at the same time and coverage improves, you don't know which caused it. One change at a time, always.

---

## 5. Environment Design

### 5.1 Stage 1 — MuJoCo Single-Agent Task

A quadrotor defined in MJCF navigates to a sequence of waypoints in 3D space. The task is simple on purpose — the goal is to learn MuJoCo's API and get a policy training, not to build something complex.

What you learn from this stage:
- How MJCF XML defines a physics body (mass, joints, actuators, geometry)
- How `mujoco.MjModel` and `mujoco.MjData` work
- How to write a custom gym-style environment wrapping MuJoCo
- How to run the MuJoCo viewer for visual debugging

### 5.2 Stage 2 — VMAS Multi-Agent Environment

**The world:** 2D grid, 32×32 cells. Each cell is either unexplored (0) or explored (agent_id). Agents start at random positions. Episode ends at 85% coverage or 500 steps.

**Agent observation vector:**
```
[
  own_x, own_y,                              # normalized position (0–1)
  own_vx, own_vy,                            # velocity
  local_grid[5×5 flattened],                 # 25 values: explored/unexplored window
  neighbor_1_rel_x, neighbor_1_rel_y, neighbor_1_msg[8],
  neighbor_2_rel_x, neighbor_2_rel_y, neighbor_2_msg[8],
  neighbor_3_rel_x, neighbor_3_rel_y, neighbor_3_msg[8],
  # zero-padded if fewer than 3 neighbors in comm radius
]
```
Total obs dim: 2 + 2 + 25 + 3×(2+8) = **59 floats**

**Agent action space:**
```
[
  delta_x, delta_y,   # continuous movement (-1 to 1)
  message[8]          # broadcast to neighbors within radius R
]
```

**Reward:**
```python
reward = (
  + 1.0 * new_cells_explored_this_step
  - 0.1 * collision_with_wall_or_agent
  - 0.01 * revisit_penalty
)
```
Start with only the first term. Add penalties only if agents behave degenerately (crashing into each other constantly, refusing to explore).

---

## 6. Policy Network Architecture

Each agent runs the same shared network (parameter sharing — standard in cooperative MARL):

```
Observation vector (59 floats)
          │
          ▼
   Linear(59 → 256) + ReLU
          │
          ▼
   Linear(256 → 128) + ReLU
          │
    ┌─────┴──────────────────┐
    ▼                        ▼
Linear(128 → 2)        Linear(128 → 8)
  (movement)            (message head)
    │                        │
  tanh()                  tanh()        ← bounded -1 to 1
    │                        │
movement action        broadcast to
                         neighbors


CRITIC (centralized — training only, not used at inference):

Global state (all agent positions + full grid summary)
          │
          ▼
   Linear(global_dim → 256) + ReLU
          │
          ▼
   Linear(256 → 128) + ReLU
          │
          ▼
   Linear(128 → 1)           ← scalar value estimate V(s)
```

**Why centralized critic:** during training the critic can see everything — all agent positions, the full grid state. This gives much better value estimates than each agent trying to evaluate the global situation from local observations only. At deployment, the critic is thrown away — only the actor runs. This is called MAPPO (Multi-Agent PPO).

**Why parameter sharing:** all agents are identical, so one shared network trains faster and generalizes better across agent positions.

---

## 7. Communication Mechanism — Explained Simply

1. At each timestep, every agent's network outputs two things: a movement action and an 8-float message vector.
2. The message is broadcast to all agents within radius R (8 cells).
3. On the *next* timestep, each agent's observation includes the messages it received from up to 3 neighbors.
4. During training, when agent B receives agent A's message and uses it to make a better decision, the reward signal flows backward through B's network and then through the message it received — which means it flows back into A's message head. So A's network learns to emit messages that help B.

You never tell the agents what to put in the message. You just give them the mechanism and let training figure out what's useful.

### Communication Ablation (three training runs, one config flag)

To make "communication helps" a defensible claim you need apples-to-apples comparison:

| Condition | Description |
|---|---|
| **No comm** | Message head removed. Obs inputs where messages would be are zero-padded. Same MLP width otherwise. |
| **Null comm** | Message head exists, agents emit messages, but neighbors' observations have message inputs zeroed out. Tests whether the head learned anything vs. just getting lucky. |
| **Full comm** | As designed. Neighbors receive and use messages. |

If full comm beats null comm, agents are learning something useful to say. That's the result. Report mean ± std time-to-90%-coverage across 50 seeds × 20 episodes for all three conditions.

---

## 8. Custom PPO — What You're Writing (~250 lines)

This is not as scary as it sounds. PPO has four components:

**1. Rollout buffer** — collect N steps of (obs, action, reward, done, value, log_prob) across all agents in parallel.

**2. GAE (Generalized Advantage Estimation)** — compute how much better each action was than the baseline value estimate. This is one loop over the buffer.

**3. Policy update** — for K epochs, sample minibatches from the buffer and compute:
```python
ratio = new_log_prob / old_log_prob
clipped = clip(ratio, 1-eps, 1+eps)
policy_loss = -min(ratio * advantage, clipped * advantage).mean()
value_loss = (returns - values).pow(2).mean()
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

**4. Training loop** — reset env, collect rollout, update, repeat.

Cursor can scaffold all four components from the spec above. Your job is to understand what each piece does, not write it from scratch from memory.

---

## 9. Backend — FastAPI WebSocket Server

```python
# Decoupled sim + broadcast pattern
app = FastAPI()
latest_state = {}

async def run_simulation():
    """Runs at full inference speed in background."""
    while True:
        step_simulation()        # advance one timestep
        latest_state.update(get_sim_state())
        await asyncio.sleep(0)   # yield to event loop

@app.on_event("startup")
async def startup():
    asyncio.create_task(run_simulation())

@app.websocket("/ws")
async def stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(latest_state)
        await asyncio.sleep(1/30)   # broadcast at 30 Hz regardless of sim speed
```

Sim state JSON schema:
```json
{
  "agents": [
    {"id": 0, "x": 0.3, "y": 0.5, "heading": 1.2, "msg_magnitude": 0.74},
    ...
  ],
  "grid": "base64-encoded flat byte array",  // 32×32 = 1024 bytes, agent_id per cell
  "coverage_pct": 0.42,
  "step": 140,
  "comm_links": [[0, 2], [1, 3]]             // agent pairs currently in comm radius
}
```

Send the grid as a base64 byte array rather than a nested JSON array — much smaller payload at 30fps.

---

## 10. Frontend — Three.js Visualizer

**Layout:** split panel. Left: top-down grid view. Right: live stats (coverage %, step, agent count, active comm links).

**Grid rendering with DataTexture:**
```javascript
// One texture, one plane mesh — not 4096 individual meshes
const gridData = new Uint8Array(32 * 32 * 4);  // RGBA per cell
const texture = new THREE.DataTexture(gridData, 32, 32, THREE.RGBAFormat);

// On each WebSocket message:
function updateGrid(gridBytes) {
  for (let i = 0; i < gridBytes.length; i++) {
    const agentId = gridBytes[i];
    const color = agentId === 0 ? UNEXPLORED_COLOR : AGENT_COLORS[agentId];
    gridData[i*4] = color.r;
    gridData[i*4+1] = color.g;
    gridData[i*4+2] = color.b;
    gridData[i*4+3] = 255;
  }
  texture.needsUpdate = true;  // Three.js uploads to GPU next frame
}
```

**Agent rendering:** small cone meshes, one per agent, positioned and rotated to heading each frame.

**Comm links:** `THREE.LineSegments` updated each frame with current comm_links pairs. Faint opacity, agent color.

---

## 11. File Structure

```
swarmsim/
├── config.yaml                   # all hyperparameters — single source of truth
├── env/
│   ├── mujoco_nav.py             # Stage 1: single-agent MuJoCo waypoint env
│   ├── mujoco_drone.xml          # MJCF quadrotor model
│   └── swarm_env.py              # Stage 2: VMAS multi-agent scenario
├── policy/
│   ├── network.py                # actor (+ message head) + centralized critic
│   ├── ppo.py                    # rollout buffer, GAE, PPO update (~250 lines)
│   ├── train_mujoco.py           # Stage 1 training loop (local)
│   ├── train_swarm.py            # Stage 2 training loop (Colab)
│   └── eval.py                   # load weights, run rollout, export state
├── server/
│   └── main.py                   # FastAPI + decoupled WebSocket server
├── visualizer/
│   ├── index.html
│   ├── scene.js                  # Three.js setup, DataTexture grid, agent meshes
│   └── socket.js                 # WebSocket client, state updates
├── weights/
│   ├── mujoco_policy.pt          # Stage 1 saved weights
│   └── swarm_policy.pt           # Stage 2 saved weights
├── analysis/
│   └── comm_analysis.py          # correlate message components with agent state
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 12. Milestones (2 Weeks)

### Week 1 — ML Core

| Day | Goal |
|---|---|
| 1 | Setup conda env. Run VMAS `navigation` + `flocking` examples, read their scenario code. Run MuJoCo viewer. Understand MJCF format. |
| 2 | Write `mujoco_nav.py` — single agent, waypoint task. Write `mujoco_drone.xml`. Get MuJoCo env stepping without errors. |
| 3 | Write `network.py` actor + `ppo.py` for single-agent case. Train on MuJoCo env. Watch agent learn to reach waypoints. |
| 4 | Write `swarm_env.py` — multi-agent VMAS scenario, no communication yet. Establish baseline coverage metric (no-comm condition). |
| 5 | Add message head and centralized critic to `network.py`. Wire communication into obs space. Retrain (null-comm condition). |
| 6 | Full communication training run on Colab. Compare all three ablation conditions. |
| 7 | Run `comm_analysis.py` — correlate message dims with agent state. Write up findings for README. This is your key analytical result. |

### Week 2 — Systems + Visualizer

| Day | Goal |
|---|---|
| 8 | Write `server/main.py` — load swarm weights, run VMAS rollout, stream state over WebSocket. Test with a plain HTML page that logs JSON. |
| 9 | Build `scene.js` — Three.js setup, DataTexture grid, verify grid updates correctly from WebSocket. |
| 10 | Add agent meshes and comm link lines to visualizer. Stats panel. |
| 11 | Connect everything end-to-end. One command starts server, opens browser, shows live swarm. |
| 12 | Polish: Dockerfile, clean README with demo GIF, ablation results table. |
| 13–14 | Buffer. Stretch: add `?speed=` query param to visualizer for 1x/4x playback. |

---

## 13. What You Can Say in an Interview

**"Why two stages — MuJoCo then VMAS?"**
MuJoCo is the industry standard for physics-grounded robotics simulation — DeepMind and OpenAI both use it. I wanted hands-on experience defining environments in MJCF and training against realistic dynamics. But MuJoCo isn't designed for fast multi-agent training at scale — VMAS is vectorized specifically for that. So the right tool is different at each stage.

**"How does the communication work?"**
Each agent's policy network has a separate output head that emits an 8-dimensional message vector every timestep. Neighbors within a fixed radius receive that vector as part of their observation. Because the reward signal backpropagates through what an agent did with the messages it received, the sending agent's message head learns to emit information that helps neighbors make better decisions. Nothing is hardcoded about what to say.

**"Why custom PPO over a library?"**
VMAS is built on TorchRL's batched tensor API — SB3 doesn't support it natively. I had to write the training loop anyway, which turned out to be valuable: I can explain GAE, the clipping objective, and why the centralized critic improves sample efficiency in cooperative settings. About 250 lines total.

**"Why a centralized critic?"**
In cooperative MARL, each agent's individual actions only partially explain the team's reward. A centralized critic that sees the global state during training gives much better value estimates, which makes the policy gradient more accurate. At deployment the critic is thrown away — only the actor runs, using only local observations.

**"What's the key result?"**
I ran three ablations: no communication, null messages (head exists but ignored), and full communication. Full comm achieves 90% coverage X% faster than no-comm, and beats null-messages by Y% — which means agents learned something useful to say, not just that a bigger model helps.

---

## 14. Dependencies

```
# requirements.txt
torch>=2.0
vmas
mujoco
fastapi
uvicorn[standard]
websockets
numpy
pyyaml
```

Frontend: Three.js via CDN. No npm build step needed.

---

## 15. Cursor Agent Instructions

Start every Cursor session with:

> "We are building SwarmSim — a two-stage project. Stage 1: single-agent MuJoCo waypoint navigation. Stage 2: multi-agent VMAS swarm with custom PPO and learned communication. Stack: MuJoCo, VMAS, custom PyTorch PPO, FastAPI WebSocket, Three.js. All hyperparameters live in config.yaml. Current task: [specific file]. Do not add complexity beyond the design doc."

One file per session. Build order matters — do not skip ahead:
1. `mujoco_drone.xml` + `mujoco_nav.py`
2. `ppo.py` (single-agent first)
3. `network.py`
4. `swarm_env.py`
5. `train_swarm.py`
6. `server/main.py`
7. `visualizer/`
