# SwarmSim — Stage 1 Agent Context (MuJoCo)

> **Status: sidelined.** Stage 1 is optional learning track. Active development focuses on Stage 2 (VMAS swarm + visualizer). Do not modify Stage 1 files when working on Stage 2 unless fixing a bug.

Use this file at the start of any Cursor session focused on MuJoCo / Stage 1.

---

## What Stage 1 is

Single-agent quadrotor waypoint navigation in **MuJoCo**. Goal is hands-on experience with MJCF, `MjModel`/`MjData`, and custom PPO — **not** to improve the swarm demo.

**Weights do not transfer to Stage 2.** `mujoco_policy.pt` is unused by the live demo.

---

## Files (do not overwrite when doing Stage 2 work)

| File | Purpose |
|------|---------|
| [`swarmsim/env/mujoco_drone.xml`](swarmsim/env/mujoco_drone.xml) | MJCF quadrotor: free joint, 4 motor sites, floor plane |
| [`swarmsim/env/mujoco_nav.py`](swarmsim/env/mujoco_nav.py) | Gymnasium env: random spawn → random waypoint |
| [`swarmsim/policy/train_mujoco.py`](swarmsim/policy/train_mujoco.py) | Stage 1 PPO training entry point |
| [`swarmsim/policy/ppo.py`](swarmsim/policy/ppo.py) | Shared PPO (also used by Stage 2) |
| [`swarmsim/policy/network.py`](swarmsim/policy/network.py) | `ActorCritic` class for single-agent |
| [`swarmsim/config.yaml`](swarmsim/config.yaml) | `stage1_mujoco:` section |

Output weights: `weights/mujoco_policy.pt` (gitignored)

---

## Environment spec

**Observation (9 floats):**
```
[rel_waypoint_x, rel_waypoint_y, rel_waypoint_z,
 velocity_x, velocity_y, velocity_z,
 height, distance_to_waypoint, yaw_rate]
```

**Action (4 floats):** motor thrust commands in `[-1, 1]`, mapped to hover ± 50% per motor.

**Reward:**
```python
reward = -distance_scale * dist + alive_bonus
       + success_bonus   # if within success_radius
       - crash_penalty   # if z < 0.05
```

**Episode:** ends on reach waypoint, crash, or 500 steps.

**Config keys** (`config.yaml` → `stage1_mujoco:`):
- `success_radius: 0.3`
- `waypoint_range` / `spawn_range` — see config file
- PPO hyperparams in `ppo:` section (shared with Stage 2)

---

## Setup (venv)

```bash
cd /path/to/SwarmSim
source .venv/bin/activate
pip install -r requirements.txt   # if not already installed
```

Set Cursor Python interpreter to `.venv/bin/python`.

---

## Commands

**Smoke test (env steps without training):**
```bash
python -c "
from swarmsim.env.mujoco_nav import MuJoCoNavEnv
env = MuJoCoNavEnv()
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step(env.action_space.sample())
print('OK', obs.shape, 'reward=', round(r, 3))
env.close()
"
```

**Short training run:**
```bash
python -m swarmsim.policy.train_mujoco --timesteps 50000
```

**Full training (default 500k steps from config):**
```bash
python -m swarmsim.policy.train_mujoco
```

**TensorBoard:**
```bash
tensorboard --logdir runs/mujoco
```

**MuJoCo viewer (visual debug):**
```python
from swarmsim.env.mujoco_nav import MuJoCoNavEnv
env = MuJoCoNavEnv(render_mode="human")
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()
env.close()
```

---

## MJCF model notes (`mujoco_drone.xml`)

- Body: `drone` with `freejoint` (6 DOF pose + 4 motor sites)
- Actuators: 4 `motor` elements on FL/FR/RL/RR sites, thrust along local Z
- `hover_thrust` computed in env as `|gravity| * mass / 4` using `model.body_mass[body_id]`
- Red `waypoint_marker` site position updated each reset

---

## Known gotchas

1. **Mass access:** use `model.body_mass[body_id]`, not `body.mass` (array in MuJoCo 3.x).
2. **Gitignore:** `swarmsim/env/` was excluded by `ENV/` rule on macOS — fixed with `!swarmsim/env/**` in `.gitignore`.
3. **Do not confuse** `swarmsim/env/` (Python package) with `.venv/` (virtual environment).

---

## Relationship to Stage 2

| Carries forward | Does not carry forward |
|-----------------|------------------------|
| `ppo.py` implementation | `mujoco_policy.pt` weights |
| PPO / GAE concepts | MuJoCo physics in demo |
| Interview story for MuJoCo | Obs/action spaces |

Stage 2 uses **VMAS** (`swarmsim/env/swarm_env.py`), demo streams VMAS rollout via FastAPI → Three.js.

---

## Interview talking points

- **Why MuJoCo?** Industry-standard physics sim (DeepMind, OpenAI). MJCF teaches how robotics envs define bodies, actuators, contacts.
- **Why sideline it?** MuJoCo isn't built for fast multi-agent training; VMAS is. Right tool per stage.
- **What did you learn?** MJCF structure, wrapping MuJoCo in Gymnasium, training with custom PPO on realistic dynamics.

---

## Cursor session prompt (copy-paste)

> We are working on SwarmSim **Stage 1 only** (MuJoCo waypoint navigation). Files: `mujoco_drone.xml`, `mujoco_nav.py`, `train_mujoco.py`. Do not modify Stage 2 / VMAS / server / visualizer. See `agents_stage_1.md` for context. Current task: [describe task].
