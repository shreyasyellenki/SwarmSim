"""FastAPI WebSocket server for live swarm visualization."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from swarmsim.env.swarm_env import load_config, make_swarm_env
from swarmsim.policy.eval import load_policy
from swarmsim.sim.state import build_sim_state

latest_state: dict = {"step": 0, "coverage_pct": 0.0, "grid": "", "agents": [], "comm_links": []}
sim_running = True


class SwarmSimulator:
    def __init__(self, cfg: dict, weights_path: Path):
        self.cfg = cfg
        self.demo_cfg = cfg.get("demo", {"sim_hz": 12, "deterministic": True})
        self.device = torch.device("cpu")
        self.env, _ = make_swarm_env(cfg, num_envs=1, device="cpu")
        self.scenario = self.env.scenario
        self.actor, _ = load_policy(weights_path, cfg, self.device)
        self.actor.eval()
        self.obs = self.env.reset()
        self.step_count = 0
        self.num_agents = cfg["env"]["num_agents"]
        self.hidden_states = [self.actor.initial_hidden(1, self.device) for _ in range(self.num_agents)]

    def step_once(self) -> dict:
        num_agents = self.num_agents
        actions = []
        deterministic = self.demo_cfg.get("deterministic", True)
        with torch.no_grad():
            for agent_idx in range(num_agents):
                agent_obs = self.obs[agent_idx].to(self.device)
                h_in = self.hidden_states[agent_idx]
                if deterministic:
                    move, message, h_out = self.actor.act_deterministic(agent_obs, h_in)
                else:
                    move, message, _, _, h_out = self.actor.act(agent_obs, h_in)
                self.hidden_states[agent_idx] = h_out
                if message is None:
                    actions.append(move)
                else:
                    actions.append(torch.cat([move, message], dim=-1))

        self.obs, _, dones, _ = self.env.step(actions)
        self.step_count += 1

        if isinstance(dones, list):
            done = bool(dones[0].any().item())
        else:
            done = bool(dones.any().item())
        if done or self.step_count >= self.cfg["env"]["episode_horizon"]:
            self.obs = self.env.reset()
            self.step_count = 0
            self.hidden_states = [
                self.actor.initial_hidden(1, self.device) for _ in range(num_agents)
            ]

        return self._build_state()

    def _build_state(self) -> dict:
        env_cfg = self.cfg["env"]
        half = env_cfg["world_size"] / 2.0
        agents = []
        for i, agent in enumerate(self.scenario.world.agents):
            pos = agent.state.pos[0].cpu().numpy()
            vel = agent.state.vel[0].cpu().numpy()
            heading = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-4 else 0.0
            msg_mag = float(self.scenario.outgoing_messages[0, i].norm().item())
            agents.append(
                {
                    "id": i,
                    "x": float((pos[0] + half) / env_cfg["world_size"]),
                    "y": float((pos[1] + half) / env_cfg["world_size"]),
                    "heading": heading,
                    "msg_magnitude": msg_mag,
                }
            )

        return build_sim_state(
            step=self.step_count,
            coverage_pct=float(self.scenario.coverage[0].item()),
            grid=self.scenario.get_grid_numpy(0),
            agents=agents,
            comm_links=self.scenario.get_comm_links(0),
        )


def _resolve_weights(cfg: dict) -> Path:
    weights_dir = Path(__file__).resolve().parents[2] / cfg["training"]["weights_dir"]
    for name in ("swarm_policy_full.pt", "swarm_policy.pt"):
        path = weights_dir / name
        if path.exists():
            return path
    return weights_dir / cfg["training"]["swarm_policy"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    weights = _resolve_weights(cfg)
    simulator = SwarmSimulator(cfg, weights)
    sim_hz = float(cfg.get("demo", {}).get("sim_hz", 12))
    sim_interval = 1.0 / sim_hz

    async def run_simulation():
        global latest_state, sim_running
        while sim_running:
            latest_state = simulator.step_once()
            await asyncio.sleep(sim_interval)

    task = asyncio.create_task(run_simulation())
    yield
    sim_running = False
    task.cancel()


app = FastAPI(title="SwarmSim", lifespan=lifespan)

visualizer_dir = Path(__file__).resolve().parents[1] / "visualizer"
app.mount("/static", StaticFiles(directory=str(visualizer_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(visualizer_dir / "index.html")


@app.websocket("/ws")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest_state)
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        pass
