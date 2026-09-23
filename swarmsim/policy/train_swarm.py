"""Stage 2: train multi-agent VMAS swarm with custom PPO and communication."""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from vmas.simulator.environment.environment import Environment as VmasEnvironment

from swarmsim.env.swarm_env import SwarmExplorationScenario, load_config, make_swarm_env
from swarmsim.policy.network import CentralizedCritic, SwarmActor, swarm_global_dim, swarm_obs_dim
from swarmsim.policy.ppo import PPOConfig, SwarmPPOTrainer, SwarmRolloutBuffer, _params_finite
from swarmsim.seeding import set_global_seeds


def set_comm_mode(cfg: dict, mode: str) -> dict:
    updated = copy.deepcopy(cfg)
    updated["comm"]["mode"] = mode
    return updated


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return start + (end - start) * progress


def delayed_linear_schedule(
    start: float, end: float, global_step: int, start_step: int, total_timesteps: int
) -> float:
    if global_step <= start_step:
        return start
    span = max(total_timesteps - start_step, 1)
    progress = (global_step - start_step) / span
    return linear_schedule(start, end, progress)


def apply_training_schedules(
    actor: SwarmActor,
    trainer: SwarmPPOTrainer,
    cfg: dict,
    global_step: int,
    total_timesteps: int,
    scenario: SwarmExplorationScenario | None = None,
) -> dict[str, float]:
    """Update log_std and entropy_coef from config schedules. Returns logged values."""
    policy_cfg = cfg.get("policy", {})
    progress = global_step / max(total_timesteps, 1)
    logged: dict[str, float] = {}

    std_sched = policy_cfg.get("std_schedule", {})
    if std_sched.get("enabled"):
        start_step = int(std_sched.get("start_step", 0))
        log_std = delayed_linear_schedule(
            float(std_sched.get("start_log_std", 0.0)),
            float(std_sched.get("end_log_std", -1.6)),
            global_step,
            start_step,
            total_timesteps,
        )
        actor.log_std.data.fill_(log_std)

    # Logged unconditionally: with the schedule off this is the value the
    # policy learned, which is the interesting case and was never recorded.
    logged["log_std"] = float(actor.log_std.mean().item())
    logged["action_std"] = float(torch.exp(actor.log_std).mean().item())

    ent_sched = policy_cfg.get("entropy_schedule", {})
    if ent_sched.get("enabled"):
        start_step = int(ent_sched.get("start_step", 0))
        entropy_coef = delayed_linear_schedule(
            float(ent_sched.get("start", 0.01)),
            float(ent_sched.get("end", 0.001)),
            global_step,
            start_step,
            total_timesteps,
        )
        trainer.cfg.entropy_coef = entropy_coef
        logged["entropy_coef"] = entropy_coef

    ppo_cfg = cfg.get("ppo", {})
    lr_sched = ppo_cfg.get("lr_schedule", {})
    if lr_sched.get("enabled"):
        start_step = int(lr_sched.get("start_step", 0))
        lr = delayed_linear_schedule(
            float(lr_sched.get("start", ppo_cfg.get("learning_rate", 3e-4))),
            float(lr_sched.get("end", 3e-5)),
            global_step,
            start_step,
            total_timesteps,
        )
        trainer.set_learning_rate(lr)
        logged["learning_rate"] = lr

    reward_cfg = cfg.get("reward", {})
    gamma_sched = reward_cfg.get("gamma_schedule", {})
    if gamma_sched.get("enabled"):
        start_step = int(gamma_sched.get("start_step", 0))
        revisit_gamma = delayed_linear_schedule(
            float(gamma_sched.get("start", 0.01)),
            float(gamma_sched.get("end", 0.5)),
            global_step,
            start_step,
            total_timesteps,
        )
        reward_cfg["gamma"] = revisit_gamma
        if scenario is not None:
            scenario.reward_cfg["gamma"] = revisit_gamma
        logged["revisit_gamma"] = revisit_gamma

    return logged


@contextlib.contextmanager
def _preserved_vmas_rng():
    """Run a block without consuming the environment RNG stream.

    VMAS keeps one process-wide state list shared by every Environment and
    swaps it in around each env method, so stepping a scratch eval env would
    otherwise shift the training env's resets.
    """
    state = VmasEnvironment.vmas_random_state
    saved = (state[0].clone(), copy.deepcopy(state[1]), copy.deepcopy(state[2]))
    try:
        yield
    finally:
        state[0], state[1], state[2] = saved


def evaluate_during_training(
    eval_env, actor, cfg: dict, device: torch.device, episodes: int, seed: int
) -> float:
    """Mean deterministic coverage over a few fixed-seed episodes.

    Training coverage is measured under action noise, so on its own it cannot
    show a train/eval gap.
    """
    num_agents = cfg["env"]["num_agents"]
    max_steps = cfg["env"]["episode_horizon"]
    scenario = eval_env.scenario
    was_training = actor.training
    actor.eval()
    coverages = []
    with _preserved_vmas_rng(), torch.no_grad():
        for ep in range(episodes):
            obs = eval_env.reset(seed=seed + ep)
            hidden = [actor.initial_hidden(1, device) for _ in range(num_agents)]
            for _ in range(max_steps):
                actions = []
                for i in range(num_agents):
                    move, message, h = actor.act_deterministic(obs[i].to(device), hidden[i])
                    hidden[i] = h
                    actions.append(move if message is None else torch.cat([move, message], dim=-1))
                obs, _, dones, _ = eval_env.step(actions)
                if bool(dones.any().item()):
                    break
            coverages.append(float(scenario.coverage[0].item()))
    if was_training:
        actor.train()
    return float(np.mean(coverages))


def train(
    comm_mode: str = "full",
    total_timesteps: int | None = None,
    num_envs: int = 8,
    rollout_steps: int | None = None,
    revisit_gamma: float | None = None,
    save_name: str | None = None,
    init_log_std: float | None = None,
    std_anneal: bool = False,
    std_anneal_start: int | None = None,
    std_final: float | None = None,
    entropy_anneal: bool = False,
    lr_anneal: bool = False,
    lr_final: float | None = None,
    gamma_anneal: bool = False,
    gamma_start: float | None = None,
    gamma_end: float | None = None,
    use_gru: bool | None = None,
    reward_mode: str | None = None,
    global_map_downsample: int | None = None,
    curiosity: float | None = None,
    frontier: float | None = None,
    repulsion: float | None = None,
    repulsion_radius: int | None = None,
    diversity: float | None = None,
    message_heading_aux: float | None = None,
    episode_horizon: int | None = None,
    local_window_k: int | None = None,
    obstacle_mode: str | None = None,
    seed: int | None = None,
) -> Path:
    cfg = set_comm_mode(load_config(), comm_mode)
    if revisit_gamma is not None:
        cfg["reward"]["gamma"] = revisit_gamma
    if reward_mode is not None:
        cfg["reward"]["mode"] = reward_mode
    if curiosity is not None:
        cfg["reward"]["delta"] = curiosity
    if frontier is not None:
        cfg["reward"]["frontier"] = frontier
    if repulsion is not None:
        cfg["reward"]["repulsion"] = repulsion
    if repulsion_radius is not None:
        cfg["reward"]["repulsion_radius"] = repulsion_radius
    if diversity is not None:
        cfg["reward"]["diversity"] = diversity
    if message_heading_aux is not None:
        cfg["reward"]["message_heading_aux"] = message_heading_aux
    if episode_horizon is not None:
        cfg["env"]["episode_horizon"] = episode_horizon
    if local_window_k is not None:
        cfg["env"]["local_window_k"] = local_window_k
    if obstacle_mode is not None:
        cfg["env"]["obstacle_mode"] = obstacle_mode
    if global_map_downsample is not None:
        cfg["env"]["global_map_downsample"] = global_map_downsample
    if init_log_std is not None:
        cfg.setdefault("policy", {})["init_log_std"] = init_log_std
    policy_cfg = cfg.setdefault("policy", {})
    if use_gru is not None:
        policy_cfg["use_gru"] = use_gru
    if std_anneal:
        std_sched = policy_cfg.setdefault("std_schedule", {})
        std_sched["enabled"] = True
        if std_anneal_start is not None:
            std_sched["start_step"] = std_anneal_start
        if std_final is not None:
            std_sched["end_log_std"] = math.log(std_final)
    if entropy_anneal:
        policy_cfg.setdefault("entropy_schedule", {})["enabled"] = True
    if lr_anneal:
        ppo_cfg = cfg.setdefault("ppo", {})
        lr_sched = ppo_cfg.setdefault("lr_schedule", {})
        lr_sched["enabled"] = True
        lr_sched.setdefault("start", ppo_cfg.get("learning_rate", 3e-4))
        if lr_final is not None:
            lr_sched["end"] = lr_final
    if gamma_anneal:
        reward_cfg = cfg.setdefault("reward", {})
        gamma_sched = reward_cfg.setdefault("gamma_schedule", {})
        gamma_sched["enabled"] = True
        gamma_sched.setdefault("start", 0.01)
        gamma_sched.setdefault("end", 0.5)
        if gamma_start is not None:
            gamma_sched["start"] = gamma_start
        if gamma_end is not None:
            gamma_sched["end"] = gamma_end
        reward_cfg["gamma"] = float(gamma_sched["start"])
    # Seed before any module is constructed so weight init is reproducible too.
    if seed is not None:
        set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_cfg = PPOConfig.from_config(cfg)
    if rollout_steps is not None:
        ppo_cfg.rollout_steps = rollout_steps
    train_cfg = cfg["training"]
    env_cfg = cfg["env"]
    comm_cfg = cfg["comm"]

    if total_timesteps is None:
        total_timesteps = train_cfg["swarm_total_timesteps"]

    env, action_dim = make_swarm_env(cfg, num_envs=num_envs, device=str(device), seed=seed)
    scenario = env.scenario

    eval_interval = int(train_cfg.get("eval_interval", 0) or 0)
    eval_episodes = int(train_cfg.get("eval_episodes", 5))
    eval_env = None
    if eval_interval > 0:
        eval_env, _ = make_swarm_env(cfg, num_envs=1, device=str(device), seed=seed)
    num_agents = env_cfg["num_agents"]
    global_map_cells = (env_cfg.get("global_map_downsample", 0) or 0) ** 2
    include_obstacles = env_cfg.get("obstacle_mode", "none") != "none"
    obs_dim = swarm_obs_dim(
        env_cfg["local_window_k"],
        env_cfg["max_neighbors"],
        comm_cfg["message_dim"],
        global_map_cells,
        include_obstacles=include_obstacles,
    )
    global_dim = swarm_global_dim(
        num_agents, cfg["critic"]["grid_downsample"], include_obstacles=include_obstacles
    )

    actor_log_std = cfg.get("policy", {}).get("init_log_std", 0.0)
    use_gru_flag = bool(policy_cfg.get("use_gru", False))
    gru_hidden = int(policy_cfg.get("gru_hidden", 128))
    actor = SwarmActor(
        obs_dim,
        comm_cfg["message_dim"],
        comm_mode=comm_cfg["mode"],
        init_log_std=actor_log_std,
        use_gru=use_gru_flag,
        gru_hidden=gru_hidden,
    ).to(device)
    critic = CentralizedCritic(global_dim).to(device)
    std_schedule_on = policy_cfg.get("std_schedule", {}).get("enabled", False)
    aux_coef = float(cfg["reward"].get("message_heading_aux", 0.0))
    trainer = SwarmPPOTrainer(
        actor,
        critic,
        ppo_cfg,
        device,
        train_log_std=not std_schedule_on,
        message_heading_aux_coef=aux_coef if comm_cfg["mode"] == "full" else 0.0,
    )

    steps_per_rollout = ppo_cfg.rollout_steps
    hidden_dim = actor.gru_hidden
    buffer = SwarmRolloutBuffer(
        rollout_steps=steps_per_rollout,
        num_agents=num_agents,
        num_envs=num_envs,
        obs_dim=obs_dim,
        global_dim=global_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=hidden_dim,
    )

    weights_dir = Path(__file__).resolve().parents[2] / train_cfg["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_stem = save_name or f"swarm_policy_{comm_mode}"
    save_path = weights_dir / f"{weight_stem}.pt"

    run_name = weight_stem.removeprefix("swarm_policy_")
    log_dir = Path(__file__).resolve().parents[2] / train_cfg["tensorboard_dir"] / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    obs = env.reset()
    global_step = 0
    update_idx = 0
    episode_returns = torch.zeros(num_envs, device=device)
    episode_count = 0
    nan_recoveries = 0
    last_good: dict[str, dict] | None = None

    def build_checkpoint() -> dict:
        ckpt = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "comm_mode": comm_mode,
            "revisit_gamma": cfg["reward"]["gamma"],
            "init_log_std": cfg.get("policy", {}).get("init_log_std", 0.0),
            "use_gru": use_gru_flag,
            "gru_hidden": gru_hidden,
            "reward_mode": cfg["reward"].get("mode", "team_new_cells"),
            "global_map_downsample": env_cfg.get("global_map_downsample", 0) or 0,
            "curiosity_delta": cfg["reward"].get("delta", 0.0),
            "frontier": cfg["reward"].get("frontier", 0.0),
            "repulsion": cfg["reward"].get("repulsion", 0.0),
            "repulsion_radius": cfg["reward"].get("repulsion_radius", 3),
            "diversity": cfg["reward"].get("diversity", 0.0),
            "message_heading_aux": cfg["reward"].get("message_heading_aux", 0.0),
            "episode_horizon": env_cfg.get("episode_horizon", 500),
            "local_window_k": env_cfg.get("local_window_k", 5),
            "obstacle_mode": env_cfg.get("obstacle_mode", "none"),
            "seed": seed,
            "std_anneal_start": policy_cfg.get("std_schedule", {}).get("start_step", 0),
            "std_final": float(
                math.exp(policy_cfg.get("std_schedule", {}).get("end_log_std", math.log(0.7)))
            ),
        }
        return ckpt

    hidden_states = [actor.initial_hidden(num_envs, device) for _ in range(num_agents)]

    while global_step < total_timesteps:
        schedule_vals = apply_training_schedules(
            actor, trainer, cfg, global_step, total_timesteps, scenario=scenario
        )
        while not buffer.full() and global_step < total_timesteps:
            global_state = scenario.build_global_state()
            with torch.no_grad():
                value = critic(global_state)

            actions_to_env = []
            step_obs, step_actions, step_log_probs, step_hidden = [], [], [], []

            for agent_idx in range(num_agents):
                agent_obs = torch.nan_to_num(obs[agent_idx].to(device), nan=0.0, posinf=1.0, neginf=-1.0)
                h_in = hidden_states[agent_idx]
                with torch.no_grad():
                    move, message, log_prob, _, h_out = actor.act(agent_obs, h_in)
                hidden_states[agent_idx] = h_out

                if comm_cfg["mode"] == "none" or message is None:
                    full_action = move
                else:
                    full_action = torch.cat([move, message], dim=-1)

                actions_to_env.append(full_action)
                step_obs.append(agent_obs)
                step_actions.append(full_action)
                step_log_probs.append(log_prob)
                if h_in is not None:
                    step_hidden.append(h_in)

            next_obs, rews, _, _ = env.step(actions_to_env)

            # VMAS collapses terminated and truncated into one flag. Split them
            # so hitting the horizon still bootstraps instead of being treated
            # as a real terminal state.
            terminated = scenario.done().clone().float()
            truncated = (env.steps >= env.max_steps).float()
            episode_end = (terminated + truncated).clamp(max=1.0).bool()

            # Value of the state that actually followed this step, captured
            # before any reset wipes it.
            with torch.no_grad():
                next_value = critic(scenario.build_global_state())

            team_reward = sum(rews) / float(num_agents)
            agent_vel = torch.stack([a.state.vel[:, :2] for a in scenario.world.agents])

            buffer.add_step(
                obs=torch.stack(step_obs),
                global_state=global_state.unsqueeze(0).expand(num_agents, -1, -1),
                actions=torch.stack(step_actions),
                velocities=agent_vel,
                rewards=team_reward.unsqueeze(0).expand(num_agents, -1),
                values=value.unsqueeze(0).expand(num_agents, -1),
                next_values=next_value.unsqueeze(0).expand(num_agents, -1),
                terminated=terminated.unsqueeze(0).expand(num_agents, -1),
                truncated=truncated.unsqueeze(0).expand(num_agents, -1),
                log_probs=torch.stack(step_log_probs),
                hidden_in=torch.stack(step_hidden) if step_hidden else None,
            )

            global_step += num_envs
            episode_returns += team_reward

            if episode_end.any():
                finished = int(episode_end.sum().item())
                writer.add_scalar(
                    "train/episode_return", episode_returns[episode_end].mean().item(), episode_count
                )
                writer.add_scalar(
                    "train/coverage", scenario.coverage[episode_end].mean().item(), episode_count
                )
                writer.add_scalar(
                    "train/message_l2",
                    scenario.outgoing_messages.norm(dim=-1).mean().item(),
                    episode_count,
                )
                writer.add_scalar(
                    "train/exploration_bonus",
                    scenario._exploration_bonus.mean().item(),
                    episode_count,
                )
                writer.add_scalar(
                    "train/frontier_bonus",
                    scenario._frontier_bonus.mean().item(),
                    episode_count,
                )
                writer.add_scalar(
                    "train/repulsion_penalty",
                    scenario._repulsion_penalty.mean().item(),
                    episode_count,
                )
                writer.add_scalar(
                    "train/diversity_penalty",
                    scenario._diversity_penalty.mean().item(),
                    episode_count,
                )
                episode_returns[episode_end] = 0.0
                episode_count += finished

                # VMAS never auto-resets. Without this the run stays inside a
                # single permanently-done episode after the first horizon.
                for env_i in torch.nonzero(episode_end).flatten().tolist():
                    next_obs = env.reset_at(env_i)
                if actor.use_gru:
                    for agent_idx in range(num_agents):
                        hidden_states[agent_idx][episode_end] = 0.0

            obs = next_obs

        if buffer.ptr == 0:
            break

        if not _params_finite(actor, critic):
            if last_good is not None:
                actor.load_state_dict(last_good["actor"])
                critic.load_state_dict(last_good["critic"])
            else:
                raise RuntimeError("Policy parameters became non-finite and no checkpoint is available.")

        buffer.compute_gae(ppo_cfg.gamma, ppo_cfg.gae_lambda)

        pre_update = {
            "actor": {k: v.clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in critic.state_dict().items()},
        }
        try:
            metrics = trainer.update(buffer)
        except (ValueError, RuntimeError) as exc:
            nan_recoveries += 1
            restore = last_good or pre_update
            actor.load_state_dict(restore["actor"])
            critic.load_state_dict(restore["critic"])
            writer.add_scalar("train/nan_recoveries", nan_recoveries, update_idx)
            print(f"Warning: PPO update failed ({exc}); restored last checkpoint.")
            buffer.reset()
            continue
        if not _params_finite(actor, critic):
            nan_recoveries += 1
            restore = last_good or pre_update
            actor.load_state_dict(restore["actor"])
            critic.load_state_dict(restore["critic"])
            writer.add_scalar("train/nan_recoveries", nan_recoveries, update_idx)
            buffer.reset()
            continue

        last_good = {
            "actor": {k: v.clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in critic.state_dict().items()},
        }
        buffer.reset()
        update_idx += 1

        if update_idx % train_cfg["log_interval"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, update_idx)
            if revisit_gamma is not None:
                writer.add_scalar("train/revisit_gamma", revisit_gamma, update_idx)
            for k, v in schedule_vals.items():
                writer.add_scalar(f"train/{k}", v, update_idx)

        if eval_env is not None and update_idx % eval_interval == 0:
            det_coverage = evaluate_during_training(
                eval_env, actor, cfg, device, eval_episodes, seed=990001
            )
            writer.add_scalar("eval/coverage_deterministic", det_coverage, update_idx)
            writer.add_scalar("eval/global_step", global_step, update_idx)

        if update_idx % train_cfg["save_interval"] == 0:
            torch.save(build_checkpoint(), save_path)

    checkpoint = build_checkpoint()

    apply_training_schedules(actor, trainer, cfg, total_timesteps, total_timesteps, scenario=scenario)
    checkpoint = build_checkpoint()

    if not _params_finite(actor, critic) and last_good is not None:
        actor.load_state_dict(last_good["actor"])
        critic.load_state_dict(last_good["critic"])

    torch.save(checkpoint, save_path)
    writer.close()
    print(f"Swarm training complete ({comm_mode}, revisit_gamma={cfg['reward']['gamma']}). Weights saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm-mode", choices=["full", "null", "none"], default="full")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Revisit penalty (reward.gamma)")
    parser.add_argument("--save-name", type=str, default=None, help="Weight filename stem, e.g. swarm_policy_full_gamma03")
    parser.add_argument("--init-log-std", type=float, default=None, help="Initial log std of movement Gaussian")
    parser.add_argument("--std-anneal", action="store_true", help="Anneal log_std per policy.std_schedule")
    parser.add_argument(
        "--std-anneal-start",
        type=int,
        default=None,
        help="Global step to begin std anneal (earlier steps keep start_log_std)",
    )
    parser.add_argument(
        "--std-final",
        type=float,
        default=None,
        help="Target action std at end of anneal (e.g. 0.7); overrides end_log_std",
    )
    parser.add_argument("--entropy-anneal", action="store_true", help="Linearly decay entropy_coef per policy.entropy_schedule")
    parser.add_argument(
        "--lr-anneal",
        action="store_true",
        help="Linearly decay Adam learning rate per ppo.lr_schedule",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Target learning rate at end of lr anneal (e.g. 3e-5; 0 decays to zero)",
    )
    parser.add_argument(
        "--gamma-anneal",
        action="store_true",
        help="Linearly increase revisit penalty (reward.gamma) over training",
    )
    parser.add_argument(
        "--gamma-start",
        type=float,
        default=None,
        help="Initial revisit penalty when --gamma-anneal (default 0.01)",
    )
    parser.add_argument(
        "--gamma-end",
        type=float,
        default=None,
        help="Final revisit penalty when --gamma-anneal (default 0.5)",
    )
    parser.add_argument("--use-gru", action="store_true", help="Use a recurrent (GRU) actor with per-episode hidden state")
    parser.add_argument("--reward-mode", choices=["team_new_cells", "spread"], default=None)
    parser.add_argument(
        "--no-global-map",
        action="store_true",
        help="Disable coarse global coverage map in actor obs (59-dim local-only)",
    )
    parser.add_argument(
        "--curiosity",
        type=float,
        default=None,
        help="Count-based exploration bonus weight (reward.delta)",
    )
    parser.add_argument(
        "--frontier",
        type=float,
        default=None,
        help="Frontier bonus weight for unexplored cells in local window (reward.frontier)",
    )
    parser.add_argument(
        "--repulsion",
        type=float,
        default=None,
        help="Inter-agent proximity penalty weight (reward.repulsion)",
    )
    parser.add_argument(
        "--repulsion-radius",
        type=int,
        default=None,
        help="Grid-cell distance threshold for repulsion penalty",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=None,
        help="Heading-alignment penalty vs in-range neighbors (reward.diversity)",
    )
    parser.add_argument(
        "--message-heading-aux",
        type=float,
        default=None,
        help="Aux loss weight: msg[:2] tracks velocity heading (Bundle H)",
    )
    parser.add_argument(
        "--episode-horizon",
        type=int,
        default=None,
        help="Max steps per episode (env.episode_horizon)",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=None,
        help="Local observation window size in grid cells (env.local_window_k)",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=["none", "scattered", "rooms", "maze"],
        default=None,
        help="Obstacle layout mode (env.obstacle_mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed weight init, action sampling and env resets (omit for nondeterministic)",
    )
    args = parser.parse_args()
    train(
        args.comm_mode,
        args.timesteps,
        args.num_envs,
        args.rollout_steps,
        revisit_gamma=args.gamma,
        save_name=args.save_name,
        init_log_std=args.init_log_std,
        std_anneal=args.std_anneal,
        std_anneal_start=args.std_anneal_start,
        std_final=args.std_final,
        entropy_anneal=args.entropy_anneal,
        lr_anneal=args.lr_anneal,
        lr_final=args.lr_final,
        gamma_anneal=args.gamma_anneal,
        gamma_start=args.gamma_start,
        gamma_end=args.gamma_end,
        use_gru=True if args.use_gru else None,
        reward_mode=args.reward_mode,
        global_map_downsample=0 if args.no_global_map else None,
        curiosity=args.curiosity,
        frontier=args.frontier,
        repulsion=args.repulsion,
        repulsion_radius=args.repulsion_radius,
        diversity=args.diversity,
        message_heading_aux=args.message_heading_aux,
        episode_horizon=args.episode_horizon,
        local_window_k=args.local_window,
        obstacle_mode=args.obstacle_mode,
        seed=args.seed,
    )
