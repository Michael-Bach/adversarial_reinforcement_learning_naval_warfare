#!/usr/bin/env python3
"""
train_cmano.py — Training entry point using CMANO as the simulation backend.

Usage
-----
  python scripts/train_cmano.py --config configs/cmano_cat_and_mouse.yaml
  python scripts/train_cmano.py --config configs/cmano_cat_and_mouse.yaml --wandb

Before running, ensure:
  1. CMO is running with cmano_bridge.lua loaded and CMATNOBridgeInit() called.
  2. The bridge_dir in the YAML is accessible to both this process and CMO.
     (For cross-machine: mount the Windows share before running this script.)

The training loop is identical to train.py; only the environment differs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from naval_rl.agents.td3 import TD3Agent
from naval_rl.envs.cmano_env import CMATNOEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], use_wandb: bool = False) -> None:
    run_name = cfg.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir  = Path(cfg.get("output_dir", "outputs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.get("wandb_project", "naval-adversarial-rl"),
            name    = run_name,
            config  = cfg,
        )

    print(f"Connecting to CMANO bridge at: {cfg.get('bridge_dir', '/tmp/cmano_bridge')}")
    print("Waiting for CMANO to signal ready (ensure CMATNOBridgeInit() has run)...")

    env = CMATNOEnv(
        cfg        = cfg,
        bridge_dir = cfg.get("bridge_dir", "/tmp/cmano_bridge"),
        ipc_timeout = cfg.get("ipc_timeout", 120.0),
    )

    print("CMANO bridge ready.")

    state_dim = env.observation_space.shape[0]
    n_alice   = env.n_alice
    n_bob     = env.n_bob
    act_dim_A = 4 * n_alice
    act_dim_B = 4 * n_bob
    agent_cfg = cfg["agent"]
    device    = cfg.get("device", "cpu")

    alice = TD3Agent(
        state_dim  = state_dim,
        action_dim = act_dim_A,
        noise_cfg  = agent_cfg["noise_alice"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = device,
    )
    bob = TD3Agent(
        state_dim  = state_dim,
        action_dim = act_dim_B,
        noise_cfg  = agent_cfg["noise_bob"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = device,
    )

    train_cfg     = cfg["training"]
    num_episodes  = train_cfg["num_episodes"]
    warmup_steps  = train_cfg["warmup_steps"]
    batch_size    = train_cfg["batch_size"]
    save_every    = train_cfg.get("save_every", 500)
    rollout_every = train_cfg.get("rollout_every", 200)

    global_step     = 0
    all_trajectories = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        alice.noise.reset()
        bob.noise.reset()

        ep_reward = np.zeros(2)
        done      = False

        while not done:
            a_alice = alice.select_action(obs)
            a_bob   = bob.select_action(obs)
            action  = np.concatenate([a_alice, a_bob])

            next_obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rare = info["rare_event"]
            alice.store(obs, a_alice, rewards[0], next_obs, done, rare=rare)
            bob.store(  obs, a_bob,   rewards[1], next_obs, done, rare=rare)

            if global_step >= warmup_steps:
                alice.train(batch_size)
                bob.train(batch_size)

            ep_reward  += rewards
            obs         = next_obs
            global_step += 1

        log = {
            "episode":           ep,
            "reward_alice":      float(ep_reward[0]),
            "reward_bob":        float(ep_reward[1]),
            "alice/actor_loss":  alice.actor_loss,
            "alice/critic_loss": alice.critic_loss,
            "alice/mean_q":      alice.mean_q,
            "alice/noise_rms":   alice.noise_rms,
            "bob/actor_loss":    bob.actor_loss,
            "bob/critic_loss":   bob.critic_loss,
            "bob/mean_q":        bob.mean_q,
            "bob/noise_rms":     bob.noise_rms,
        }

        if ep % 50 == 0:
            print(
                f"Ep {ep:5d} | "
                f"Alice: {ep_reward[0]:8.2f}  Bob: {ep_reward[1]:8.2f} | "
                f"Step: {global_step:7d}"
            )

        if use_wandb:
            import wandb
            wandb.log(log, step=global_step)

        # Greedy rollout (deterministic eval)
        if ep % rollout_every == 0 and ep > 0:
            traj_obs, _ = env.reset()
            traj_done   = False
            traj_A, traj_B = [], []
            while not traj_done:
                # Record positions from last observed state
                a_raw = [env.alice_ids, env.bob_ids]
                traj_A.append(traj_obs[:n_alice * 5].reshape(n_alice, 5)[:, :2].copy())
                traj_B.append(traj_obs[n_alice * 5:].reshape(n_bob,   5)[:, :2].copy())
                a_A = alice.select_action_deterministic(traj_obs)
                a_B = bob.select_action_deterministic(traj_obs)
                traj_obs, _, t1, t2, _ = env.step(np.concatenate([a_A, a_B]))
                traj_done = t1 or t2
            all_trajectories.append({
                "episode": ep,
                "alice":   np.array(traj_A, dtype=np.float32),
                "bob":     np.array(traj_B, dtype=np.float32),
            })

        if ep % save_every == 0 and ep > 0:
            alice.save(str(out_dir / f"alice_ep{ep}.pt"))
            bob.save(  str(out_dir / f"bob_ep{ep}.pt"))

    alice.save(str(out_dir / "alice_final.pt"))
    bob.save(  str(out_dir / "bob_final.pt"))

    if all_trajectories:
        traj_path = out_dir / "trajectories.npy"
        np.save(traj_path, np.array(all_trajectories, dtype=object), allow_pickle=True)
        print(f"Saved {len(all_trajectories)} trajectory snapshots → {traj_path}")

    env.close()

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"Training complete. Outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adversarial naval RL agents via CMANO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, use_wandb=args.wandb)
