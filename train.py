"""
Training Pipeline for Basic SAC-PSO on Multi-AMR Warehouse Scheduling (JSSPLA).
Simplified 6-dimensional State and 3-dimensional Action (w, c1, c2).
Supports 3-Stage Curriculum Training:
- Episodes 1..100:   Stage 1 - Small Scenario (12 - 50 jobs)
- Episodes 101..200: Stage 2 - Medium Scenario (50 - 150 jobs)
- Episodes 201..300: Stage 3 - Large Scenario (150 - 300 jobs)
"""

import os
import sys
import time
import random
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from warehouse_env import WarehouseInstance
from sac_agent import SACAgent, SACReplayBuffer
from pso_env import BasicWarehousePSORLEnv


def generate_scenario_suites() -> Tuple[list, list, list]:
    """
    Generates 3 dedicated suites corresponding to 3 distinct scenario scales:
    1. Small Scenario Suite (12 - 40 jobs): Baseline & quick flow balancing.
    2. Medium Scenario Suite (50 - 120 jobs): Bottleneck & inbound surge flows.
    3. Large Scenario Suite (150 - 250 jobs): High congestion & large stress testing.
    """
    # 1. Small Scenario Suite
    small_suite = [
        WarehouseInstance.create_scenario(1, seed=101),
        WarehouseInstance.generate(num_inbound=25, num_outbound=25, seed=101, name="Small_W1"),
        WarehouseInstance.generate(num_inbound=15, num_outbound=35, seed=102, name="Small_W2"),
        WarehouseInstance.generate(num_inbound=35, num_outbound=15, seed=103, name="Small_W3"),
    ]
    for i in range(15):
        n = random.randint(8, 16)
        n_in, n_out = n, n + random.randint(-3, 3)
        small_suite.append(WarehouseInstance.generate(
            num_inbound=n_in, num_outbound=n_out, seed=200 + i,
            name=f"Small_Rand_{i+1}_{n_in}In_{n_out}Out"
        ))

    # 2. Medium Scenario Suite
    med_suite = [
        WarehouseInstance.create_scenario(2, seed=102),
        WarehouseInstance.create_scenario(3, seed=103),
        WarehouseInstance.generate(num_inbound=75, num_outbound=75, seed=201, name="Med_W4"),
        WarehouseInstance.generate(num_inbound=45, num_outbound=105, seed=202, name="Med_W5"),
        WarehouseInstance.generate(num_inbound=105, num_outbound=45, seed=203, name="Med_W6"),
    ]
    for i in range(15):
        n_in = random.randint(25, 60)
        n_out = random.randint(25, 60)
        med_suite.append(WarehouseInstance.generate(
            num_inbound=n_in, num_outbound=n_out, seed=300 + i,
            name=f"Med_Rand_{i+1}_{n_in}In_{n_out}Out"
        ))

    # 3. Large Scenario Suite
    large_suite = [
        WarehouseInstance.create_scenario(4, seed=104),
        WarehouseInstance.generate(num_inbound=150, num_outbound=150, seed=301, name="Large_W7"),
        WarehouseInstance.generate(num_inbound=90, num_outbound=210, seed=302, name="Large_W8"),
        WarehouseInstance.generate(num_inbound=210, num_outbound=90, seed=303, name="Large_W9"),
    ]
    for i in range(15):
        n_in = random.randint(65, 120)
        n_out = random.randint(65, 120)
        large_suite.append(WarehouseInstance.generate(
            num_inbound=n_in, num_outbound=n_out, seed=400 + i,
            name=f"Large_Rand_{i+1}_{n_in}In_{n_out}Out"
        ))

    return small_suite, med_suite, large_suite


def train_basic_sac(
    num_episodes: int = 300,
    pop_size: int = 50,
    max_iter: int = 50,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_model_path: str = os.path.join(os.path.dirname(__file__), "models", "sac_basic_pso.pt"),
    plot_path: str = os.path.join(os.path.dirname(__file__), "results", "sac_basic_training.png"),
):
    print("=" * 85)
    print(f"TRAINING BASIC SAC-PSO (3-STAGE SCENARIO CURRICULUM: {num_episodes} EPISODES)")
    print(f"Device: {device.upper()} | Swarm: {pop_size} particles, {max_iter} iters/ep")
    print("Stage 1 (Ep 1-100):   Small Scenarios")
    print("Stage 2 (Ep 101-200): Medium Scenarios")
    print("Stage 3 (Ep 201-300): Large Scenarios")
    print("=" * 85)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    small_suite, med_suite, large_suite = generate_scenario_suites()
    print(f"Scenario Suites initialized: Small={len(small_suite)}, Medium={len(med_suite)}, Large={len(large_suite)}")

    agent = SACAgent(state_dim=6, action_dim=3, lr=3e-4, gamma=0.95, device=device)
    replay_buffer = SACReplayBuffer(state_dim=6, action_dim=3, capacity=50000)

    # Replay buffer warm-up
    print("\nWarmup: Collecting initial transitions...")
    for _ in range(6):
        inst = random.choice(small_suite)
        env = BasicWarehousePSORLEnv(inst, num_particles=pop_size, max_iter=max_iter, seed=random.randint(1, 10000))
        s = env.reset()
        done = False
        while not done:
            a = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
            s_next, r, done, _ = env.step(a)
            replay_buffer.add(s, a, r, s_next, done)
            s = s_next

    print(f"Warmup complete. Replay buffer size: {replay_buffer.size}\n")

    ep_rewards = []
    actor_losses = []
    critic_losses = []
    alphas = []
    makespan_reductions = []

    t_start = time.time()
    for ep in range(1, num_episodes + 1):
        # 3 Stages: 100 episodes per scenario scale
        if ep <= 100:
            stage_str = "S1-Small"
            inst = random.choice(small_suite)
        elif ep <= 200:
            stage_str = "S2-Med"
            inst = random.choice(med_suite)
        else:
            stage_str = "S3-Large"
            inst = random.choice(large_suite)

        if ep == 101:
            print(f"\n>>> [TRANSITION] Ep 101: Switched to Stage 2 - Medium Scenario (50-150 Jobs)...")
        elif ep == 201:
            print(f"\n>>> [TRANSITION] Ep 201: Switched to Stage 3 - Large Scenario (150-300 Jobs)...")

        env = BasicWarehousePSORLEnv(inst, num_particles=pop_size, max_iter=max_iter, seed=ep * 17)
        state = env.reset()
        ep_reward = 0.0
        done = False

        c_losses = []
        a_losses = []

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # 2 update steps per env step
            for _ in range(2):
                loss_info = agent.update(replay_buffer, batch_size=batch_size)
                if loss_info:
                    c_losses.append(loss_info["critic_loss"])
                    a_losses.append(loss_info["actor_loss"])

        reduction_pct = (env.initial_gbest - env.gbest_val) / max(1e-5, env.initial_gbest) * 100.0

        ep_rewards.append(ep_reward)
        makespan_reductions.append(reduction_pct)
        actor_losses.append(np.mean(a_losses) if a_losses else 0.0)
        critic_losses.append(np.mean(c_losses) if c_losses else 0.0)
        alphas.append(float(agent.alpha.item()))

        if ep % 10 == 0 or ep == num_episodes:
            elapsed = time.time() - t_start
            avg_r = np.mean(ep_rewards[-10:])
            avg_red = np.mean(makespan_reductions[-10:])
            print(f"[{stage_str}] Ep [{ep:3d}/{num_episodes}] | Avg R: {avg_r:7.2f} | Cmax Imp: {avg_red:5.1f}% | Alpha: {agent.alpha.item():.4f} | Time: {elapsed:5.1f}s")

    # Save model weights
    agent.save(save_model_path)
    print(f"\nModel successfully saved to: {save_model_path}")

    # Save training history JSON
    history_data = {
        "ep_rewards": [float(r) for r in ep_rewards],
        "makespan_reductions": [float(m) for m in makespan_reductions],
        "critic_losses": [float(c) for c in critic_losses],
        "alphas": [float(a) for a in alphas],
    }
    hist_path = os.path.join(os.path.dirname(plot_path), "sac_basic_training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    # Plot training metrics with Stage dividers (Times New Roman, clean titles)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.patch.set_facecolor("#ffffff")

    def _draw_stage_dividers(ax):
        ax.axvline(100, color="#7f8c8d", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.axvline(200, color="#7f8c8d", linestyle=":", linewidth=1.5, alpha=0.8)

    # 1. Episode Reward
    axes[0, 0].plot(ep_rewards, color="#1f77b4", linewidth=1.2, alpha=0.85)
    _draw_stage_dividers(axes[0, 0])
    axes[0, 0].set_title("Episode Reward", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Episode", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Makespan Reduction
    axes[0, 1].plot(makespan_reductions, color="#2ca02c", linewidth=1.2, alpha=0.85)
    _draw_stage_dividers(axes[0, 1])
    axes[0, 1].set_title("Makespan Reduction %", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Episode", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Critic Loss
    axes[1, 0].plot(critic_losses, color="#d62728", linewidth=1.2, alpha=0.85)
    _draw_stage_dividers(axes[1, 0])
    axes[1, 0].set_title("Critic Loss", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Episode", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Alpha
    axes[1, 1].plot(alphas, color="#9467bd", linewidth=1.5)
    _draw_stage_dividers(axes[1, 1])
    axes[1, 1].set_title("Entropy Temperature Alpha", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Episode", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {plot_path}")

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--pop", type=int, default=50, help="Swarm population size")
    parser.add_argument("--iter", type=int, default=50, help="PSO iterations per episode")
    args = parser.parse_args()

    train_basic_sac(num_episodes=args.episodes, pop_size=args.pop, max_iter=args.iter)
