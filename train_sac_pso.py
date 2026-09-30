"""
Training Pipeline for SAC-PSO on Multi-AMR Warehouse Scheduling (JSSPLA).

Trains the SAC Agent across a diverse suite of 30+ warehouse instances
ranging from small (12 jobs) to large stress-test instances (350 jobs).
Saves the optimized model weights to models/sac_warehouse_pso.pt
and training curves to results/sac_pso_training.png.
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from warehouse_env import WarehouseInstance
from sac_pso import SACAgent, SACReplayBuffer, WarehousePSORLEnv


def generate_training_suite(num_instances: int = 30) -> list:
    """
    Generates a diverse suite of warehouse instances across different scales
    and inbound/outbound ratios.
    """
    suite = []
    # 1. Official paper benchmark scenarios
    for sc_id in [1, 2, 3, 4]:
        suite.append(WarehouseInstance.create_scenario(sc_id, seed=100 + sc_id))

    # 2. Small scale instances (12 - 25 jobs)
    for i in range(8):
        n_in = random.randint(4, 10)
        n_out = random.randint(8, 15)
        suite.append(WarehouseInstance.generate(
            num_inbound=n_in,
            num_outbound=n_out,
            seed=200 + i,
            name=f"Train_Small_{i+1}_{n_in}In_{n_out}Out"
        ))

    # 3. Medium scale instances (30 - 80 jobs)
    for i in range(10):
        n_in = random.randint(10, 30)
        n_out = random.randint(20, 50)
        suite.append(WarehouseInstance.generate(
            num_inbound=n_in,
            num_outbound=n_out,
            seed=300 + i,
            name=f"Train_Med_{i+1}_{n_in}In_{n_out}Out"
        ))

    # 4. Large scale instances (100 - 350 jobs)
    for i in range(8):
        n_in = random.randint(50, 150)
        n_out = random.randint(60, 200)
        suite.append(WarehouseInstance.generate(
            num_inbound=n_in,
            num_outbound=n_out,
            seed=400 + i,
            name=f"Train_Large_{i+1}_{n_in}In_{n_out}Out"
        ))

    return suite


def train_sac_pso(
    num_episodes: int = 80,
    pop_size: int = 25,
    max_iter: int = 20,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_model_path: str = "models/sac_warehouse_pso.pt",
    plot_path: str = "results/sac_pso_training.png",
):
    print("=" * 80)
    print(f"TRAINING SAC-PSO ON MULTI-AMR WAREHOUSE INSTANCES (DEVICE: {device.upper()})")
    print("=" * 80)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    suite = generate_training_suite(30)
    print(f"Generated {len(suite)} diverse warehouse instances for RL training.")

    agent = SACAgent(state_dim=14, action_dim=6, lr=3e-4, gamma=0.95, device=device)
    replay_buffer = SACReplayBuffer(state_dim=14, action_dim=6, capacity=40000)

    # Pre-populate replay buffer with random exploration steps
    print("\nWarmup: Collecting initial transitions into replay buffer...")
    for _ in range(5):
        inst = random.choice(suite[:12])  # pick small/med for fast warmup
        env = WarehousePSORLEnv(inst, num_particles=pop_size, max_iter=max_iter, seed=random.randint(1, 10000))
        s = env.reset()
        done = False
        while not done:
            a = np.random.uniform(-1.0, 1.0, size=6).astype(np.float32)
            s_next, r, done, _ = env.step(a)
            replay_buffer.add(s, a, r, s_next, done)
            s = s_next

    print(f"Warmup complete. Replay buffer size: {replay_buffer.size}")

    episode_rewards = []
    critic_losses = []
    actor_losses = []
    alphas = []
    gbest_reductions = []

    best_sc4_eval = float("inf")
    sc4_eval_inst = WarehouseInstance.create_scenario(4, seed=42)

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        # Sample an instance: mix of small, med, large
        inst = random.choice(suite)
        env = WarehousePSORLEnv(
            instance=inst,
            num_particles=pop_size,
            max_iter=max_iter,
            seed=random.randint(1, 50000),
        )
        state = env.reset()
        ep_reward = 0.0
        c_loss_list = []
        a_loss_list = []

        done = False
        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Training updates
            metrics = agent.update(replay_buffer, batch_size=batch_size)
            if metrics:
                c_loss_list.append(metrics["critic_loss"])
                a_loss_list.append(metrics["actor_loss"])

        init_cmax = env.initial_gbest
        final_cmax = env.gbest_val
        reduction = (init_cmax - final_cmax) / max(1e-5, init_cmax) * 100.0

        episode_rewards.append(ep_reward)
        gbest_reductions.append(reduction)
        c_loss = float(np.mean(c_loss_list)) if c_loss_list else 0.0
        a_loss = float(np.mean(a_loss_list)) if a_loss_list else 0.0
        critic_losses.append(c_loss)
        actor_losses.append(a_loss)
        alphas.append(agent.alpha.item())

        if ep % 5 == 0 or ep == 1:
            print(
                f"Ep [{ep:3d}/{num_episodes:3d}] | "
                f"Instance: {inst.name:<24} | "
                f"Reward: {ep_reward:6.2f} | "
                f"Imp: {reduction:5.2f}% | "
                f"C_max: {final_cmax:8.2f}s | "
                f"Alpha: {agent.alpha.item():.4f}"
            )

        # Evaluation on Scenario 4 every 15 episodes
        if ep % 15 == 0 or ep == num_episodes:
            eval_env = WarehousePSORLEnv(sc4_eval_inst, num_particles=pop_size, max_iter=max_iter, seed=42)
            s_eval = eval_env.reset()
            d_eval = False
            while not d_eval:
                a_eval = agent.select_action(s_eval, evaluate=True)
                s_eval, _, d_eval, _ = eval_env.step(a_eval)
            print(f"  >>> [Eval Scen 4 @ Ep {ep}] C_max = {eval_env.gbest_val:.2f}s")
            if eval_env.gbest_val < best_sc4_eval:
                best_sc4_eval = eval_env.gbest_val
                agent.save(save_model_path)
                print(f"  >>> Saved NEW BEST model to: {save_model_path}")

    # Always ensure model is saved
    agent.save(save_model_path)
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s. Model saved to: {save_model_path}")

    # Plot training curves
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.suptitle("SAC-PSO Training Progression on Warehouse JSSPLA Instances", fontsize=15, fontweight="bold")

    # 1. Episode Returns
    axs[0, 0].plot(episode_rewards, color="#1f77b4", alpha=0.4, label="Raw Return")
    # Smooth return
    if len(episode_rewards) >= 5:
        smoothed = np.convolve(episode_rewards, np.ones(5)/5, mode="valid")
        axs[0, 0].plot(range(4, len(episode_rewards)), smoothed, color="#1f77b4", linewidth=2.5, label="5-Ep MA")
    axs[0, 0].set_title("Episode RL Return", fontweight="bold")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. Makespan Improvement %
    axs[0, 1].plot(gbest_reductions, color="#2ca02c", alpha=0.4, label="Makespan Imp %")
    if len(gbest_reductions) >= 5:
        smoothed_imp = np.convolve(gbest_reductions, np.ones(5)/5, mode="valid")
        axs[0, 1].plot(range(4, len(gbest_reductions)), smoothed_imp, color="#2ca02c", linewidth=2.5, label="5-Ep MA")
    axs[0, 1].set_title("Makespan Reduction (%) per PSO Run", fontweight="bold")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Reduction %")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. Critic & Actor Loss
    axs[1, 0].plot(critic_losses, color="#d62728", label="Critic Loss")
    axs[1, 0].set_title("Critic Loss", fontweight="bold")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Entropy Alpha Temperature
    axs[1, 1].plot(alphas, color="#9467bd", label="Alpha Temperature")
    axs[1, 1].set_title("Entropy Temperature Alpha", fontweight="bold")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Alpha")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training plot to: {plot_path}")


if __name__ == "__main__":
    train_sac_pso(num_episodes=80, pop_size=30, max_iter=25)
