"""
Main Entry Point: Multi-AMR Smart Warehouse Scheduling Optimization (JSSPLA).
Based on: "Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems
and Validation via Gazebo Simulation" (Supervisor: Dr. Truong Ngoc Cuong, Student: Tran Viet Trung An)

Compares:
- Heuristics: FIFO, SPT, LPT
- Conventional Metaheuristics: Standard GA, Standard PSO
- Proposed CDPSO (Cooperative Discrete PSO from Paper)
- Proposed SAC-PSO (Deep RL Meta-Optimization for Standard PSO - Ours)
Across Scenarios 1 (24J), 2 (28J), 3 (28J), and 4 (350J - Large Instance).
"""

import os
import sys
from benchmark_experiment import run_full_benchmarks


def main():
    print("\n" + "=" * 95)
    print("   JOB SHOP SCHEDULING WITH LIMITED AMRS (JSSPLA) IN MULTI-AMR WAREHOUSE")
    print("   DEEP RL (SAC) DYNAMIC CONTROL vs CDPSO, PSO, GA, FIFO, SPT, LPT")
    print("=" * 95)

    model_path = os.path.join("models", "sac_warehouse_pso.pt")
    if not os.path.exists(model_path):
        print("[INFO] Model not found. Training SAC-PSO on multi-instance suite...")
        from train_sac_pso import train_sac_pso
        train_sac_pso(num_episodes=50, pop_size=30, max_iter=25, save_model_path=model_path)

    # Run full benchmarks matching paper Table 1 and Figures 3-6
    run_full_benchmarks(
        num_runs=5,
        pop_size=30,
        max_iter=25,
        model_path=model_path,
    )

    print("\n[SUCCESS] All benchmark experiments completed successfully.")
    print("Check the results folder for generated figures and summary tables:")
    print("  - Results Table : results/table1_comparison.md")
    print("  - Boxplots (Fig 3-6): results/boxplots_comparison.png")
    print("  - Convergence Curve : results/scenario4_convergence.png")
    print("  - Training Curve    : results/sac_pso_training.png")


if __name__ == "__main__":
    main()