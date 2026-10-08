"""
Main Entry Point: Multi-AMR Smart Warehouse Scheduling Optimization (JSSPLA).
Soft Actor-Critic Dynamic Meta-Optimization for Particle Swarm Optimization (SAC-PSO Basic).
State Space: 6 dimensions | Action Space: 3 continuous parameters (w, c1, c2).

Evaluates and compares:
- Priority Dispatching Heuristics: FIFO, SPT, LPT
- Conventional Metaheuristics: Standard GA, Standard PSO
- Proposed SAC-PSO (Deep RL Meta-Optimization for Standard PSO - Ours)
Across 9 Standard Warehouse Instances (W1 to W9: Small, Medium, Large scales).
"""

import os
import sys
import argparse
import time
from warehouse_env import WarehouseInstance
from baselines import solve_fifo, solve_spt, solve_lpt, StandardGA, StandardPSO
from pso_env import BasicSACPSOSolver
from benchmark import run_benchmark, generate_outputs, STANDARD_SCENARIOS
from train import train_basic_sac


def run_quick_demo(model_path: str = os.path.join("models", "sac_basic_pso.pt")):
    """Runs a quick evaluation demo on Scenario W1 (50 jobs: 25 Inbound, 25 Outbound)."""
    print("\n" + "=" * 85)
    print("   QUICK DEMO EVALUATION ON SCENARIO W1 (Small - 50 Jobs / 100 Operations)")
    print("=" * 85)

    sc = STANDARD_SCENARIOS[0]
    inst = WarehouseInstance.generate(
        num_inbound=sc["nin"],
        num_outbound=sc["nout"],
        name=sc["name"],
        seed=sc["seed"],
    )

    print(f"Instance: {sc['code']} ({sc['desc']})")
    print(f"Total Operations: {len(inst.base_sequence)} | Active AMRs: {inst.num_amrs} (4 Clusters)\n")

    # 1. Deterministic Heuristics
    t0 = time.time()
    c_fifo, sched_fifo = solve_fifo(inst)
    t_fifo = time.time() - t0
    print(f"  [1/6] FIFO Heuristic : Makespan = {c_fifo:8.2f}s (CPU Time: {t_fifo:.3f}s)")

    t0 = time.time()
    c_spt, sched_spt = solve_spt(inst)
    t_spt = time.time() - t0
    print(f"  [2/6] SPT Heuristic  : Makespan = {c_spt:8.2f}s (CPU Time: {t_spt:.3f}s)")

    t0 = time.time()
    c_lpt, sched_lpt = solve_lpt(inst)
    t_lpt = time.time() - t0
    print(f"  [3/6] LPT Heuristic  : Makespan = {c_lpt:8.2f}s (CPU Time: {t_lpt:.3f}s)")

    # 2. Metaheuristics (Pop=30, MaxIter=25 for quick demo)
    pop_size = 30
    max_iter = 25

    t0 = time.time()
    ga_solver = StandardGA(pop_size=pop_size, max_gen=max_iter, seed=42)
    c_ga, _, _ = ga_solver.solve(inst)
    t_ga = time.time() - t0
    print(f"  [4/6] Standard GA    : Makespan = {c_ga:8.2f}s (CPU Time: {t_ga:.3f}s)")

    t0 = time.time()
    pso_solver = StandardPSO(num_particles=pop_size, max_iter=max_iter, seed=42)
    c_pso, _, _ = pso_solver.solve(inst)
    t_pso = time.time() - t0
    print(f"  [5/6] Standard PSO   : Makespan = {c_pso:8.2f}s (CPU Time: {t_pso:.3f}s)")

    t0 = time.time()
    sac_solver = BasicSACPSOSolver(
        model_path=model_path,
        num_particles=pop_size,
        max_iter=max_iter,
        seed=42,
        verbose=True,
    )
    c_sac, _, _, actions_hist = sac_solver.solve(inst)
    t_sac = time.time() - t0
    print(f"  [6/6] SAC-PSO (Ours) : Makespan = {c_sac:8.2f}s (CPU Time: {t_sac:.3f}s)")

    imp_fifo = (c_fifo - c_sac) / c_fifo * 100.0
    imp_pso = (c_pso - c_sac) / c_pso * 100.0
    print("\n" + "-" * 85)
    print(f"  >> SAC-PSO Improvement vs FIFO : {imp_fifo:+.2f}%")
    print(f"  >> SAC-PSO Improvement vs PSO  : {imp_pso:+.2f}%")
    print("-" * 85)


def main():
    parser = argparse.ArgumentParser(
        description="JSSPLA Optimization: SAC-PSO Dynamic Control vs PSO, GA, FIFO, SPT, LPT"
    )
    parser.add_argument("--demo", action="store_true", help="Run quick demo on Scenario W1")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark on all 9 standard instances (W1-W9)")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per scenario for benchmark (default: 10)")
    parser.add_argument("--train", action="store_true", help="Train the SAC agent (3-stage curriculum)")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes (default: 300)")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate benchmark charts/tables from JSON")
    args = parser.parse_args()

    print("\n" + "=" * 95)
    print("   JOB SHOP SCHEDULING WITH LIMITED AMRS (JSSPLA) IN MULTI-AMR WAREHOUSE")
    print("   SAC-PSO (6-DIM STATE, 3-DIM ACTION: w, c1, c2) vs PSO, GA, FIFO, SPT, LPT")
    print("=" * 95)

    model_path = os.path.join("models", "sac_basic_pso.pt")
    results_dir = os.path.join("results")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.train:
        print("\n[ACTION] Starting SAC Agent Training Pipeline...")
        train_basic_sac(num_episodes=args.episodes, save_model_path=model_path)
        return

    if not os.path.exists(model_path):
        print(f"[WARNING] Model weights '{model_path}' not found!")
        print("Training SAC-PSO agent automatically...")
        train_basic_sac(num_episodes=100, save_model_path=model_path)

    if args.plot_only:
        import json
        json_path = os.path.join(results_dir, "benchmark_results.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            generate_outputs(data["results"], data["times"], results_dir)
        else:
            print(f"[ERROR] Cannot find {json_path} to plot.")
        return

    if args.benchmark:
        print(f"\n[ACTION] Running Benchmark on 9 Standard Instances ({args.runs} runs)...")
        run_benchmark(num_runs=args.runs, model_path=model_path, results_dir=results_dir)
        return

    # Default action: run quick demo and display summary of available artifacts
    run_quick_demo(model_path=model_path)

    print("\nAvailable benchmark outputs in 'results/':")
    print("  - Grouped Bar Chart          : results/barchart_standard_scenarios.png")
    print("  - Small Scenario Boxplot     : results/boxplot_small_scenario.png")
    print("  - Medium Scenario Boxplot    : results/boxplot_medium_scenario.png")
    print("  - Large Scenario Boxplot     : results/boxplot_large_scenario.png")
    print("  - SAC Training Curves        : results/sac_basic_training.png")
    print("  - Metrics Table (Cmax, SD)   : results/table_metrics_cmax_sd_cputime.md")
    print("  - Full Summary Table         : results/table_standard_scenarios.md")
    print("\nTip: Run with '--benchmark' to run 10-run benchmarks, or '--train' to retrain.")


if __name__ == "__main__":
    main()