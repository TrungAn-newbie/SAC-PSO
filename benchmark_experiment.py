"""
Full Experimental Benchmark Suite (matching Paper Table 1 and Figures 3-6):
Evaluates FIFO, SPT, LPT, GA, PSO, CDPSO (Paper), and SAC-PSO (Ours)
across Scenario 1 (24J), Scenario 2 (28J), Scenario 3 (28J), and Scenario 4 (350J).
Generates publication-quality comparison tables, boxplots, and convergence graphs.
"""

import time
import os
import matplotlib.pyplot as plt
import numpy as np

from warehouse_env import WarehouseInstance
from baselines import solve_fifo, solve_spt, solve_lpt, StandardGA, StandardPSO, CDPSO
from sac_pso import SACPSOSolver


def run_full_benchmarks(
    num_runs: int = 5,
    pop_size: int = 30,
    max_iter: int = 25,
    model_path: str = "models/sac_warehouse_pso.pt",
):
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    scenarios = [1, 2, 3, 4]
    scenario_names = {
        1: "Scenario 1 (24 Jobs: 8 In, 16 Out)",
        2: "Scenario 2 (28 Jobs: 4 In, 24 Out)",
        3: "Scenario 3 (28 Jobs: 20 In, 8 Out)",
        4: "Scenario 4 (350 Jobs: 150 In, 200 Out - Large)",
    }

    algorithms = ["FIFO", "SPT", "LPT", "GA", "PSO", "CDPSO", "SAC-PSO"]
    all_results = {}
    box_data = {sc: {"GA": [], "PSO": [], "CDPSO": [], "SAC-PSO": []} for sc in scenarios}
    conv_history_sc4 = {}

    print("=" * 85)
    print("STARTING COMPREHENSIVE BENCHMARK EVALUATION ACROSS ALL SCENARIOS")
    print("=" * 85)

    for sc_id in scenarios:
        inst = WarehouseInstance.create_scenario(sc_id, seed=42)
        print(f"\n---> Evaluating {scenario_names[sc_id]} ({inst.total_operations} operations)...")

        # 1. Deterministic heuristics
        t0 = time.time()
        m_fifo, _ = solve_fifo(inst)
        t_fifo = time.time() - t0

        t0 = time.time()
        m_spt, _ = solve_spt(inst)
        t_spt = time.time() - t0

        t0 = time.time()
        m_lpt, _ = solve_lpt(inst)
        t_lpt = time.time() - t0

        all_results[(sc_id, "FIFO")] = {"best": m_fifo, "avg": m_fifo, "sd": 0.0, "time": t_fifo}
        all_results[(sc_id, "SPT")] = {"best": m_spt, "avg": m_spt, "sd": 0.0, "time": t_spt}
        all_results[(sc_id, "LPT")] = {"best": m_lpt, "avg": m_lpt, "sd": 0.0, "time": t_lpt}

        # 2. Metaheuristics over independent runs
        ga_vals, pso_vals, cdpso_vals, sac_vals = [], [], [], []
        t_ga_list, t_pso_list, t_cdpso_list, t_sac_list = [], [], [], []

        for r in range(num_runs):
            run_seed = 42 + r * 13

            # GA
            t0 = time.time()
            ga = StandardGA(pop_size=pop_size, max_gen=max_iter, seed=run_seed)
            m_ga, _, h_ga = ga.solve(inst)
            ga_vals.append(m_ga)
            t_ga_list.append(time.time() - t0)

            # PSO
            t0 = time.time()
            pso = StandardPSO(num_particles=pop_size, max_iter=max_iter, seed=run_seed)
            m_pso, _, h_pso = pso.solve(inst)
            pso_vals.append(m_pso)
            t_pso_list.append(time.time() - t0)

            # CDPSO
            t0 = time.time()
            cdpso = CDPSO(pop_size=pop_size, max_iter=max_iter, seed=run_seed)
            m_cdpso, _, h_cdpso = cdpso.solve(inst)
            cdpso_vals.append(m_cdpso)
            t_cdpso_list.append(time.time() - t0)

            # SAC-PSO (Ours)
            t0 = time.time()
            sac = SACPSOSolver(model_path=model_path, num_particles=pop_size, max_iter=max_iter, seed=run_seed)
            m_sac, _, h_sac, _ = sac.solve(inst)
            sac_vals.append(m_sac)
            t_sac_list.append(time.time() - t0)

            if sc_id == 4 and r == 0:
                conv_history_sc4["GA"] = h_ga
                conv_history_sc4["PSO"] = h_pso
                conv_history_sc4["CDPSO"] = h_cdpso
                conv_history_sc4["SAC-PSO (Ours)"] = h_sac

        # Save boxplot data
        box_data[sc_id]["GA"] = ga_vals
        box_data[sc_id]["PSO"] = pso_vals
        box_data[sc_id]["CDPSO"] = cdpso_vals
        box_data[sc_id]["SAC-PSO"] = sac_vals

        def calc_stats(vals, times):
            best = min(vals)
            avg = float(np.mean(vals))
            sd = float(np.std(vals)) / max(1e-5, avg) * 100.0
            t_avg = float(np.mean(times))
            return {"best": best, "avg": avg, "sd": sd, "time": t_avg}

        all_results[(sc_id, "GA")] = calc_stats(ga_vals, t_ga_list)
        all_results[(sc_id, "PSO")] = calc_stats(pso_vals, t_pso_list)
        all_results[(sc_id, "CDPSO")] = calc_stats(cdpso_vals, t_cdpso_list)
        all_results[(sc_id, "SAC-PSO")] = calc_stats(sac_vals, t_sac_list)

    # 3. Print & Save Table (Matching Table 1 in paper)
    print("\n" + "=" * 95)
    print("TABLE 1: COMPARISON OF HEURISTIC, METAHEURISTIC AND PROPOSED SAC-PSO")
    print("=" * 95)
    print(f"{'Scenario':<10} | {'Algorithm':<12} | {'C_max best (s)':<15} | {'C_max avg (s)':<15} | {'SD (%)':<8} | {'Baseline Imp (%)':<18} | {'Time (s)':<8}")
    print("-" * 95)

    table_lines = [
        "# Table 1: Comparison of Heuristic, Metaheuristic and Proposed SAC-PSO",
        "",
        "| Scenario | Algorithm | C_max best (s) | C_max avg (s) | SD (%) | Baseline Imp (%) | Time (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for sc_id in scenarios:
        fifo_avg = all_results[(sc_id, "FIFO")]["avg"]
        for algo in algorithms:
            res = all_results[(sc_id, algo)]
            imp = ((fifo_avg - res["best"]) / fifo_avg) * 100.0
            row_str = f"{sc_id:<10} | {algo:<12} | {res['best']:<15.2f} | {res['avg']:<15.2f} | {res['sd']:<8.2f} | {imp:<18.2f} | {res['time']:<8.3f}"
            print(row_str)
            table_lines.append(f"| {sc_id} | {algo} | {res['best']:.2f} | {res['avg']:.2f} | {res['sd']:.2f} | {imp:.2f}% | {res['time']:.3f} |")
        print("-" * 95)

    with open("results/table1_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    print("-> Table saved to: results/table1_comparison.md")

    # 4. Plot Boxplots (Matching Figures 3, 4, 5, 6 in paper)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sc_id in enumerate(scenarios):
        ax = axes[idx]
        data = [box_data[sc_id]["GA"], box_data[sc_id]["PSO"], box_data[sc_id]["CDPSO"], box_data[sc_id]["SAC-PSO"]]
        labels = ["Standard GA", "Standard PSO", "CDPSO (Paper)", "SAC-PSO (Ours)"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels)
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"Figure {idx+3}: Box plot of Scenario {sc_id}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Makespan C_max (s)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/boxplots_comparison.png", dpi=300)
    plt.close()
    print("-> Boxplots saved to: results/boxplots_comparison.png")

    # 5. Plot Scenario 4 Convergence Curve
    if conv_history_sc4:
        plt.figure(figsize=(10, 5))
        for algo, hist in conv_history_sc4.items():
            plt.plot(hist, label=algo, linewidth=2.0)
        plt.title("Convergence Comparison on Scenario 4 (350 Jobs / 750 Ops)", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration", fontsize=11, fontweight="bold")
        plt.ylabel("Makespan C_max (s)", fontsize=11, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig("results/scenario4_convergence.png", dpi=300)
        plt.close()
        print("-> Scenario 4 convergence plot saved to: results/scenario4_convergence.png")


if __name__ == "__main__":
    run_full_benchmarks(num_runs=5, pop_size=30, max_iter=25)
