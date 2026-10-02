"""
Full Experimental Benchmark Suite:
Evaluates FIFO, SPT, LPT, GA, PSO, and Proposed SAC-PSO
across Scenario 1 (24J), Scenario 2 (28J), Scenario 3 (28J), and Scenario 4 (350J).
Generates publication-quality comparison tables, grouped bar charts, boxplots, and convergence graphs.
"""

import time
import os
import sys
import io

# Ensure UTF-8 output encoding in Windows console
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np

# Set clean matplotlib fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

from warehouse_env import WarehouseInstance
from baselines import solve_fifo, solve_spt, solve_lpt, StandardGA, StandardPSO
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
    scenario_short_names = {
        1: "Scenario 1\n(24 Jobs)",
        2: "Scenario 2\n(28 Jobs A)",
        3: "Scenario 3\n(28 Jobs B)",
        4: "Scenario 4\n(350 Jobs)",
    }

    # Algorithms: Exclude CDPSO, only FIFO, SPT, LPT, GA, PSO, SAC-PSO
    algorithms = ["FIFO", "SPT", "LPT", "GA", "PSO", "SAC-PSO"]
    meta_algorithms = ["GA", "PSO", "SAC-PSO"]

    all_results = {}
    box_data = {sc: {"GA": [], "PSO": [], "SAC-PSO": []} for sc in scenarios}
    conv_runs_sc4 = {"GA": [], "PSO": [], "SAC-PSO (Ours)": []}

    print("=" * 95)
    print("STARTING BENCHMARK EVALUATION (FIFO, SPT, LPT, GA, PSO, SAC-PSO)")
    print(f"Configurations: Runs={num_runs}, PopSize={pop_size}, MaxIter={max_iter}")
    print("=" * 95)

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

        all_results[(sc_id, "FIFO")] = {
            "best": m_fifo, "avg": m_fifo, "worst": m_fifo,
            "sd_val": 0.0, "sd_pct": 0.0, "time": t_fifo, "runs": [m_fifo] * num_runs
        }
        all_results[(sc_id, "SPT")] = {
            "best": m_spt, "avg": m_spt, "worst": m_spt,
            "sd_val": 0.0, "sd_pct": 0.0, "time": t_spt, "runs": [m_spt] * num_runs
        }
        all_results[(sc_id, "LPT")] = {
            "best": m_lpt, "avg": m_lpt, "worst": m_lpt,
            "sd_val": 0.0, "sd_pct": 0.0, "time": t_lpt, "runs": [m_lpt] * num_runs
        }

        # 2. Metaheuristics over independent runs
        ga_vals, pso_vals, sac_vals = [], [], []
        t_ga_list, t_pso_list, t_sac_list = [], [], []

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

            # SAC-PSO (Ours)
            t0 = time.time()
            sac = SACPSOSolver(model_path=model_path, num_particles=pop_size, max_iter=max_iter, seed=run_seed)
            m_sac, _, h_sac, _ = sac.solve(inst)
            sac_vals.append(m_sac)
            t_sac_list.append(time.time() - t0)

            if sc_id == 4:
                conv_runs_sc4["GA"].append(h_ga)
                conv_runs_sc4["PSO"].append(h_pso)
                conv_runs_sc4["SAC-PSO (Ours)"].append(h_sac)

        # Save boxplot data
        box_data[sc_id]["GA"] = ga_vals
        box_data[sc_id]["PSO"] = pso_vals
        box_data[sc_id]["SAC-PSO"] = sac_vals

        def calc_stats(vals, times):
            best = min(vals)
            avg = float(np.mean(vals))
            worst = max(vals)
            sd_val = float(np.std(vals))
            sd_pct = (sd_val / max(1e-5, avg)) * 100.0
            t_avg = float(np.mean(times))
            return {
                "best": best,
                "avg": avg,
                "worst": worst,
                "sd_val": sd_val,
                "sd_pct": sd_pct,
                "time": t_avg,
                "runs": vals
            }

        all_results[(sc_id, "GA")] = calc_stats(ga_vals, t_ga_list)
        all_results[(sc_id, "PSO")] = calc_stats(pso_vals, t_pso_list)
        all_results[(sc_id, "SAC-PSO")] = calc_stats(sac_vals, t_sac_list)

    # 3. Print & Save Detailed Comparison Table
    print("\n" + "=" * 115)
    print("TABLE: COMPARISON OF C_max AND STABILITY (DO ON DINH) ACROSS 4 SCENARIOS")
    print("=" * 115)
    header = f"{'Scenario':<12} | {'Algorithm':<10} | {'C_max Best(s)':<14} | {'C_max Avg(s)':<14} | {'C_max Worst(s)':<14} | {'SD (s)':<10} | {'SD (%)':<10} | {'Imp(%)':<9} | {'Time(s)':<8}"
    print(header)
    print("-" * 115)

    table_lines = [
        "# Kết quả Thử nghiệm So sánh 4 Instance (Scenarios 1 - 4)",
        "",
        "Đánh giá hiệu năng và độ ổn định của các thuật toán: FIFO, SPT, LPT, GA, PSO, SAC-PSO.",
        "- **Độ ổn định**: Được lượng hóa qua độ lệch chuẩn $SD$ (s) và hệ số biến thiên $SD$ (%). Giá trị càng nhỏ thể hiện thuật toán hội tụ càng nhất quán và ổn định qua nhiều lần chạy độc lập.",
        "- **FIFO, SPT, LPT** là thuật toán heuristic đơn định (deterministic) nên $SD = 0.00$ (ổn định tuyệt đối).",
        "",
        "| Scenario | Thuật toán | C_max Best (s) | C_max Avg (s) | C_max Worst (s) | SD (s) | SD (%) (Độ ổn định) | Cải thiện FIFO (%) | Thời gian (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    scenario_labels = {
        1: "Scenario 1 (24 Jobs)",
        2: "Scenario 2 (28 Jobs A)",
        3: "Scenario 3 (28 Jobs B)",
        4: "Scenario 4 (350 Jobs)",
    }

    for sc_id in scenarios:
        fifo_avg = all_results[(sc_id, "FIFO")]["avg"]
        for algo in algorithms:
            res = all_results[(sc_id, algo)]
            imp = ((fifo_avg - res["best"]) / fifo_avg) * 100.0
            stability_str_cli = f"{res['sd_pct']:.2f}%" if algo in meta_algorithms else "0.00% (Det)"
            stability_str_md = f"{res['sd_pct']:.2f}%" if algo in meta_algorithms else "0.00% (Đơn định)"
            row_str = (
                f"{sc_id:<12} | {algo:<10} | {res['best']:<14.2f} | {res['avg']:<14.2f} | "
                f"{res['worst']:<14.2f} | {res['sd_val']:<10.2f} | {stability_str_cli:<10} | "
                f"{imp:<9.2f} | {res['time']:<8.3f}"
            )
            print(row_str)
            table_lines.append(
                f"| {scenario_labels[sc_id]} | **{algo}** | "
                f"{res['best']:.2f} | {res['avg']:.2f} | {res['worst']:.2f} | "
                f"{res['sd_val']:.2f} | {stability_str_md} | {imp:.2f}% | {res['time']:.3f} |"
            )
        print("-" * 115)

    with open("results/table_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    print("-> Table saved to: results/table_comparison.md")

    # Also save compact Table 1 format (results/table1_comparison.md) without CDPSO
    table1_lines = [
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
            table1_lines.append(
                f"| {sc_id} | {algo} | {res['best']:.2f} | {res['avg']:.2f} | "
                f"{res['sd_pct']:.2f} | {imp:.2f}% | {res['time']:.3f} |"
            )
    with open("results/table1_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(table1_lines))
    print("-> Compact Table 1 saved to: results/table1_comparison.md")

    # 4. PLOT 1: Grouped Bar Chart of ALL 4 Instances (Dual-panel split for optimal readability)
    # Left: Small/Medium (Scenarios 1-3), Right: Large (Scenario 4)
    colors = {
        "FIFO": "#7f8c8d",
        "SPT": "#e67e22",
        "LPT": "#9b59b6",
        "GA": "#3498db",
        "PSO": "#2ecc71",
        "SAC-PSO": "#e74c3c",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1.4]})
    plt.subplots_adjust(wspace=0.25)

    bar_width = 0.13
    x_sc123 = np.arange(3)
    offsets = np.linspace(-2.5 * bar_width, 2.5 * bar_width, len(algorithms))

    # --- Subplot 1: Scenarios 1, 2, 3 ---
    for i, algo in enumerate(algorithms):
        means = [all_results[(sc, algo)]["avg"] for sc in [1, 2, 3]]
        sds = [all_results[(sc, algo)]["sd_val"] for sc in [1, 2, 3]]
        bars = ax1.bar(
            x_sc123 + offsets[i],
            means,
            bar_width,
            yerr=sds if algo in meta_algorithms else None,
            capsize=3,
            label=algo,
            color=colors[algo],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        # Add value labels
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
                fontweight="bold",
            )

    ax1.set_title("Scenarios 1 - 3 (Small & Medium Instances)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Instance / Scenario", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Makespan C_max (seconds) [Thấp hơn là tốt hơn]", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_sc123)
    ax1.set_xticklabels([scenario_short_names[s] for s in [1, 2, 3]], fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(all_results[(s, "LPT")]["avg"] for s in [1, 2, 3]) * 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # --- Subplot 2: Scenario 4 (Large-Scale) ---
    x_sc4 = np.array([0])
    for i, algo in enumerate(algorithms):
        mean = [all_results[(4, algo)]["avg"]]
        sd = [all_results[(4, algo)]["sd_val"]]
        bars = ax2.bar(
            x_sc4 + offsets[i],
            mean,
            bar_width,
            yerr=sd if algo in meta_algorithms else None,
            capsize=3,
            label=algo,
            color=colors[algo],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
                fontweight="bold",
            )

    ax2.set_title("Scenario 4 (Large: 350 Jobs)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Instance / Scenario", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_sc4)
    ax2.set_xticklabels([scenario_short_names[4]], fontsize=10, fontweight="bold")
    ax2.set_ylim(0, all_results[(4, "LPT")]["avg"] * 1.18)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        "So sánh Makespan C_max giữa các thuật toán trên 4 Instance (Có thanh đo sai số SD - Độ ổn định)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("results/barchart_all_instances.png", dpi=300)
    plt.close()
    print("-> Grouped Bar Chart saved to: results/barchart_all_instances.png")

    # 5. PLOT 2: Single-axes Grouped Bar Chart across ALL 4 instances (unified view)
    fig, ax = plt.subplots(figsize=(15, 7))
    x_all = np.arange(4)
    for i, algo in enumerate(algorithms):
        means = [all_results[(sc, algo)]["avg"] for sc in scenarios]
        sds = [all_results[(sc, algo)]["sd_val"] for sc in scenarios]
        bars = ax.bar(
            x_all + offsets[i],
            means,
            bar_width,
            yerr=sds if algo in meta_algorithms else None,
            capsize=3,
            label=algo,
            color=colors[algo],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=45,
                fontweight="bold",
            )

    ax.set_title("Biểu đồ cột Makespan C_max tổng hợp 4 Instance (FIFO, SPT, LPT, GA, PSO, SAC-PSO)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Instance", fontsize=11, fontweight="bold")
    ax.set_ylabel("Makespan C_max (seconds)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_all)
    ax.set_xticklabels([f"Scenario {s}\n({inst_label})" for s, inst_label in [(1, '24 Jobs'), (2, '28 Jobs A'), (3, '28 Jobs B'), (4, '350 Jobs')]], fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/barchart_unified_grouped.png", dpi=300)
    plt.close()
    print("-> Unified Grouped Bar Chart saved to: results/barchart_unified_grouped.png")

    # 6. PLOT 3: 2x2 Grid of Bar Charts (1 Subplot per Instance - Perfect Per-Instance Resolution)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, sc_id in enumerate(scenarios):
        ax = axes[idx]
        x_pos = np.arange(len(algorithms))
        means = [all_results[(sc_id, algo)]["avg"] for algo in algorithms]
        sds = [all_results[(sc_id, algo)]["sd_val"] for algo in algorithms]
        bar_colors = [colors[algo] for algo in algorithms]

        bars = ax.bar(
            x_pos,
            means,
            width=0.55,
            yerr=[sds[i] if algo in meta_algorithms else np.nan for i, algo in enumerate(algorithms)],
            capsize=4,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{scenario_names[sc_id]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Makespan C_max (s)", fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, max(means) * 1.16)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        "So sánh Chi tiết Makespan C_max và Độ ổn định (Thanh SD) trên từng Instance",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("results/barchart_4instances_grid.png", dpi=300)
    plt.close()
    print("-> 2x2 Grid Bar Chart saved to: results/barchart_4instances_grid.png")

    # 7. PLOT 4: Boxplots of Metaheuristics (GA, PSO, SAC-PSO) - Stability Visualizer
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for idx, sc_id in enumerate(scenarios):
        ax = axes[idx]
        data = [box_data[sc_id]["GA"], box_data[sc_id]["PSO"], box_data[sc_id]["SAC-PSO"]]
        labels = ["Standard GA", "Standard PSO", "SAC-PSO (Ours)"]
        bp_colors = ["#3498db", "#2ecc71", "#e74c3c"]

        bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True)
        for patch, color in zip(bplot["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"Scenario {sc_id} - Độ ổn định phân bố C_max", fontsize=11, fontweight="bold")
        ax.set_ylabel("Makespan C_max (s)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Độ ổn định Makespan qua các lần chạy (GA vs PSO vs SAC-PSO)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/boxplots_comparison.png", dpi=300)
    plt.close()
    print("-> Boxplots saved to: results/boxplots_comparison.png")

    # 8. PLOT 5: Scenario 4 Convergence Curve (Average across runs)
    if conv_runs_sc4 and any(conv_runs_sc4.values()):
        plt.figure(figsize=(10, 5))
        for algo, runs in conv_runs_sc4.items():
            if runs:
                mean_hist = np.mean(runs, axis=0)
                if "GA" in algo:
                    color, ls = "#3498db", "-"
                elif "PSO" in algo and "SAC" not in algo:
                    color, ls = "#2ecc71", "--"
                else:
                    color, ls = "#e74c3c", "-"
                plt.plot(mean_hist, label=algo, linewidth=2.2, color=color, linestyle=ls)
        plt.title("Quá trình hội tụ trung bình trên Scenario 4 (350 Jobs / 750 Ops - 5 Runs)", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration", fontsize=11, fontweight="bold")
        plt.ylabel("Makespan C_max trung bình (s)", fontsize=11, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig("results/scenario4_convergence.png", dpi=300)
        plt.close()
        print("-> Convergence curve saved to: results/scenario4_convergence.png")



if __name__ == "__main__":
    run_full_benchmarks(num_runs=5, pop_size=30, max_iter=25)

