"""
Standard Benchmark Experiment for Basic SAC-PSO (6-dim State, 3-dim Action: w, c1, c2).
Evaluates 6 Algorithms:
- Deterministic: FIFO, SPT, LPT
- Metaheuristics: GA, PSO, SAC-PSO (Ours - 6-dim State Model)
Across 9 Standard Warehouse Instances (W1 to W9).

Generates:
1. Grouped Bar Chart (barchart_standard_scenarios.png)
2. Box Plots for Small, Medium, Large scenarios (boxplot_small_scenario.png, boxplot_medium_scenario.png, boxplot_large_scenario.png)
3. Target Markdown Table: table_metrics_cmax_sd_cputime.md
4. Summary Markdown Table: table_standard_scenarios.md
All figures strictly use Times New Roman typography and contain NO titles.
"""

import os
import sys
import io
import time
import json
import argparse
import concurrent.futures
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding in Windows console
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from warehouse_env import WarehouseInstance
from baselines import solve_fifo, solve_spt, solve_lpt, StandardGA, StandardPSO
from pso_env import BasicSACPSOSolver

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


STANDARD_SCENARIOS = [
    # Small (50 jobs / 100 ops)
    {
        "code": "W1", "scale": "Small (50 jobs)", "name": "W1_Small_In25_Out25",
        "nin": 25, "nout": 25, "nops": 100, "seed": 101,
        "flow_type": "Cân bằng (In = Out)",
        "desc": "Small: 50 đơn hàng (25 In, 25 Out) - Cân bằng luồng"
    },
    {
        "code": "W2", "scale": "Small (50 jobs)", "name": "W2_Small_In15_Out35",
        "nin": 15, "nout": 35, "nops": 100, "seed": 102,
        "flow_type": "Thiên xuất (In < Out)",
        "desc": "Small: 50 đơn hàng (15 In, 35 Out) - Tải xuất kho cao"
    },
    {
        "code": "W3", "scale": "Small (50 jobs)", "name": "W3_Small_In35_Out15",
        "nin": 35, "nout": 15, "nops": 100, "seed": 103,
        "flow_type": "Thiên nhập (In > Out)",
        "desc": "Small: 50 đơn hàng (35 In, 15 Out) - Tải nhập kho cao"
    },
    # Medium (150 jobs / 300 ops)
    {
        "code": "W4", "scale": "Medium (150 jobs)", "name": "W4_Medium_In75_Out75",
        "nin": 75, "nout": 75, "nops": 300, "seed": 201,
        "flow_type": "Cân bằng (In = Out)",
        "desc": "Medium: 150 đơn hàng (75 In, 75 Out) - Cân bằng luồng"
    },
    {
        "code": "W5", "scale": "Medium (150 jobs)", "name": "W5_Medium_In45_Out105",
        "nin": 45, "nout": 105, "nops": 300, "seed": 202,
        "flow_type": "Thiên xuất (In < Out)",
        "desc": "Medium: 150 đơn hàng (45 In, 105 Out) - Tải xuất kho cao"
    },
    {
        "code": "W6", "scale": "Medium (150 jobs)", "name": "W6_Medium_In105_Out45",
        "nin": 105, "nout": 45, "nops": 300, "seed": 203,
        "flow_type": "Thiên nhập (In > Out)",
        "desc": "Medium: 150 đơn hàng (105 In, 45 Out) - Tải nhập kho cao"
    },
    # Large (300 jobs / 600 ops)
    {
        "code": "W7", "scale": "Large (300 jobs)", "name": "W7_Large_In150_Out150",
        "nin": 150, "nout": 150, "nops": 600, "seed": 301,
        "flow_type": "Cân bằng (In = Out)",
        "desc": "Large: 300 đơn hàng (150 In, 150 Out) - Cân bằng luồng"
    },
    {
        "code": "W8", "scale": "Large (300 jobs)", "name": "W8_Large_In90_Out210",
        "nin": 90, "nout": 210, "nops": 600, "seed": 302,
        "flow_type": "Thiên xuất (In < Out)",
        "desc": "Large: 300 đơn hàng (90 In, 210 Out) - Tải xuất kho cao"
    },
    {
        "code": "W9", "scale": "Large (300 jobs)", "name": "W9_Large_In210_Out90",
        "nin": 210, "nout": 90, "nops": 600, "seed": 303,
        "flow_type": "Thiên nhập (In > Out)",
        "desc": "Large: 300 đơn hàng (210 In, 90 Out) - Tải nhập kho cao"
    },
]


def _worker_eval(args):
    code, algo, sc_nin, sc_nout, sc_name, sc_seed, seed, pop_size, max_iter, model_path = args
    inst = WarehouseInstance.generate(num_inbound=sc_nin, num_outbound=sc_nout, name=sc_name, seed=sc_seed)
    t0 = time.time()
    if algo == "GA":
        solver = StandardGA(pop_size=pop_size, max_gen=max_iter, seed=seed)
        c, _, _ = solver.solve(inst)
    elif algo == "PSO":
        solver = StandardPSO(num_particles=pop_size, max_iter=max_iter, seed=seed)
        c, _, _ = solver.solve(inst)
    elif algo == "SAC-PSO":
        solver = BasicSACPSOSolver(model_path=model_path, num_particles=pop_size, max_iter=max_iter, seed=seed, verbose=False)
        c, _, _, _ = solver.solve(inst)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    dt = time.time() - t0
    return code, algo, seed, c, dt


def run_benchmark(
    num_runs: int = 10,
    pop_size: int = 50,
    max_iter: int = 50,
    model_path: str = os.path.join(os.path.dirname(__file__), "models", "sac_basic_pso.pt"),
    results_dir: str = os.path.join(os.path.dirname(__file__), "results"),
):
    os.makedirs(results_dir, exist_ok=True)
    seeds = [42 + i * 73 for i in range(num_runs)]

    all_algorithms = ["FIFO", "SPT", "LPT", "GA", "PSO", "SAC-PSO"]
    meta_algorithms = ["GA", "PSO", "SAC-PSO"]

    print("=" * 100)
    print("BENCHMARK EXPERIMENT: 6 ALGORITHMS ON 9 STANDARD INSTANCES (W1 - W9)")
    print(f"Algorithms: {', '.join(all_algorithms)}")
    print(f"SAC-PSO Model: {model_path}")
    print(f"Configuration: {num_runs} runs per instance | Swarm: {pop_size} particles, {max_iter} iterations")
    print("=" * 100)

    results_data = {sc["code"]: {algo: [] for algo in all_algorithms} for sc in STANDARD_SCENARIOS}
    times_data = {sc["code"]: {algo: [] for algo in all_algorithms} for sc in STANDARD_SCENARIOS}

    # 1. Deterministic Heuristics
    print("\n[PHASE 1] Evaluating Deterministic Baselines (FIFO, SPT, LPT)...")
    for sc in STANDARD_SCENARIOS:
        code = sc["code"]
        inst = WarehouseInstance.generate(num_inbound=sc["nin"], num_outbound=sc["nout"], name=sc["name"], seed=sc["seed"])

        t0 = time.time()
        c_fifo, _ = solve_fifo(inst)
        t_fifo = time.time() - t0
        results_data[code]["FIFO"] = [c_fifo] * num_runs
        times_data[code]["FIFO"] = [t_fifo] * num_runs

        t0 = time.time()
        c_spt, _ = solve_spt(inst)
        t_spt = time.time() - t0
        results_data[code]["SPT"] = [c_spt] * num_runs
        times_data[code]["SPT"] = [t_spt] * num_runs

        t0 = time.time()
        c_lpt, _ = solve_lpt(inst)
        t_lpt = time.time() - t0
        results_data[code]["LPT"] = [c_lpt] * num_runs
        times_data[code]["LPT"] = [t_lpt] * num_runs

        print(f"  [{code}] FIFO={c_fifo:8.2f}s | SPT={c_spt:8.2f}s | LPT={c_lpt:8.2f}s")

    # 2. Parallel Metaheuristics
    tasks = []
    for sc in STANDARD_SCENARIOS:
        for algo in meta_algorithms:
            for s in seeds:
                tasks.append((
                    sc["code"], algo, sc["nin"], sc["nout"], sc["name"], sc["seed"],
                    s, pop_size, max_iter, model_path
                ))

    n_workers = min(10, (os.cpu_count() or 4))
    print(f"\n[PHASE 2] Running {len(tasks)} metaheuristic evaluations with {n_workers} processes (ProcessPool)...")
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, (code, algo, seed, c, dt) in enumerate(executor.map(_worker_eval, tasks), 1):
            results_data[code][algo].append(c)
            times_data[code][algo].append(dt)
            if idx % 15 == 0 or idx == len(tasks):
                elapsed = time.time() - t_start
                print(f"  Progress: {idx}/{len(tasks)} ({idx/len(tasks)*100:.1f}%) in {elapsed:.1f}s")

    print(f"\nAll evaluations completed in {time.time()-t_start:.1f}s!")

    # Save raw json
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results_data, "times": times_data}, f, indent=2)

    generate_outputs(results_data, times_data, results_dir)
    return results_data, times_data


def generate_outputs(results_data, times_data, results_dir):
    all_algorithms = ["FIFO", "SPT", "LPT", "GA", "PSO", "SAC-PSO"]
    meta_algorithms = ["GA", "PSO", "SAC-PSO"]
    num_runs = len(results_data[STANDARD_SCENARIOS[0]["code"]]["FIFO"])

    # ---------------------------------------------------------------------------
    # 3. Generate Grouped Bar Chart (NO Title)
    # ---------------------------------------------------------------------------
    print("\n[PLOTTING] Generating Grouped Bar Chart for W1 - W9 (No Title)...")
    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#d3d3d3", zorder=0)

    bar_colors = {
        "FIFO": "#7f8c8d",
        "SPT": "#95a5a6",
        "LPT": "#bdc3c7",
        "GA": "#e67e22",
        "PSO": "#2980b9",
        "SAC-PSO": "#27ae60",
    }

    n_sc = len(STANDARD_SCENARIOS)
    n_alg = len(all_algorithms)
    bar_width = 0.13
    x = np.arange(n_sc)

    for i, algo in enumerate(all_algorithms):
        means = [np.mean(results_data[sc["code"]][algo]) for sc in STANDARD_SCENARIOS]
        stds = [np.std(results_data[sc["code"]][algo]) for sc in STANDARD_SCENARIOS]
        pos = x - (n_alg - 1) * bar_width / 2.0 + i * bar_width

        if algo in ["FIFO", "SPT", "LPT"]:
            bars = ax.bar(pos, means, bar_width, label=algo, color=bar_colors[algo], edgecolor="black", linewidth=0.7, zorder=3)
        else:
            bars = ax.bar(pos, means, bar_width, yerr=stds, capsize=3.0, label=algo, color=bar_colors[algo], edgecolor="black", linewidth=0.7, zorder=3)

        for bar, m in zip(bars, means):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 150,
                f"{m:.0f}",
                ha="center", va="bottom", fontsize=7.5, rotation=60, fontweight="bold" if algo == "SAC-PSO" else "normal"
            )

    ax.axvline(2.5, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(5.5, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([sc["code"] for sc in STANDARD_SCENARIOS], fontsize=13, fontweight="bold")
    ax.set_ylabel("Makespan (s)", fontsize=13, fontweight="bold")
    max_c = max([max(results_data[sc["code"]]["LPT"]) for sc in STANDARD_SCENARIOS])
    ax.set_ylim(0, max_c * 1.15)
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10.5)

    plt.tight_layout()
    barchart_path = os.path.join(results_dir, "barchart_standard_scenarios.png")
    plt.savefig(barchart_path, dpi=300)
    plt.close()
    print(f"Saved bar chart to: {barchart_path}")

    # ---------------------------------------------------------------------------
    # 4. Generate 3 Separate Box Plots for Small, Medium, Large Scenarios (NO Title)
    # ---------------------------------------------------------------------------
    print("\n[PLOTTING] Generating 3 Separate Box Plots for Small, Medium, Large (No Title)...")
    scenarios_groups = [
        ("Small", ["W1", "W2", "W3"], os.path.join(results_dir, "boxplot_small_scenario.png")),
        ("Medium", ["W4", "W5", "W6"], os.path.join(results_dir, "boxplot_medium_scenario.png")),
        ("Large", ["W7", "W8", "W9"], os.path.join(results_dir, "boxplot_large_scenario.png")),
    ]

    box_colors = {
        "GA": "#f39c12",
        "PSO": "#3498db",
        "SAC-PSO": "#2ecc71"
    }

    for group_name, codes, out_path in scenarios_groups:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#ffffff")

        for idx, code in enumerate(codes):
            ax = axs[idx]
            ax.set_facecolor("#ffffff")
            ax.grid(axis="y", linestyle="--", alpha=0.5, color="#d3d3d3", zorder=0)

            data_to_plot = [results_data[code][algo] for algo in meta_algorithms]

            bplot = ax.boxplot(
                data_to_plot,
                patch_artist=True,
                widths=0.55,
                whis=(0, 100),
                showfliers=False,
                showmeans=False,
                medianprops=dict(color="#1a1a1a", linewidth=2.0),
                whiskerprops=dict(color="#333333", linewidth=1.2),
                capprops=dict(color="#333333", linewidth=1.2),
                zorder=3
            )

            for a_idx, (patch, algo) in enumerate(zip(bplot["boxes"], meta_algorithms), start=1):
                patch.set_facecolor(box_colors[algo])
                patch.set_alpha(0.75)
                patch.set_edgecolor("#2c3e50")
                patch.set_linewidth(1.3)

                vals = results_data[code][algo]
                q25, q75 = np.percentile(vals, [25, 75])
                # If box height is zero (IQR == 0), draw solid indicator bar so algorithm level is distinct
                if abs(q75 - q25) < 1e-4:
                    ax.hlines(
                        y=np.median(vals),
                        xmin=a_idx - 0.275,
                        xmax=a_idx + 0.275,
                        color=box_colors[algo],
                        linewidth=5.0,
                        alpha=0.95,
                        zorder=3
                    )

            # Balanced y-axis limits to prevent distorted zooming on tight-variance instances (W3, W6, W9)
            all_vals = [v for a in meta_algorithms for v in results_data[code][a]]
            y_min, y_max = min(all_vals), max(all_vals)
            y_span = y_max - y_min
            median_val = np.median(all_vals)
            min_span = max(40.0, 0.05 * median_val)
            if y_span < min_span:
                center = (y_max + y_min) / 2.0
                ax.set_ylim(center - min_span / 2.0, center + min_span / 2.0)
            else:
                pad = y_span * 0.15
                ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(meta_algorithms, fontsize=11.5, fontweight="bold")
            ax.set_xlabel(code, fontsize=13.5, fontweight="bold", labelpad=8)
            ax.set_ylabel("Makespan (s)", fontsize=11.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {group_name} boxplot to: {out_path}")

    # ---------------------------------------------------------------------------
    # 5. Generate Target Table: table_metrics_cmax_sd_cputime.md
    # ---------------------------------------------------------------------------
    cmax_sd_lines = [
        "# Bảng Dữ liệu Đối sánh 9 Instance ($W_1 \\to W_9$): $C_{max}$, SD (s), %SD và CPU Time\n",
        "| Instance | Quy mô | Cấu hình | Đặc trưng luồng | Thuật toán | $C_{max}^{best}$ (s) | $C_{max}^{avg}$ (s) | $C_{max}^{worst}$ (s) | SD (s) | %SD | CPU Time (s) |",
        "| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for sc_info in STANDARD_SCENARIOS:
        code = sc_info["code"]
        first = True
        for algo in all_algorithms:
            arr = results_data[code][algo]
            c_best = np.min(arr)
            c_avg = np.mean(arr)
            c_worst = np.max(arr)
            sd = np.std(arr)
            sd_pct = (sd / max(1e-5, c_avg)) * 100.0
            avg_time = np.mean(times_data[code][algo])

            inst_col = f"**{code}**" if first else ""
            scale_col = sc_info["scale"] if first else ""
            cfg_col = f"{sc_info['nin']} In, {sc_info['nout']} Out" if first else ""
            flow_col = sc_info["flow_type"] if first else ""

            if algo == "SAC-PSO":
                cmax_sd_lines.append(
                    f"| {inst_col} | {scale_col} | {cfg_col} | {flow_col} | **{algo} (Ours)** | **{c_best:.2f}** | **{c_avg:.2f}** | **{c_worst:.2f}** | **{sd:.2f}** | **{sd_pct:.2f}%** | **{avg_time:.3f}** |"
                )
            else:
                cmax_sd_lines.append(
                    f"| {inst_col} | {scale_col} | {cfg_col} | {flow_col} | {algo} | {c_best:.2f} | {c_avg:.2f} | {c_worst:.2f} | {sd:.2f} | {sd_pct:.2f}% | {avg_time:.3f} |"
                )
            first = False

    cmax_sd_lines.append("\n---\n### Ghi chú Giải thích Thống kê:")
    cmax_sd_lines.append(f"1. **$C_{{max}}^{{best}}, C_{{max}}^{{avg}}, C_{{max}}^{{worst}}$**: Thời gian hoàn thành đơn hàng (Makespan) tính bằng **giây (s)** của {num_runs} lần chạy độc lập ({num_runs} random seeds).")
    cmax_sd_lines.append("2. **SD (s)**: Độ lệch chuẩn tuyệt đối tính bằng **giây (s)**.")
    cmax_sd_lines.append("3. **%SD**: Hệ số biến thiên (Relative Standard Deviation / Coefficient of Variation): $\\%SD = \\frac{\\text{SD}}{\\text{Mean}} \\times 100\\%$.")
    cmax_sd_lines.append("4. **CPU Time (s)**: Thời gian tính toán trung bình cho 1 lần giải trên CPU.\n")

    cmax_sd_path = os.path.join(results_dir, "table_metrics_cmax_sd_cputime.md")
    with open(cmax_sd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cmax_sd_lines) + "\n")
    print(f"Saved target Cmax, SD, CPU time table to: {cmax_sd_path}")

    # ---------------------------------------------------------------------------
    # 6. Generate Summary Table: table_standard_scenarios.md
    # ---------------------------------------------------------------------------
    table_lines = [
        "# Bảng Danh mục và Kết quả Đối sánh Thực nghiệm 9 Instance (W1 - W9)\n",
        "## 1. Danh mục 9 Kịch bản Thử nghiệm (Small, Medium, Large)\n",
        "| Instance | Quy mô ($N_{jobs}$) | Nhập ($N_{in}$) | Xuất ($N_{out}$) | Tổng Ops ($N_{ops}$) | Tỉ lệ $\\beta = \\frac{N_{in}}{N_{out}}$ | Đặc trưng luồng kho |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for sc in STANDARD_SCENARIOS:
        beta = sc["nin"] / max(1, sc["nout"])
        table_lines.append(
            f"| **{sc['code']}** | {sc['scale']} | {sc['nin']} | {sc['nout']} | {sc['nops']} | {beta:.2f} | **{sc['desc']}** |"
        )

    table_lines.append(f"\n## 2. Kết quả Đối sánh Hiệu năng Chi tiết qua {num_runs} Runs Độc lập\n")
    table_lines.append("| Instance | Quy mô | Thuật toán | $C_{max}^{best}$ (s) | $C_{max}^{avg}$ (s) | $C_{max}^{worst}$ (s) | SD (s) | SD (%) | Cải thiện FIFO (%) | Cải thiện PSO (%) | Thời gian (s) |")
    table_lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sc_info in STANDARD_SCENARIOS:
        code = sc_info["code"]
        c_fifo_avg = np.mean(results_data[code]["FIFO"])
        c_pso_avg = np.mean(results_data[code]["PSO"])

        first = True
        for algo in all_algorithms:
            arr = results_data[code][algo]
            c_best = np.min(arr)
            c_avg = np.mean(arr)
            c_worst = np.max(arr)
            sd = np.std(arr)
            sd_pct = (sd / max(1e-5, c_avg)) * 100.0
            imp_fifo = (c_fifo_avg - c_avg) / max(1e-5, c_fifo_avg) * 100.0
            imp_pso = (c_pso_avg - c_avg) / max(1e-5, c_pso_avg) * 100.0
            avg_time = np.mean(times_data[code][algo])

            inst_col = f"**{code}**" if first else ""
            grp_col = sc_info["scale"] if first else ""

            if algo == "SAC-PSO":
                table_lines.append(
                    f"| {inst_col} | {grp_col} | **{algo} (Ours)** | **{c_best:.2f}** | **{c_avg:.2f}** | **{c_worst:.2f}** | **{sd:.2f}** | **{sd_pct:.2f}%** | **{imp_fifo:+.2f}%** | **{imp_pso:+.2f}%** | {avg_time:.3f} |"
                )
            else:
                table_lines.append(
                    f"| {inst_col} | {grp_col} | {algo} | {c_best:.2f} | {c_avg:.2f} | {c_worst:.2f} | {sd:.2f} | {sd_pct:.2f}% | {imp_fifo:+.2f}% | {imp_pso:+.2f}% | {avg_time:.3f} |"
                )
            first = False

    table_path = os.path.join(results_dir, "table_standard_scenarios.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Saved summary Markdown table to: {table_path}")

    return results_data, times_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per scenario")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--iter", type=int, default=50, help="Max iterations")
    parser.add_argument("--plot-only", action="store_true", help="Only replot from benchmark_results.json")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if args.plot_only:
        json_path = os.path.join(results_dir, "benchmark_results.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generate_outputs(data["results"], data["times"], results_dir)
    else:
        run_benchmark(num_runs=args.runs, pop_size=args.pop, max_iter=args.iter)
