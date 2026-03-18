"""
Figure 2B — Detection Latency Across Simulation Configurations
         (Mean ± Std bars with jittered raw trial scatter)
Data sources: batch_results_50_trials.csv, batch_results_valid_reach_patch.csv,
              batch_results_valid_adaptive.csv, batch_results_valid_adaptive_PATCHRSPLIT.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/home/claude/neuromorphic_project/neuromorphic_project-main/one_node_sim"
OUT  = "/mnt/user-data/outputs"

COLORS = ["#2563EB", "#16A34A", "#EA580C", "#DC2626"]
GRAY   = "#6B7280"


def plot_detection_latency(save_path: str = f"{OUT}/figure_2b_detection_latency.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   9.5,
        "ytick.labelsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    dfs = [
        pd.read_csv(f"{BASE}/batch_results_50_trials.csv"),
        pd.read_csv(f"{BASE}/batch_results_valid_reach_patch.csv"),
        pd.read_csv(f"{BASE}/batch_results_valid_adaptive.csv"),
        pd.read_csv(f"{BASE}/batch_results_valid_adaptive_PATCHRSPLIT.csv"),
    ]
    labels = [
        "Baseline\n(All 50 trials)",
        "Valid-Patch\nFilter",
        "Adaptive\nThreshold",
        "Patch-Radius\nSplit (Best)",
    ]

    times_list = [df[df["detected"] == True]["detect_minutes"].dropna().values
                  for df in dfs]
    means = [t.mean() for t in times_list]
    stds  = [t.std()  for t in times_list]

    x_pos = np.arange(len(dfs))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.bar(x_pos, means, color=COLORS, alpha=0.72,
           edgecolor="white", linewidth=1.3, width=0.55, zorder=2)

    ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                color=GRAY, capsize=5, linewidth=1.8, zorder=3)

    np.random.seed(42)
    for i, times in enumerate(times_list):
        jitter = np.random.uniform(-0.2, 0.2, len(times))
        ax.scatter(i + jitter, times, alpha=0.38, s=22,
                   color=COLORS[i], edgecolors="none", zorder=4)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f"{m:.1f} min",
                ha="center", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detection Time (minutes)")
    ax.set_title(
        "Detection Latency by Configuration\n"
        "(bars = mean ± std;  dots = individual detected trials)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_detection_latency()
