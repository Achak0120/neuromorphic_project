"""
Figure 2A — Detection Rate Across Simulation Configurations (Bar Chart)
Data sources: batch_results_50_trials.csv, batch_results_valid_reach_patch.csv,
              batch_results_valid_adaptive.csv, batch_results_valid_adaptive_PATCHRSPLIT.csv
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/home/claude/neuromorphic_project/neuromorphic_project-main/one_node_sim"
OUT  = "/mnt/user-data/outputs"

COLORS = ["#2563EB", "#16A34A", "#EA580C", "#DC2626"]


def plot_detection_rates(save_path: str = f"{OUT}/figure_2a_detection_rates.png"):
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

    detected = [int(df["detected"].sum()) for df in dfs]
    totals   = [len(df) for df in dfs]
    rates    = [d / t * 100 for d, t in zip(detected, totals)]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    bars = ax.bar(labels, rates, color=COLORS, alpha=0.84,
                  edgecolor="white", linewidth=1.3, width=0.55)

    for bar, rate, nd, tot in zip(bars, rates, detected, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{rate:.0f}%  ({nd}/{tot})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.axhline(86, color=COLORS[3], ls="--", lw=1.3, alpha=0.55)
    ax.text(3.42, 87.5, "Best: 86%", color=COLORS[3], fontsize=8.5)

    ax.set_ylim(0, 108)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title(
        "Detection Rate by Simulation Configuration\n(n = 50 trials each)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_detection_rates()
