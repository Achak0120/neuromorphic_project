"""
Figure 4A — Detection Time Histogram: Baseline vs. Best Configuration
Data sources: batch_results_50_trials.csv, batch_results_valid_adaptive_PATCHRSPLIT.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\one_node_sim"
OUT  = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure4"

BLUE = "#2563EB"
RED  = "#DC2626"
GRAY = "#6B7280"


def plot_detection_time_histogram(save_path: str = f"{OUT}/figure_4a_detection_time_histogram.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    df_base  = pd.read_csv(f"{BASE}/batch_results_50_trials.csv")
    df_patch = pd.read_csv(f"{BASE}/batch_results_valid_adaptive_PATCHRSPLIT.csv")

    base_times  = df_base[df_base["detected"]  == True]["detect_minutes"].values
    patch_times = df_patch[df_patch["detected"] == True]["detect_minutes"].values

    bin_edges = np.arange(0, 23, 2)

    fig, ax = plt.subplots(figsize=(7, 4.8))

    ax.hist(base_times,  bins=bin_edges, alpha=0.65, color=BLUE,
            label=f"Baseline  (n={len(base_times)} detected / 50)",
            edgecolor="white")
    ax.hist(patch_times, bins=bin_edges, alpha=0.65, color=RED,
            label=f"Best Config  (n={len(patch_times)} detected / 50)",
            edgecolor="white")

    mu_base  = np.mean(base_times)
    mu_patch = np.mean(patch_times)
    ax.axvline(mu_base,  color=BLUE, ls="--", lw=1.6,
               label=f"Baseline μ = {mu_base:.1f} min")
    ax.axvline(mu_patch, color=RED,  ls="--", lw=1.6,
               label=f"Best Config μ = {mu_patch:.1f} min")

    ax.set_xlabel("Detection Time (minutes)")
    ax.set_ylabel("Number of Trials")
    ax.set_title(
        "Detection Time Histogram\nBaseline vs. Patch-Radius Split Configuration",
        fontweight="bold", pad=10,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_detection_time_histogram()
