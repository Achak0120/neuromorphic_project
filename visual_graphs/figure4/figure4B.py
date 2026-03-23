"""
Figure 4B — Empirical CDF of Detection Times: Baseline vs. Best Configuration
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


def plot_detection_time_cdf(save_path: str = f"{OUT}/figure_4b_detection_time_cdf.png"):
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

    fig, ax = plt.subplots(figsize=(7, 4.8))

    for times, color, label in [
        (base_times,  BLUE, f"Baseline  (n={len(base_times)})"),
        (patch_times, RED,  f"Best Config  (n={len(patch_times)})"),
    ]:
        xs = np.sort(times)
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax.step(xs, ys, color=color, lw=2.2, label=label, where="post")

    ax.axhline(50, color=GRAY, ls=":", lw=1.1, alpha=0.7)
    ax.axhline(90, color=GRAY, ls=":", lw=1.1, alpha=0.7)
    ax.text(0.5, 51.5, "50th percentile", color=GRAY, fontsize=8.2)
    ax.text(0.5, 91.5, "90th percentile", color=GRAY, fontsize=8.2)

    ax.set_xlabel("Detection Time (minutes)")
    ax.set_ylabel("Cumulative % of Detected Trials")
    ax.set_ylim(0, 106)
    ax.set_title(
        "Empirical CDF of Detection Times\nBaseline vs. Patch-Radius Split Configuration",
        fontweight="bold", pad=10,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_detection_time_cdf()
