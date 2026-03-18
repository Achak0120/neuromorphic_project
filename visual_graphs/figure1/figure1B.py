"""
Figure 1B — Output Spike-Count Histograms by Class (Fire Neuron)
Data source: firedata_validation_with_preds.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BASE = "/home/claude/neuromorphic_project/neuromorphic_project-main"
OUT  = "/mnt/user-data/outputs"

BLUE = "#2563EB"
RED  = "#DC2626"
GRAY = "#6B7280"


def plot_spike_histograms(save_path: str = f"{OUT}/figure_1b_spike_histograms.png"):
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

    df = pd.read_csv(f"{BASE}/firedata_validation_with_preds.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    bins = np.arange(0, 18, 1)

    ax.hist(
        df[df["Fire"] == 1]["Spikes_Class1"],
        bins=bins, alpha=0.72, color=RED,
        label="True Fire", density=True, edgecolor="white",
    )
    ax.hist(
        df[df["Fire"] == 0]["Spikes_Class1"],
        bins=bins, alpha=0.72, color=BLUE,
        label="True No-Fire", density=True, edgecolor="white",
    )

    ax.axvline(
        0.5, color=GRAY, lw=1.4, ls="--", alpha=0.8, label="Decision boundary (≥1 spike → Fire)"
    )

    # Annotate means
    mean_fire   = df[df["Fire"] == 1]["Spikes_Class1"].mean()
    mean_nofire = df[df["Fire"] == 0]["Spikes_Class1"].mean()
    ax.axvline(mean_fire,   color=RED,  lw=1.2, ls=":", alpha=0.7)
    ax.axvline(mean_nofire, color=BLUE, lw=1.2, ls=":", alpha=0.7)
    ax.text(mean_fire   + 0.2, ax.get_ylim()[1] * 0.88,
            f"μ={mean_fire:.1f}", color=RED,  fontsize=8.5)
    ax.text(mean_nofire + 0.2, ax.get_ylim()[1] * 0.78,
            f"μ={mean_nofire:.1f}", color=BLUE, fontsize=8.5)

    ax.set_xlabel("Fire-Neuron Output Spike Count (over 35 timesteps)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Fire-Neuron Spike Counts by Ground-Truth Class\n(Validation Set, n = 30,000)",
        fontweight="bold", pad=10,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_spike_histograms()
