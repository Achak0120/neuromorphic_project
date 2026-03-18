"""
Figure 5B — Feature-Space Scatter: CO₂ vs. Audio by Class
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


def plot_co2_vs_audio(save_path: str = f"{OUT}/figure_5b_co2_vs_audio.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9.5,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    df = pd.read_csv(f"{BASE}/firedata_validation_with_preds.csv")

    np.random.seed(0)
    idx = np.random.choice(len(df), 3000, replace=False)
    ds  = df.iloc[idx]

    fire   = ds[ds["Fire"] == 1]
    nofire = ds[ds["Fire"] == 0]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    ax.scatter(nofire["CO2"],  nofire["Audio"],
               c=BLUE, alpha=0.20, s=12, edgecolors="none", label="No Fire")
    ax.scatter(fire["CO2"],    fire["Audio"],
               c=RED,  alpha=0.20, s=12, edgecolors="none", label="Fire")

    p1 = mpatches.Patch(color=RED,  alpha=0.75, label="Fire")
    p2 = mpatches.Patch(color=BLUE, alpha=0.75, label="No Fire")
    ax.legend(handles=[p1, p2], loc="upper left")

    ax.set_xlabel("CO₂ (normalized)")
    ax.set_ylabel("Audio Signal (normalized)")
    ax.set_title(
        "Feature Space: CO₂ vs. Audio Signal\n"
        "(Validation set, 3,000-sample subset, colored by class)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_co2_vs_audio()
