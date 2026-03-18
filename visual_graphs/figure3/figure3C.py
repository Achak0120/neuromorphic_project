"""
Figure 3C — Model Memory Footprint vs. ESP32-S3 Flash Capacity
Values sourced from paper Section 3.3: weight binary ≈ 1.02 MB
ESP32-S3 typical flash: 8 MB
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT    = "/mnt/user-data/outputs"
ORANGE = "#EA580C"
LIGHT  = "#D1D5DB"
GRAY   = "#6B7280"


def plot_memory_footprint(save_path: str = f"{OUT}/figure_3c_memory_footprint.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    model_mb = 1.02
    total_mb = 8.0
    remain   = total_mb - model_mb

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    bars = ax.bar(
        ["SNN Model\nWeights", "Remaining\nFlash"],
        [model_mb, remain],
        color=[ORANGE, LIGHT],
        edgecolor=GRAY, linewidth=1.1, width=0.5,
    )

    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        model_mb + 0.12,
        f"{model_mb} MB\n({model_mb/total_mb*100:.1f}% of flash)",
        ha="center", fontsize=10, fontweight="bold", color=ORANGE,
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        remain + 0.12,
        f"{remain:.2f} MB\navailable",
        ha="center", fontsize=10, color=GRAY,
    )

    ax.set_ylim(0, total_mb * 1.18)
    ax.set_ylabel("Flash Memory (MB)")
    ax.set_title(
        "Model Memory Footprint — ESP32-S3\n(8 MB total flash)",
        fontweight="bold", pad=10,
    )

    # Reference line at total flash
    ax.axhline(total_mb, color=GRAY, ls=":", lw=1.2, alpha=0.6)
    ax.text(1.3, total_mb + 0.1, "Total flash (8 MB)", fontsize=8.5, color=GRAY)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_memory_footprint()
