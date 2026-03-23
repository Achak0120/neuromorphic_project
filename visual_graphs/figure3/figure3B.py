"""
Figure 3B — Embedded Classification Metrics Bar Chart (ESP32-S3)
Values sourced from paper Section 3.3:
  accuracy=99.0%, precision=0.994, recall=0.986, FPR=0.006
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure3"

COLORS = ["#16A34A", "#2563EB", "#EA580C", "#7C3AED"]


def plot_embedded_metrics(save_path: str = f"{OUT}/figure_3b_embedded_metrics.png"):
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

    metrics = ["Accuracy", "Precision", "Recall", "Specificity\n(1 − FPR)"]
    values  = [0.990,       0.994,        0.986,    1 - 0.006]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    bars = ax.bar(metrics, values, color=COLORS, alpha=0.84,
                  edgecolor="white", linewidth=1.3, width=0.55)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.0008,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        )

    ax.set_ylim(0.970, 1.000)
    ax.set_ylabel("Score")
    ax.set_title(
        "Classification Metrics — ESP32-S3 Embedded Inference\n(Replay dataset, n = 1,024 samples)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_embedded_metrics()
