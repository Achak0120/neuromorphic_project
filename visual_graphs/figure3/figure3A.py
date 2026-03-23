"""
Figure 3A — Embedded Confusion Matrix (ESP32-S3 Replay, n = 1,024)
Values sourced from paper Section 3.3:
  accuracy=99.0%, precision=0.994, recall=0.986, FPR=0.006
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


OUT  = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure3"
GRAY = "#6B7280"


def plot_embedded_confusion_matrix(save_path: str = f"{OUT}/figure_3a_embedded_confusion_matrix.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    # Reconstruct from reported metrics on 1,024 samples (512 per class assumed balanced)
    n_per_class = 512
    recall  = 0.986
    fpr     = 0.006
    tp = int(round(recall * n_per_class))   # 505
    fn = n_per_class - tp                   # 7
    fp = int(round(fpr    * n_per_class))   # 3
    tn = n_per_class - fp                   # 509
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Fire", "Fire"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")

    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight("bold")

    ax.annotate(
        "Accuracy = 99.0%\nPrecision = 0.994\nRecall = 0.986\nFPR = 0.006",
        xy=(0.97, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, lw=0.9),
    )

    ax.set_title(
        "Embedded Confusion Matrix — ESP32-S3\n(Replay dataset, n = 1,024 samples)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_embedded_confusion_matrix()
