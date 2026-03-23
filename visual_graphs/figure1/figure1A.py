"""
Figure 1A — Confusion Matrix (Offline SNN Classification)
Data source: firedata_validation_with_preds.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


BASE = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project"
OUT  = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure1"

GRAY  = "#6B7280"
GREEN = "#16A34A"


def plot_confusion_matrix(save_path: str = f"{OUT}/figure_1a_confusion_matrix.png"):
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

    df = pd.read_csv(f"{BASE}/firedata_validation_with_preds.csv")

    tp = int(((df["Fire"] == 1) & (df["Pred_Fire"] == 1)).sum())
    tn = int(((df["Fire"] == 0) & (df["Pred_Fire"] == 0)).sum())
    fp = int(((df["Fire"] == 0) & (df["Pred_Fire"] == 1)).sum())
    fn = int(((df["Fire"] == 1) & (df["Pred_Fire"] == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Fire", "Fire"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight("bold")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr      = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    ax.annotate(
        f"Accuracy = {accuracy:.2%}\nFPR = {fpr:.2%}\nFNR = {fnr:.2%}",
        xy=(0.97, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, lw=0.9),
    )

    ax.set_title(
        "Confusion Matrix — Offline SNN Validation\n(n = 30,000 held-out samples)",
        fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_confusion_matrix()