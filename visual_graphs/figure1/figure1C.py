"""
Figure 1C — Input Feature Distributions by Class (Box Plots)
Data source: firedata_validation_with_preds.csv
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BASE = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project"
OUT  = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure1"

BLUE = "#2563EB"
RED  = "#DC2626"


def plot_feature_distributions(save_path: str = f"{OUT}/figure_1c_feature_distributions.png"):
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

    cols            = ["Temp", "Audio", "Humidity", "CO2"]
    feature_labels  = ["Temp\n(norm.)", "Audio\n(norm.)", "Humidity\n(norm.)", "CO₂\n(norm.)"]
    fire_vals       = [df[df["Fire"] == 1][c].values for c in cols]
    nofire_vals     = [df[df["Fire"] == 0][c].values for c in cols]
    positions_fire  = [1, 4, 7, 10]
    positions_nofire= [2, 5, 8, 11]

    fig, ax = plt.subplots(figsize=(8, 5))

    box_kwargs = dict(showfliers=False, patch_artist=True,
                      medianprops=dict(color="white", lw=2))

    ax.boxplot(fire_vals, positions=positions_fire, widths=0.75,
               boxprops=dict(facecolor=RED,  alpha=0.72),
               whiskerprops=dict(color=RED), capprops=dict(color=RED),
               **box_kwargs)

    ax.boxplot(nofire_vals, positions=positions_nofire, widths=0.75,
               boxprops=dict(facecolor=BLUE, alpha=0.72),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               **box_kwargs)

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(feature_labels, fontsize=9.5)
    ax.set_ylabel("Normalized Value")
    ax.set_title(
        "Input Feature Distributions by Class\n(Validation Set, n = 30,000)",
        fontweight="bold", pad=10,
    )

    p1 = mpatches.Patch(color=RED,  alpha=0.72, label="Fire")
    p2 = mpatches.Patch(color=BLUE, alpha=0.72, label="No Fire")
    ax.legend(handles=[p1, p2], loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_feature_distributions()
