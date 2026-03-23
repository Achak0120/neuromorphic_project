"""
Figure 7 — Trial Outcome Breakdown Across All Simulation Configurations
         (Why trials succeed or fail: detected vs. fire_died)
Data sources: batch_results_50_trials.csv, batch_results_valid_reach_patch.csv,
              batch_results_valid_adaptive.csv, batch_results_valid_adaptive_PATCHRSPLIT.csv
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\one_node_sim"
OUT  = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure7"

RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
BLUE   = "#2563EB"
GRAY   = "#6B7280"
LIGHT  = "#D1D5DB"


def plot_trial_outcomes(save_path: str = f"{OUT}/figure_7_trial_outcomes.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    11,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    configs = [
        ("Baseline\n(All 50 trials)",       f"{BASE}/batch_results_50_trials.csv",                    BLUE),
        ("Valid-Patch\nFilter",              f"{BASE}/batch_results_valid_reach_patch.csv",             GREEN),
        ("Adaptive\nThreshold",             f"{BASE}/batch_results_valid_adaptive.csv",                ORANGE),
        ("Patch-Radius\nSplit (Best)",       f"{BASE}/batch_results_valid_adaptive_PATCHRSPLIT.csv",    RED),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle(
        "Trial Outcome Breakdown Across Simulation Configurations  (n = 50 each)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, (title, path, accent_color) in zip(axes, configs):
        df = pd.read_csv(path)

        if "end_reason" in df.columns:
            counts = df["end_reason"].value_counts()
            slice_labels = [r.replace("_", "\n") for r in counts.index]
        else:
            n_det = int(df["detected"].sum())
            n_nd  = len(df) - n_det
            counts = {"detected": n_det, "fire\ndied": n_nd}
            slice_labels = list(counts.keys())
            counts = list(counts.values())

        wedge_colors = [accent_color if "detected" in str(lbl).lower() else LIGHT
                        for lbl in slice_labels]

        wedges, texts, autotexts = ax.pie(
            counts if isinstance(counts, list) else counts.values,
            labels=slice_labels,
            autopct="%1.0f%%",
            colors=wedge_colors,
            startangle=90,
            textprops=dict(fontsize=9),
            wedgeprops=dict(edgecolor="white", linewidth=1.8),
            pctdistance=0.68,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")

        n_detected = int(df["detected"].sum()) if "detected" in df.columns else 0
        ax.set_title(
            f"{title}\n({n_detected}/50 detected)",
            fontweight="bold", pad=6,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_trial_outcomes()
