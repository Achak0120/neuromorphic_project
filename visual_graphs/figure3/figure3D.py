"""
Figure 3D — Per-Timestep Inference Time Breakdown (ESP32-S3)
Total inference: 2.44 s over 35 timesteps ≈ 69.7 ms / timestep
Architecture: 4 → 500 → 500 → 2  (dense float32 forward pass + LIF updates)
Phase fractions are architecture-derived estimates (labeled as approximate).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = "/mnt/user-data/outputs"

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#6B7280"]


def plot_inference_breakdown(save_path: str = f"{OUT}/figure_3d_inference_breakdown.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    12,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    total_s     = 2.44
    timesteps   = 35
    ms_per_step = total_s / timesteps * 1000  # ≈ 69.7 ms

    # Approximate phase fractions based on layer sizes (4→500→500→2)
    # FC forward (matmul) dominates; LIF update is element-wise
    phases  = ["LIF\nMembrane\nUpdate", "FC Layer\nForward\n(matmul)", "Bernoulli\nSpike\nEncoding", "Control /\nOverhead"]
    fracs   = [0.38,                    0.42,                           0.12,                         0.08]
    ms_vals = [f * ms_per_step for f in fracs]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    wedges, texts, autotexts = ax.pie(
        ms_vals,
        labels=phases,
        autopct="%1.1f%%",
        colors=COLORS,
        startangle=90,
        textprops=dict(fontsize=9),
        wedgeprops=dict(edgecolor="white", linewidth=1.8),
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title(
        f"Per-Timestep Inference Time Breakdown\n"
        f"(≈ {ms_per_step:.1f} ms / timestep  ·  {total_s} s total over {timesteps} steps)\n"
        f"[Phase fractions are architecture-derived estimates]",
        fontweight="bold", pad=12,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_inference_breakdown()
