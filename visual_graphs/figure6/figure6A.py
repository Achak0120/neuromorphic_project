"""
Figure 6 — End-to-End Pipeline Architecture Diagram
Train → Simulate → Deploy
All metrics embedded in the diagram are sourced directly from project results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


OUT = "c:\\Users\\achak\\OneDrive\\Desktop\\neuromorphic\\neuromorphic_project\\visual_graphs\\figure6"

BLUE   = "#2563EB"
PURPLE = "#7C3AED"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
RED    = "#DC2626"
GRAY   = "#6B7280"


def plot_pipeline_architecture(save_path: str = f"{OUT}/figure_6_pipeline_architecture.png"):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    15,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.2, 14)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.set_title(
        "End-to-End Neuromorphic Pipeline:  Train → Simulate → Deploy",
        fontweight="bold", pad=12, fontsize=13,
    )
    dx = 1
    stages = [
        (1.0 + dx,  "① Data\nGeneration",
         "100 K synthetic samples\nTemp · Audio\nHumidity · CO₂\n(Gaussian noise added)",
         BLUE),
        (3.5 + dx,  "② SNN\nTraining",
         "snnTorch / PyTorch\nLIF neurons  β = 0.7\nSurrogate gradient\n4 → 500 → 500 → 2",
         PURPLE),
        (6.0 + dx,  "③ Offline\nValidation",
         "30 K held-out samples\nAccuracy = 93.77%\nFPR ≈ 0.00%\nFNR = 12.56%",
         GREEN),
        (8.5 + dx,  "④ Fire-Spread\nSimulation",
         "Grid-based spread model\nAdaptive thresholding\nDetection rate = 86%\nMean latency = 7.35 min",
         ORANGE),
        (11.0 + dx, "⑤ ESP32-S3\nDeployment",
         "Float32 weights · 1.02 MB\nAccuracy = 99.0%\n2.44 s / sample\n35-step LIF forward pass",
         RED),
    ]

    BOX_W, BOX_H = 2.0, 2.9

    for x, title, body, color in stages:
        rect = mpatches.FancyBboxPatch(
            (x - BOX_W / 2, 0.65), BOX_W, BOX_H,
            boxstyle="round,pad=0.10",
            linewidth=2.0, edgecolor=color,
            facecolor=color + "1A",   # ~10% alpha fill
        )
        ax.add_patch(rect)
        ax.text(x, 0.65 + BOX_H - 0.25, title,
                ha="center", va="top",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x, 0.65 + BOX_H / 2 - 0.15, body,
                ha="center", va="center",
                fontsize=8.0, color="#111827", linespacing=1.6)

    # Arrows
    arrow_kw = dict(
        arrowstyle="-|>", color=GRAY, lw=1.8,
        mutation_scale=15,
    )
    xs = [s[0] for s in stages]
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - BOX_W / 2 - 0.06, 0.65 + BOX_H / 2),
            xytext=(xs[i] + BOX_W / 2 + 0.06, 0.65 + BOX_H / 2),
            arrowprops=arrow_kw,
        )

    # Phase bracket labels at bottom
    for x, lbl, color in [
        (1.0,  "Generate",  BLUE),
        (3.5,  "Train",     PURPLE),
        (6.0,  "Evaluate",  GREEN),
        (8.5,  "Simulate",  ORANGE),
        (11.0, "Deploy",    RED),
    ]:
        ax.text(x, 0.30, lbl,
                ha="center", va="bottom",
                fontsize=9, color=color,
                style="italic", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_pipeline_architecture()
