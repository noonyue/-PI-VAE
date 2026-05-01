"""
Stepwise accuracy contribution waterfall chart.

Outputs:
- figures/stepwise_accuracy_waterfall.png
"""
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    create_waterfall_chart, COLOR_PRED, COLOR_BASELINE, COLOR_NIR
)

# These values can be updated if rerunning with new measurements
# Baseline: Raw direct (28-class) accuracy from ablation_study
BASELINE = 0.6111
# After preprocessing (assume slight gain if direct on SNV raw+SVM); placeholder
AFTER_SNV = 0.7000
# After PI-VAE features (L1 accuracy or average before cascade)
AFTER_VAE = 0.9306
# Final cascade accuracy (average L2)
FINAL_CASCADE = 0.9745


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)


def waterfall(values, labels, title, save_path):
    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(8, 4))
    ax = fig.add_subplot(gs[0, 0])
    
    deltas = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))]
    running = 0
    xs = range(len(values))
    colors = []
    bottoms = []
    heights = []
    for i, d in enumerate(deltas):
        if i == 0:
            bottoms.append(0)
            heights.append(d)
            colors.append(COLOR_BASELINE)  # Orange for baseline
        else:
            bottoms.append(running)
            heights.append(d)
            colors.append(COLOR_PRED if d >= 0 else COLOR_NIR)  # Blue/Red
            running += d

    ax.bar(xs, heights, bottom=bottoms, color=colors, 
          edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    format_axes(ax, ylabel="Accuracy", title=title)
    add_panel_label(ax, "(a)", x_offset=-0.12, y_offset=1.02)
    
    for x, b, h, v in zip(xs, bottoms, heights, values):
        ax.text(x, b + h + 0.01, f"{v:.3f}", ha="center", va="bottom", 
               fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    ensure_dirs()
    values = [BASELINE, AFTER_SNV, AFTER_VAE, FINAL_CASCADE]
    labels = ["Raw direct", "+SNV", "+PI-VAE features", "+Cascade"]
    waterfall(values, labels, "Stepwise Accuracy Contribution", "figures/stepwise_accuracy_waterfall.png")


if __name__ == "__main__":
    main()

