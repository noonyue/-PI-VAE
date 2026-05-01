"""
Plot L2 manufacturer performance overview:
- (A) Direct vs Cascade overall accuracy bar chart
- (B) Per-drug cascade accuracy bar chart

Inputs:
- results/model_comparison_l2_overview.csv
- results/model_comparison_l2_cascade_per_drug.csv

Output:
- figures/l2_overview_direct_vs_cascade.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    os.makedirs("figures", exist_ok=True)

    overview = pd.read_csv("results/model_comparison_l2_overview.csv")
    per_drug = pd.read_csv("results/model_comparison_l2_cascade_per_drug.csv")

    # Figure layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (A) Overall direct vs cascade
    ax = axes[0]
    # Choose a subset of methods to plot (Direct-PLSDA Raw/Latent, Direct-SVM Raw, Direct-RF Raw, Cascade)
    plot_rows = overview.copy()
    # Build readable labels
    labels = []
    for _, row in plot_rows.iterrows():
        if row["Strategy"] == "Direct":
            lbl = f"{row['Model']} ({row['Feature']})"
        else:
            lbl = f"Cascade RF ({row['Feature']})"  # Now shows "Fused" instead of "Latent"
        labels.append(lbl)
    plot_rows["Label"] = labels

    sns.barplot(
        data=plot_rows,
        x="Accuracy",
        y="Label",
        hue="Strategy",
        palette={"Direct": "#4C72B0", "Cascade": "#DD8452"},
        ax=ax,
        dodge=False,
    )
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("")
    ax.set_title("(A) L2: Direct 28-class vs Cascade")
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f"{width*100:.1f}%",
            va="center",
            fontsize=8,
        )
    ax.legend(title="Strategy", fontsize=8)

    # (B) Per-drug cascade accuracy
    ax2 = axes[1]
    per_drug_sorted = per_drug.sort_values("Accuracy", ascending=False)
    sns.barplot(
        data=per_drug_sorted,
        x="Drug",
        y="Accuracy",
        color="#55A868",
        ax=ax2,
    )
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Drug (code)")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("(B) L2 Cascade per-drug accuracy (RF + Fused)")
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(
            p.get_x() + p.get_width() / 2,
            height + 0.01,
            f"{height*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = os.path.join("figures", "l2_overview_direct_vs_cascade.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

