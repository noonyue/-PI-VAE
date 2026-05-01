"""
Compose a 2x2 big figure for L1/L2 performance:
- Top-left: L1 drug confusion matrix
- Top-right: L2 manufacturer (28-class) confusion matrix
- Bottom-left: L2 model performance heatmap (per drug)
- Bottom-right: L2 decision boundary (e.g., drug=MHR)
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(os.path.join(fig_dir, "composite"), exist_ok=True)

    # Source figure paths
    f_l1 = os.path.join(fig_dir, "fig4_l1_confusion.png")
    if not os.path.exists(f_l1):
        # fallback to older name
        f_l1 = os.path.join(fig_dir, "4-l1_confusion_matrix.png")

    f_l2_conf = os.path.join(fig_dir, "fig_l2_manufacturer_confusion.png")
    f_l2_heat = os.path.join(fig_dir, "6-l2_model_performance_heatmap.png")
    f_l2_dec = os.path.join(fig_dir, "5-l2_decision_boundary_zoom.png")

    imgs = [mpimg.imread(p) for p in [f_l1, f_l2_conf, f_l2_heat, f_l2_dec]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panel_labels = ["(A)", "(B)", "(C)", "(D)"]

    for ax, img, label in zip(axes.flat, imgs, panel_labels):
        ax.imshow(img)
        ax.set_axis_off()
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    plt.tight_layout()
    out_path = os.path.join(fig_dir, "composite", "fig_l1_l2_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved big figure: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

