"""
Figure 3: L1 Drug Classification Results
Combined: Confusion matrix (left) + Model accuracy/F1 bar chart (right)

Layout: 1 row x 2 panels
- (a) Normalized confusion matrix with drug name labels (wider)
- (b) Three-model Accuracy & Macro-F1 grouped bar (Raw vs Latent)

Output: figures/figure3_l1_classification.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label, format_axes

setup_style()

# ── Data ─────────────────────────────────────────────────────────────────────
DRUG_NAMES = ['CIM', 'FMD', 'GLD', 'GSR', 'HCT', 'IBU', 'MHE', 'MHL', 'MHR']
N = len(DRUG_NAMES)

# Confusion matrix — SVM Raw, 100% overall accuracy (from model_comparison_l1.csv)
# All 72 test samples correctly classified, perfect diagonal
CM = np.array([
    [10,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  6,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  4,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  4,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  6,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  8,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  7,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0, 11,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, 16],
])

MODELS        = ['PLS-DA', 'SVM', 'RandomForest']
MODEL_LABELS  = ['PLS-DA', 'SVM', 'RF']
COLOR_RAW     = '#FF7F0E'
COLOR_LATENT  = '#5BA3D9'
BAR_W         = 0.28


def draw_confusion_matrix(ax):
    cm_norm = CM.astype(float) / CM.sum(axis=1, keepdims=True)
    per_acc = CM.diagonal() / CM.sum(axis=1)
    overall = CM.diagonal().sum() / CM.sum()

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label('Normalized Proportion', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Tick labels
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(DRUG_NAMES, fontsize=9, fontweight='bold')
    ax.set_yticklabels(DRUG_NAMES, fontsize=9, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    # Cell annotations
    thresh = 0.5
    for i in range(N):
        for j in range(N):
            count = CM[i, j]
            prop  = cm_norm[i, j]
            if count > 0:
                color = 'white' if prop > thresh else 'black'
                if i == j:
                    txt = f'{count}\n({prop*100:.0f}%)'
                    ax.text(j, i, txt, ha='center', va='center',
                            color=color, fontsize=8.5, fontweight='bold')
                else:
                    ax.text(j, i, str(count), ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')

    # Grid lines
    ax.set_xticks(np.arange(N + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.8)
    ax.tick_params(which='minor', size=0)

    # Red border on misclassified cells
    for i in range(N):
        for j in range(N):
            if i != j and CM[i, j] > 0:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='red',
                                     linewidth=2.2, zorder=5)
                ax.add_patch(rect)

    ax.set_xlabel('Predicted Drug Type', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Drug Type',      fontsize=10, fontweight='bold')
    ax.set_title(
        f'(a)  L1 Confusion Matrix  (SVM, Raw)\n'
        f'Overall Accuracy = {overall*100:.2f}%,  n = {CM.sum()} test samples',
        fontsize=10, fontweight='bold', pad=10, loc='left')

    # Stats box
    stats = (f'Correct: {CM.diagonal().sum()}/{CM.sum()}\n'
             f'Misclassified: {CM.sum() - CM.diagonal().sum()}\n'
             f'Perfect classes: {(per_acc == 1.0).sum()}/9')
    ax.text(0.02, 0.02, stats, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    add_panel_label(ax, '(a)', x_offset=-0.10, y_offset=1.04)


def draw_bar_comparison(ax):
    df = pd.read_csv('results/model_comparison_l1.csv')

    def get(model, feature, col):
        row = df[(df['Model'] == model) & (df['Feature'] == feature)]
        return row[col].values[0] * 100

    x = np.arange(len(MODELS))

    # Two metric groups stacked vertically using twin-x trick:
    # Draw Accuracy bars with solid fill, F1 with hatch
    acc_raw    = [get(m, 'Raw',    'Accuracy') for m in MODELS]
    acc_latent = [get(m, 'Latent', 'Accuracy') for m in MODELS]
    f1_raw     = [get(m, 'Raw',    'Macro_F1') for m in MODELS]
    f1_latent  = [get(m, 'Latent', 'Macro_F1') for m in MODELS]

    # Group: [Acc_Raw, Acc_Latent, F1_Raw, F1_Latent] per model, 4 bars
    x_offsets = [-1.5*BAR_W, -0.5*BAR_W, 0.5*BAR_W, 1.5*BAR_W]
    colors    = [COLOR_RAW, COLOR_LATENT, COLOR_RAW, COLOR_LATENT]
    hatches   = ['', '', '///', '///']
    datasets  = [acc_raw, acc_latent, f1_raw, f1_latent]
    labels    = ['Accuracy – Raw', 'Accuracy – Latent',
                 'Macro F1 – Raw', 'Macro F1 – Latent']

    bars_list = []
    for i, (vals, xoff, col, hatch) in enumerate(
            zip(datasets, x_offsets, colors, hatches)):
        b = ax.bar(x + xoff, vals, BAR_W,
                   color=col, alpha=0.85, edgecolor='black',
                   linewidth=0.7, hatch=hatch, label=labels[i])
        bars_list.append(b)
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                    f'{h:.1f}', ha='center', va='bottom',
                    fontsize=7.0, fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylim(0, 120)
    ax.axhline(100, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title('(b)  L1: All Models — Accuracy & Macro F1\n'
                 'Raw (SNV) vs Latent (PI-VAE encoded)',
                 fontsize=10, fontweight='bold', pad=10, loc='left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(length=3, width=0.8)

    # Legend
    patch_acc_raw    = mpatches.Patch(color=COLOR_RAW,    alpha=0.85, label='Accuracy — Raw')
    patch_acc_lat    = mpatches.Patch(color=COLOR_LATENT, alpha=0.85, label='Accuracy — Latent')
    patch_f1_raw     = mpatches.Patch(color=COLOR_RAW,    alpha=0.85, hatch='///', label='Macro F1 — Raw')
    patch_f1_lat     = mpatches.Patch(color=COLOR_LATENT, alpha=0.85, hatch='///', label='Macro F1 — Latent')
    ax.legend(handles=[patch_acc_raw, patch_acc_lat, patch_f1_raw, patch_f1_lat],
              fontsize=8, loc='lower right', ncol=2, framealpha=0.9)

    add_panel_label(ax, '(b)', x_offset=-0.10, y_offset=1.04)


def main():
    os.makedirs('figures', exist_ok=True)

    fig = plt.figure(figsize=(16, 7))
    fig.text(0.50, 0.995,
             'Figure 3.  L1 Drug Classification: Confusion Matrix & Model Comparison',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.88, bottom=0.10,
                           left=0.06, right=0.97,
                           wspace=0.38,
                           width_ratios=[1.1, 1.0])

    ax_cm  = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    draw_confusion_matrix(ax_cm)
    draw_bar_comparison(ax_bar)

    out_path = 'figures/figure3_l1_classification.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
