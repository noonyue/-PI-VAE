"""
Figure 2: L1 Drug Classification & L2 Manufacturer Identification
Three models x Raw/Latent comparison bar charts

Layout: 1 row x 4 panels
- (a) L1 Accuracy    (b) L1 Macro-F1
- (c) L2 Accuracy    (d) L2 Macro-F1

Output: figures/figure2_model_comparison.png
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

MODELS   = ['PLS-DA', 'SVM', 'RandomForest']
MODEL_LABELS = ['PLS-DA', 'SVM', 'RF']

# Raw = solid orange, Latent = light blue
COLOR_RAW    = '#FF7F0E'
COLOR_LATENT = '#5BA3D9'

BAR_W  = 0.32
ALPHA  = 0.88


def bar_group(ax, metric, l1_df, l2_df, panel_a, panel_b):
    """Draw two side-by-side grouped bars (L1 left, L2 right) for one metric."""
    x = np.arange(len(MODELS))

    l1_raw    = [l1_df.loc[(l1_df.Model == m) & (l1_df.Feature == 'Raw'),    metric].values[0] * 100
                 for m in MODELS]
    l1_latent = [l1_df.loc[(l1_df.Model == m) & (l1_df.Feature == 'Latent'), metric].values[0] * 100
                 for m in MODELS]
    l2_raw    = [l2_df.loc[(l2_df.Model == m) & (l2_df.Feature == 'Raw'),    metric].values[0] * 100
                 for m in MODELS]
    l2_latent = [l2_df.loc[(l2_df.Model == m) & (l2_df.Feature == 'Latent'), metric].values[0] * 100
                 for m in MODELS]

    return (l1_raw, l1_latent, l2_raw, l2_latent)


def draw_bars(ax, raw_vals, latent_vals, panel_label, title,
              show_ylabel=True, ylim=(0, 108)):
    x = np.arange(len(MODELS))
    b1 = ax.bar(x - BAR_W / 2, raw_vals,    BAR_W, label='Raw',
                color=COLOR_RAW,    alpha=ALPHA, edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x + BAR_W / 2, latent_vals, BAR_W, label='Latent (VAE)',
                color=COLOR_LATENT, alpha=ALPHA, edgecolor='black', linewidth=0.8)

    # Value annotations
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if show_ylabel:
        ax.set_ylabel('Score (%)', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(length=3, width=0.8)
    add_panel_label(ax, panel_label, x_offset=-0.14, y_offset=1.04)

    # Reference line at 100
    ax.axhline(100, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)


def main():
    os.makedirs('figures', exist_ok=True)

    l1_df = pd.read_csv('results/model_comparison_l1.csv')
    l2_df = pd.read_csv('results/model_comparison_l2_direct_classic.csv')

    # Rename RF column for consistency
    l1_df['Model'] = l1_df['Model'].replace('RandomForest', 'RandomForest')
    l2_df['Model'] = l2_df['Model'].replace('RandomForest', 'RandomForest')

    # Extract values
    def get(df, model, feature, col):
        row = df[(df['Model'] == model) & (df['Feature'] == feature)]
        return row[col].values[0] * 100

    metrics = [('Accuracy', 'Accuracy (%)'), ('Macro_F1', 'Macro F1 (%)')]

    fig = plt.figure(figsize=(13, 11))
    fig.text(0.50, 0.995,
             'Figure 2.  Model Comparison: L1 Drug Classification & L2 Manufacturer Identification',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.92, bottom=0.10,
                           left=0.08, right=0.97,
                           wspace=0.32, hspace=0.42)

    panel_defs = [
        # (row, col, task, metric_col, panel_label, title)
        (0, 0, 'L1', 'Accuracy', '(a)', 'L1: Drug Classification — Accuracy'),
        (0, 1, 'L1', 'Macro_F1', '(b)', 'L1: Drug Classification — Macro F1'),
        (1, 0, 'L2', 'Accuracy', '(c)', 'L2: Manufacturer ID — Accuracy'),
        (1, 1, 'L2', 'Macro_F1', '(d)', 'L2: Manufacturer ID — Macro F1'),
    ]

    axes = []
    for row, col, task, metric, plabel, title in panel_defs:
        ax = fig.add_subplot(gs[row, col])
        df = l1_df if task == 'L1' else l2_df

        raw_vals    = [get(df, m, 'Raw',    metric) for m in MODELS]
        latent_vals = [get(df, m, 'Latent', metric) for m in MODELS]

        draw_bars(ax, raw_vals, latent_vals,
                  plabel, title,
                  show_ylabel=(col == 0),
                  ylim=(0, 112))
        axes.append(ax)

    # Shared legend below all panels
    patch_raw    = mpatches.Patch(color=COLOR_RAW,    alpha=ALPHA, label='Raw (SNV-normalized)')
    patch_latent = mpatches.Patch(color=COLOR_LATENT, alpha=ALPHA, label='Latent (PI-VAE encoded)')
    fig.legend(handles=[patch_raw, patch_latent],
               loc='lower center', ncol=2,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.50, 0.01),
               edgecolor='gray')

    out_path = 'figures/figure2_model_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
