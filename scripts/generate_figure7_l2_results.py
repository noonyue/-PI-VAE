"""
Figure 7: L2 Manufacturer Identification Results
Combined: (a) Model performance heatmap  |  (b) Per-drug accuracy bar chart

Layout: 1 row x 2 panels
- (a) Heatmap: 9 drugs x 3 models, color = test accuracy, star marks best model
- (b) Bar chart: per-drug best-model accuracy + overall mean line

Output: figures/figure7_l2_results.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label

setup_style()

DRUG_NAMES  = ['CIM', 'FMD', 'GLD', 'GSR', 'HCT', 'IBU', 'MHE', 'MHL', 'MHR']
MODEL_NAMES = ['SVM', 'RF', 'PLS-DA']

COLOR_RF   = '#2CA02C'
COLOR_SVM  = '#1F77B4'
COLOR_PLS  = '#FF7F0E'
COLOR_MEAN = '#D62728'


# ── (a) Heatmap ───────────────────────────────────────────────────────────────
def draw_heatmap(ax, df4, df5):
    mat = np.array([[row['SVM_Acc'], row['RF_Acc'], row['PLS_Acc']]
                    for _, row in df5.iterrows()])

    cmap = LinearSegmentedColormap.from_list(
        'perf', ['#FFFFFF', '#C6EFCE', '#2CA02C'], N=256)

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Test Accuracy', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, fontsize=10.5, fontweight='bold')
    ax.set_yticks(range(len(DRUG_NAMES)))
    ax.set_yticklabels(DRUG_NAMES, fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Classifier', fontsize=10)
    ax.set_ylabel('Drug', fontsize=10)

    best_col_map = {'SVM': 0, 'RandomForest': 1, 'PLS-DA': 2}
    for i, row4 in enumerate(df4.itertuples()):
        best_col = best_col_map.get(row4.Best_Model, 1)
        for j in range(len(MODEL_NAMES)):
            val = mat[i, j]
            txt_color = 'white' if val > 0.65 else 'black'
            star = ' ★' if j == best_col else ''
            ax.text(j, i, f'{val*100:.1f}%{star}',
                    ha='center', va='center',
                    fontsize=8.5, color=txt_color, fontweight='bold')

    ax.set_title('L2 Model Performance Heatmap\n(★ = selected best model per drug)',
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.tick_params(length=0)
    add_panel_label(ax, '(a)', x_offset=-0.14, y_offset=1.04)


# ── (b) Per-drug accuracy bar chart ───────────────────────────────────────────
def draw_per_drug_bar(ax, df4):
    n   = len(DRUG_NAMES)
    acc = df4['Test_Accuracy'].values * 100
    models = df4['Best_Model'].tolist()

    model_short = {'SVM': 'SVM', 'RandomForest': 'RF', 'PLS-DA': 'PLS'}
    bar_colors  = [COLOR_SVM if m == 'SVM' else
                   COLOR_RF  if m == 'RandomForest' else
                   COLOR_PLS for m in models]

    bars = ax.bar(range(n), acc, color=bar_colors,
                  edgecolor='black', linewidth=0.8,
                  alpha=0.88, width=0.6)

    for i, (bar, a, m) in enumerate(zip(bars, acc, models)):
        ax.text(i, a + 0.8, f'{a:.1f}%',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.text(i, a / 2, model_short.get(m, m),
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')

    mean_acc = acc.mean()
    ax.axhline(mean_acc, color=COLOR_MEAN, lw=1.8, linestyle='--', zorder=5)
    ax.text(n - 0.52, mean_acc + 1.2, f'Mean={mean_acc:.1f}%',
            ha='right', va='bottom', fontsize=8.5,
            color=COLOR_MEAN, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_xticklabels(DRUG_NAMES, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.axhline(100, color='gray', lw=0.8, linestyle=':', alpha=0.6)
    ax.set_ylabel('Test Accuracy (%)', fontsize=10)
    ax.set_title('Per-Drug L2 Accuracy (Best Model Selected)',
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.28, linestyle='--')
    ax.tick_params(length=3)

    patches = [
        mpatches.Patch(color=COLOR_RF,  alpha=0.88, label='Random Forest'),
        mpatches.Patch(color=COLOR_SVM, alpha=0.88, label='SVM'),
        mpatches.Patch(color=COLOR_PLS, alpha=0.88, label='PLS-DA'),
    ]
    ax.legend(handles=patches, fontsize=8.5, loc='lower left',
              framealpha=0.9, ncol=3)
    add_panel_label(ax, '(b)', x_offset=-0.12, y_offset=1.04)


def main():
    os.makedirs('figures', exist_ok=True)

    df4 = pd.read_csv('results/4-l2_classification_results.csv')
    df5 = pd.read_csv('results/5-l2_model_performance.csv')

    df4 = df4.set_index('Drug').loc[DRUG_NAMES].reset_index()
    df5 = df5.set_index('Drug').loc[DRUG_NAMES].reset_index()

    fig = plt.figure(figsize=(16, 6.5))
    fig.text(0.50, 0.985,
             'Figure 7.  L2 Manufacturer Identification: '
             'Model Selection & Per-Drug Performance',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.87, bottom=0.10,
                           left=0.07, right=0.97,
                           wspace=0.40,
                           width_ratios=[1.05, 1.0])

    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    draw_heatmap(ax_heat, df4, df5)
    draw_per_drug_bar(ax_bar, df4)

    out_path = 'figures/figure7_l2_results.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
