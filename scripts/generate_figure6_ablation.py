"""
Figure 6: Ablation Study & Performance Decomposition
Combined: (a) Waterfall chart  |  (b) Radar chart

Layout: 1 row x 2 panels
- (a) Stepwise accuracy waterfall (Direct → Standard AE → PI-VAE → Cascade)
- (b) Multi-dimensional radar (Accuracy / F1 / Speed / Stability / Recon)
    Each dimension independently min-max normalized to [0,1].
    Speed is inverted (lower inference time → higher score).

Output: figures/figure6_ablation_performance.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label

setup_style()

COLOR_BASELINE = '#FF7F0E'
COLOR_GAIN     = '#1F77B4'
COLOR_FINAL    = '#2CA02C'
COLOR_ZERO     = '#AAAAAA'

RADAR_COLORS = ['#FF7F0E', '#9467BD', '#1F77B4']


# ── Waterfall chart ────────────────────────────────────────────────────────────
def draw_waterfall(ax):
    steps = [
        ('Direct\n28-class',    0.611111),
        ('+ Standard\nAE',      0.930556),
        ('+ PI-VAE\nPrior',     0.930556),
        ('+ Cascade\nStrategy', 0.974537),
    ]
    labels = [s[0] for s in steps]
    values = [s[1] * 100 for s in steps]
    n = len(steps)

    # Compute bottoms & heights
    bottoms = [0]
    heights = [values[0]]
    colors  = [COLOR_BASELINE]
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        bottoms.append(values[i - 1])
        heights.append(abs(delta) if delta != 0 else 1e-6)   # zero-delta: invisible sliver
        colors.append(COLOR_GAIN if delta > 0 else (COLOR_ZERO if delta == 0 else '#D62728'))

    # Last bar: full-height green
    bottoms[-1] = 0
    heights[-1] = values[-1]
    colors[-1]  = COLOR_FINAL

    ax.bar(range(n), heights, bottom=bottoms,
           color=colors, edgecolor='black', linewidth=1.0,
           alpha=0.88, width=0.55)

    # Value labels on top of each bar
    for i, (bot, h, val) in enumerate(zip(bottoms, heights, values)):
        ax.text(i, bot + h + 0.5, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Delta annotations between consecutive bars
    for i in range(1, n - 1):
        delta = values[i] - values[i - 1]
        if delta > 0:
            mid_y = values[i - 1] + delta / 2
            ax.text(i + 0.33, mid_y,
                    f'+{delta:.2f}%', va='center', fontsize=8,
                    color=COLOR_GAIN, fontweight='bold')
        elif delta == 0:
            ax.text(i + 0.33, values[i] + 4,
                    '±0%', va='center', fontsize=8,
                    color='#888888', fontstyle='italic')

    # Total-gain double-headed arrow
    total_gain = values[-1] - values[0]
    ax.annotate('', xy=(n - 1, values[-1] + 1),
                xytext=(0, values[0] + 1),
                arrowprops=dict(arrowstyle='<->', color='gray',
                                lw=1.5, connectionstyle='arc3,rad=0'))
    ax.text((n - 1) / 2, max(values) + 5,
            f'Total gain: +{total_gain:.2f}%',
            ha='center', fontsize=9, color='gray', fontstyle='italic')

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Stepwise Accuracy Decomposition', fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.28, linestyle='--')
    ax.axhline(100, color='gray', lw=0.8, linestyle=':', alpha=0.6)
    ax.tick_params(length=3)

    patches = [
        mpatches.Patch(color=COLOR_BASELINE, alpha=0.88, label='Baseline'),
        mpatches.Patch(color=COLOR_GAIN,     alpha=0.88, label='Improvement'),
        mpatches.Patch(color=COLOR_FINAL,    alpha=0.88, label='Final result'),
    ]
    ax.legend(handles=patches, fontsize=8.5, loc='lower right')
    add_panel_label(ax, '(a)', x_offset=-0.13, y_offset=1.04)


# ── Radar chart (polar) ────────────────────────────────────────────────────────
def draw_radar_polar(fig, gs_slot, df, categories):
    ax = fig.add_subplot(gs_slot, projection='polar')
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    # ── Normalization: keep [0,1] dims as-is; convert Speed to efficiency ──
    df_norm = df.copy()
    # Accuracy, F1, Stability, Recon are already in [0,1] — use directly
    # Speed = inference time (seconds, smaller = better)
    # → efficiency score = min_speed / speed  ∈ (0, 1]
    min_speed = df['Speed'].min()
    df_norm['Speed'] = min_speed / df['Speed']

    # Background rings
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles, [r] * (N + 1), color='gray', lw=0.5, alpha=0.35)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='gray')
    ax.set_ylim(0, 1.1)
    ax.spines['polar'].set_visible(False)

    model_names = df['Model'].tolist()
    for i, row in df_norm.iterrows():
        vals = [row[c] for c in categories]
        vals += vals[:1]
        col = RADAR_COLORS[i % len(RADAR_COLORS)]
        ax.plot(angles, vals, color=col, lw=2.0, label=model_names[i])
        ax.fill(angles, vals, color=col, alpha=0.13)
        ax.scatter(angles[:-1], vals[:-1], color=col, s=35, zorder=5)

    ax.set_title('Multi-Dimensional Model Comparison\n(Normalized per dimension)',
                 fontsize=10.5, fontweight='bold', pad=16)

    # Radial tick labels: show only on one radial direction to avoid clutter
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=6.5, color='gray')
    ax.set_rlabel_position(45)   # place radial labels at 45° to avoid overlap

    # Legend: placed inside the radar axes at lower-left (avoids clipping)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles, labels_leg,
              loc='lower left', bbox_to_anchor=(-0.18, -0.22),
              fontsize=8.5, framealpha=0.9, ncol=1)

    # (b) panel label — use fig.text to avoid polar-axes coordinate issues
    return ax


def main():
    os.makedirs('figures', exist_ok=True)

    df_radar    = pd.read_csv('results/7-ablation_radar.csv')
    categories  = ['Accuracy', 'F1', 'Speed', 'Stability', 'Recon']

    fig = plt.figure(figsize=(16, 7.5))
    # Main title — leave enough headroom
    fig.text(0.50, 0.98,
             'Figure 6.  Ablation Study & Multi-Dimensional Performance Analysis',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.88, bottom=0.10,
                           left=0.06, right=0.96,
                           wspace=0.46)

    ax_water = fig.add_subplot(gs[0])
    draw_waterfall(ax_water)

    ax_radar = draw_radar_polar(fig, gs[1], df_radar, categories)

    # (b) panel label using fig.text positioned relative to the right half
    # Right panel occupies roughly x=[0.51, 0.96] in figure coords
    fig.text(0.515, 0.90, '(b)', fontsize=14, fontweight='bold',
             ha='left', va='bottom')

    out_path = 'figures/figure6_ablation_performance.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
