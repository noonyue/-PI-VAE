#!/usr/bin/env python3
"""Combined Figure 7: Ablation Study — Waterfall + Radar Chart (2 panels)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from plotting_style import setup_style, add_panel_label
setup_style()

ORANGE = '#E87722'; BLUE = '#1565C0'; GREEN = '#2E7D32'; GRAY = '#757575'; RED = '#C62828'

def draw_waterfall(ax):
    df = pd.read_csv('results/table3_5_cascade_ablation_real.csv')
    labels = df['Step'].tolist()
    values = df['Accuracy'].tolist()
    deltas = [0] + [values[i] - values[i-1] for i in range(1, len(values))]

    ax.bar(0, values[0], width=0.6, color=GRAY, alpha=0.75, edgecolor='black', lw=1.5)
    for i in range(1, len(values)):
        d = deltas[i]
        color = ORANGE if i == len(values)-1 else (BLUE if d >= 0 else RED)
        bottom = values[i-1] if d >= 0 else values[i]
        ax.bar(i, abs(d), width=0.6, bottom=bottom,
               color=color, alpha=0.82, edgecolor='black', lw=1.5)
        mid_y = values[i-1] + d / 2
        sign = '+' if d >= 0 else ''
        ax.text(i, mid_y, f'{sign}{d:.1f}pp', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')
    for i, v in enumerate(values):
        ax.text(i, v + 1.2, f'{v:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='black')

    ax.axhline(values[-1], color=ORANGE, ls=':', lw=1.8, alpha=0.6)
    ax.text(len(values)-0.5, values[-1]+1.0, f'Final: {values[-1]:.2f}%',
            fontsize=9, color=ORANGE, fontweight='bold', style='italic')

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel('L2 Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 108); ax.set_xlim(-0.6, len(labels)-0.4)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, ls='--'); ax.set_axisbelow(True)
    ax.set_title('a Stepwise Ablation Waterfall\n(Marginal contribution of each component)',
                 fontsize=10.5, fontweight='bold')
    legend_els = [
        mpatches.Patch(facecolor=GRAY,   edgecolor='black', label='Baseline (Direct 28-class)'),
        mpatches.Patch(facecolor=BLUE,   edgecolor='black', label='Positive contribution (+)'),
        mpatches.Patch(facecolor=RED,    edgecolor='black', label='Negative contribution (−)'),
        mpatches.Patch(facecolor=ORANGE, edgecolor='black', label='Cascade strategy (Final)'),
    ]
    ax.legend(handles=legend_els, loc='upper left', fontsize=8.5, framealpha=0.95)
    add_panel_label(ax, 'a', x_offset=-0.10, y_offset=1.04, fontsize=20)

def draw_radar(ax):
    df = pd.read_csv('results/7-ablation_radar.csv')
    metrics = ['Accuracy','F1','Speed','Stability','Recon']
    labels_cn = ['Accuracy','Macro-F1','Speed','Stability','Recon Quality']
    models = df['Model'].tolist()
    colors_r = ['#E87722','#1565C0','#1F77B4']
    linestyles = ['--','-.','-']

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_cn, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=7, color='gray')
    ax.grid(True, alpha=0.35)

    for i, (model, col, ls) in enumerate(zip(models, colors_r, linestyles)):
        vals = df[df['Model']==model][metrics].values.flatten().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=col, lw=2.2, ls=ls, label=model)
        ax.fill(angles, vals, color=col, alpha=0.1)

    ax.set_title('b Multi-Dimensional Ablation\n(Radar chart comparison)',
                 fontsize=10.5, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.08), fontsize=8.5, framealpha=0.9)
    # panel label manually for polar ax
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='bottom', ha='right')

def main():
    os.makedirs('figures/redrawn', exist_ok=True)
    fig = plt.figure(figsize=(16, 6.5), facecolor='white')
    fig.suptitle('Figure 7.  Ablation Study: Stepwise Contribution Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                           left=0.07, right=0.97, top=0.88, bottom=0.13)
    draw_waterfall(fig.add_subplot(gs[0]))
    draw_radar(fig.add_subplot(gs[1], polar=True))
    plt.savefig('figures/redrawn/combined_fig7_ablation.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    print('[OK] figures/redrawn/combined_fig7_ablation.png')
    plt.close()

if __name__ == '__main__':
    main()
