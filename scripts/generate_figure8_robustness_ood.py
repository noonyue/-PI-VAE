"""
Figure 8: Robustness & OOD Safety Assessment
Combined: (a) SNR robustness stress test  |  (b) OOD detection histogram

Layout: 1 row x 2 panels
- (a) Line chart: SNR (50→10 dB) vs Accuracy for PI-VAE vs Raw+SVM
- (b) Histogram: reconstruction error distribution (in-dist vs OOD),
      threshold line, AUC annotation

Data sources:
  results/8-robustness_stress.csv
  results/9-ood_performance_metrics.csv

Output: figures/figure8_robustness_ood.png
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

COLOR_PIVAE = '#1F77B4'   # blue  — PI-VAE
COLOR_RAW   = '#FF7F0E'   # orange — Raw+SVM
COLOR_IN    = '#2CA02C'   # green  — in-distribution
COLOR_OOD   = '#D62728'   # red    — OOD
COLOR_THRESH= '#9467BD'   # purple — threshold


# ── (a) Robustness stress test ─────────────────────────────────────────────────
def draw_robustness(ax, df):
    snr    = df['SNR_dB'].values
    pivae  = df['PI-VAE'].values * 100
    raw    = df['Raw+SVM'].values * 100

    ax.plot(snr, raw,   color=COLOR_RAW,   lw=2.2, marker='s', markersize=7,
            label='Raw + SVM (baseline)', zorder=4)
    ax.plot(snr, pivae, color=COLOR_PIVAE, lw=2.2, marker='o', markersize=7,
            label='PI-VAE (ours)', zorder=5)

    # Highlight crossover zone (SNR ≤ 20 dB)
    ax.axvspan(10, 20, color='#FFDDDD', alpha=0.35, label='High-noise region')
    ax.axvline(20, color='gray', lw=1.0, linestyle=':', alpha=0.7)

    # Annotate key values at SNR=10
    for val, col, va in [(pivae[-1], COLOR_PIVAE, 'bottom'),
                         (raw[-1],   COLOR_RAW,   'top')]:
        ax.annotate(f'{val:.1f}%',
                    xy=(snr[-1], val),
                    xytext=(3, 6 if va == 'bottom' else -6),
                    textcoords='offset points',
                    fontsize=8.5, color=col, fontweight='bold', va=va)

    # Advantage annotation
    adv = pivae[-1] - raw[-1]
    ax.text(12, (pivae[-1] + raw[-1]) / 2,
            f'PI-VAE\n+{adv:.1f}%\nadvantage',
            ha='left', va='center', fontsize=8,
            color=COLOR_PIVAE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLOR_PIVAE, alpha=0.85))

    ax.set_xlim(52, 8)          # descending SNR (left=clean, right=noisy)
    ax.set_ylim(0, 110)
    ax.set_xlabel('SNR (dB)  ← noisier', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_xticks(snr)
    ax.set_xticklabels([f'{s} dB' for s in snr], fontsize=9)
    ax.axhline(100, color='gray', lw=0.8, linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(length=3)
    ax.set_title('Noise Robustness Stress Test\n(Additive Gaussian noise, L2 classification)',
                 fontsize=10.5, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    add_panel_label(ax, '(a)', x_offset=-0.13, y_offset=1.04)


# ── (b) OOD detection histogram ────────────────────────────────────────────────
def draw_ood_histogram(ax, ood_metrics, ood_dist):
    threshold = ood_metrics['Best_Threshold'].iloc[0]
    auc       = ood_metrics['AUC'].iloc[0]
    tpr       = ood_metrics['TPR_at_best'].iloc[0]
    fpr       = ood_metrics['FPR_at_best'].iloc[0]

    # Load real reconstruction error distributions from experiment
    err_in  = ood_dist[ood_dist['label'] == 'in-distribution']['error'].values
    err_ood = ood_dist[ood_dist['label'] == 'OOD']['error'].values
    n_in  = len(err_in)
    n_ood = len(err_ood)

    x_max = max(err_ood.max(), err_in.max()) * 1.05
    bins = np.linspace(0, x_max, 40)
    ax.hist(err_in,  bins=bins, color=COLOR_IN,  alpha=0.70,
            label=f'In-distribution (n={n_in})',  density=True, edgecolor='white', lw=0.4)
    ax.hist(err_ood, bins=bins, color=COLOR_OOD, alpha=0.70,
            label=f'OOD samples (n={n_ood})',     density=True, edgecolor='white', lw=0.4)

    # Threshold line
    ax.axvline(threshold, color=COLOR_THRESH, lw=2.2, linestyle='--',
               label=f'Threshold = {threshold:.3f}')

    # Metrics box
    info = (f'AUC  = {auc:.3f}\n'
            f'TPR = {tpr*100:.1f}%\n'
            f'FPR = {fpr*100:.1f}%')
    ax.text(0.97, 0.95, info, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.92))

    # Shade rejection zone
    x_fill = np.linspace(threshold, bins[-1], 200)
    ax.axvspan(threshold, bins[-1], color=COLOR_OOD, alpha=0.08,
               label='Rejected (OOD) zone')

    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(0, x_max)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(length=3)
    ax.set_title('OOD Detection via Reconstruction Error\n(Perfect separation: AUC = 1.000)',
                 fontsize=10.5, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, loc='upper center', framealpha=0.9, ncol=2)
    add_panel_label(ax, '(b)', x_offset=-0.13, y_offset=1.04)


def main():
    os.makedirs('figures', exist_ok=True)

    df_rob  = pd.read_csv('results/8-robustness_stress.csv')
    df_ood  = pd.read_csv('results/9-ood_performance_metrics.csv')
    df_dist = pd.read_csv('results/9-ood_error_distribution.csv')  # real error distributions

    fig = plt.figure(figsize=(16, 6.5))
    fig.text(0.50, 0.985,
             'Figure 8.  Robustness & Out-of-Distribution Safety Assessment',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.87, bottom=0.12,
                           left=0.07, right=0.97,
                           wspace=0.36)

    ax_rob = fig.add_subplot(gs[0])
    ax_ood = fig.add_subplot(gs[1])

    draw_robustness(ax_rob, df_rob)
    draw_ood_histogram(ax_ood, df_ood, df_dist)

    out_path = 'figures/figure8_robustness_ood.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
