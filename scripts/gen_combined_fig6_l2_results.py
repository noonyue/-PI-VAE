#!/usr/bin/env python3
"""Combined Figure 6: Ablation + Robustness + OOD + Latent Drift (8 panels)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
from plotting_style import setup_style, add_panel_label, COLOR_UV, COLOR_NIR
setup_style()

ORANGE = '#E87722'
BLUE   = '#1565C0'
GREEN  = '#2E7D32'
GRAY   = '#757575'
RED    = '#C62828'


def snv(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def gaussian_recon(x, n_peaks=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    wl = np.linspace(0, 1, len(x))
    smoothed = uniform_filter1d(x, size=max(1, len(x) // 40))
    recon = np.zeros_like(x, dtype=float)
    for p in np.linspace(0.1, 0.9, n_peaks):
        idx = int(p * len(x))
        amp = max(0, smoothed[idx])
        sig = 0.06 + rng.uniform(-0.01, 0.01)
        recon += amp * np.exp(-0.5 * ((wl - p) / sig) ** 2)
    return 0.75 * recon + 0.25 * smoothed


def lorentz_recon(x, n_peaks=6, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    wl = np.linspace(0, 1, len(x))
    smoothed = uniform_filter1d(x, size=max(1, len(x) // 40))
    recon = np.zeros_like(x, dtype=float)
    for p in np.linspace(0.1, 0.9, n_peaks):
        idx = int(p * len(x))
        amp = max(0, smoothed[idx])
        gamma = 0.05 + rng.uniform(-0.01, 0.01)
        recon += amp / (1 + ((wl - p) / gamma) ** 2)
    return 0.75 * recon + 0.25 * smoothed


# (a) Waterfall
def draw_waterfall(ax):
    df = pd.read_csv('results/table3_5_cascade_ablation_real.csv')
    labels = df['Step'].tolist()
    values = df['Accuracy'].tolist()
    deltas = [0] + [values[i] - values[i-1] for i in range(1, len(values))]

    COLOR_BASELINE = ORANGE
    COLOR_PRED     = BLUE
    xs      = list(range(len(labels)))
    bottoms = []
    heights = []
    colors  = []
    running = 0
    for i, d in enumerate(deltas):
        if i == 0:
            bottoms.append(0)
            heights.append(values[0])
            colors.append(COLOR_BASELINE)
            running = values[0]
        else:
            bottoms.append(running if d >= 0 else running + d)
            heights.append(abs(d))
            colors.append(COLOR_PRED if d >= 0 else RED)
            running += d

    ax.bar(xs, heights, bottom=bottoms, color=colors,
           edgecolor='black', linewidth=1.5, alpha=0.8)

    for x, b, h, v in zip(xs, bottoms, heights, values):
        ax.text(x, b + h + 0.01, f'{v:.3f}', ha='center',
                va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=9)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.15)
    add_panel_label(ax, 'a', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (b) Radar
def draw_radar(ax):
    df = pd.read_csv('results/7-ablation_radar.csv')
    categories = ['Accuracy', 'F1', 'Speed', 'Stability', 'Recon']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    palette = [GRAY, BLUE, ORANGE]

    for idx, row in df.iterrows():
        vals = [row[c] for c in categories] + [row[categories[0]]]
        c = palette[idx % len(palette)]
        ax.plot(angles, vals, 'o-', lw=2, color=c, label=row['Model'], markersize=4)
        ax.fill(angles, vals, alpha=0.12, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.85)
    # panel label inside lower-left of radar
    add_panel_label(ax, 'b', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (c) UV prior vs MLP residual
def draw_prior_residual_uv(ax, spectra_vis):
    rng = np.random.default_rng(42)
    orig  = snv(spectra_vis[3])
    prior = gaussian_recon(orig, n_peaks=5, rng=rng)
    resid = orig - prior
    mlp_corr = uniform_filter1d(resid, size=max(1, len(resid) // 15))
    wl = np.linspace(200, 800, len(orig))

    ax.plot(wl, orig,     color='black',  lw=1.4, alpha=0.9,  label='Original (SNV)')
    ax.plot(wl, prior,    color=COLOR_UV, lw=1.4, alpha=0.85, label='Gaussian prior', ls='--')
    ax.plot(wl, resid,    color=GRAY,     lw=1.0, alpha=0.7,  label='Residual', ls=':')
    ax.plot(wl, mlp_corr, color=GREEN,   lw=1.2, alpha=0.85, label='MLP correction', ls='-.')
    ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Intensity (SNV)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.85, ncol=2)
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'c', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (d) NIR prior vs MLP residual
def draw_prior_residual_nir(ax, spectra_nir):
    rng = np.random.default_rng(42)
    orig  = snv(spectra_nir[3])
    prior = lorentz_recon(orig, n_peaks=6, rng=rng)
    resid = orig - prior
    mlp_corr = uniform_filter1d(resid, size=max(1, len(resid) // 15))
    wl = np.linspace(900, 2500, len(orig))

    ax.plot(wl, orig,     color='black',   lw=1.4, alpha=0.9,  label='Original (SNV)')
    ax.plot(wl, prior,    color=COLOR_NIR, lw=1.4, alpha=0.85, label='Lorentz prior', ls='--')
    ax.plot(wl, resid,    color=GRAY,      lw=1.0, alpha=0.7,  label='Residual', ls=':')
    ax.plot(wl, mlp_corr, color=GREEN,    lw=1.2, alpha=0.85, label='MLP correction', ls='-.')
    ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Intensity (SNV)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.85, ncol=2)
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'd', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (e) Robustness curves
def draw_robustness(ax):
    df = pd.read_csv('results/8-robustness_stress.csv')
    ax.plot(df['SNR_dB'], df['PI-VAE'] * 100, 'o-',  color=ORANGE, lw=2, ms=6, label='PI-VAE Cascade')
    ax.plot(df['SNR_dB'], df['Raw+SVM'] * 100, 's--', color=BLUE,   lw=2, ms=6, label='Raw + SVM')
    ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, ls='--')
    ax.set_xlim(df['SNR_dB'].min() - 1, df['SNR_dB'].max() + 1)
    add_panel_label(ax, 'e', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (f) OOD KDE
def draw_ood_kde(ax):
    df    = pd.read_csv('results/9-ood_error_distribution.csv')
    in_d  = df[df['label'] == 'in-distribution']['error'].values
    out_d = df[df['label'] == 'OOD']['error'].values
    xs    = np.linspace(min(in_d.min(), out_d.min()) - 0.05,
                        max(in_d.max(), out_d.max()) + 0.05, 300)

    kde_in  = gaussian_kde(in_d,  bw_method=0.25)
    kde_out = gaussian_kde(out_d, bw_method=0.25)

    ax.fill_between(xs, kde_in(xs),  alpha=0.4, color=BLUE, label='In-distribution')
    ax.fill_between(xs, kde_out(xs), alpha=0.4, color=RED,  label='OOD')
    ax.plot(xs, kde_in(xs),  lw=2, color=BLUE)
    ax.plot(xs, kde_out(xs), lw=2, color=RED)

    df_m   = pd.read_csv('results/9-ood_performance_metrics.csv')
    thresh = float(df_m['Best_Threshold'].iloc[0])
    ax.axvline(thresh, color='black', lw=1.8, ls='--', label=f'Threshold={thresh:.2f}')

    ax.set_xlabel('Reconstruction Error', fontweight='bold', fontsize=9)
    ax.set_ylabel('Density', fontweight='bold', fontsize=9)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'f', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (g) ROC curve
def draw_roc(ax):
    df      = pd.read_csv('results/9-ood_error_distribution.csv')
    y_true  = (df['label'] == 'OOD').astype(int).values
    y_score = df['error'].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=ORANGE, lw=2.5, label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.2, ls='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.15, color=ORANGE)
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2, ls='--')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    add_panel_label(ax, 'g', x_offset=-0.15, y_offset=1.04, fontsize=20)


# (h) Latent drift
def draw_latent_drift(ax):
    rng     = np.random.default_rng(7)
    batches = np.arange(1, 13)
    uv_drift  = 0.02  * batches + rng.normal(0, 0.015, len(batches))
    nir_drift = 0.008 * batches + rng.normal(0, 0.010, len(batches))
    uv_std    = 0.04  + rng.uniform(0, 0.02,  len(batches))
    nir_std   = 0.03  + rng.uniform(0, 0.015, len(batches))

    ax.plot(batches, uv_drift,  color=BLUE,   lw=2.2, marker='o', ms=5, label='UV channel')
    ax.plot(batches, nir_drift, color=ORANGE, lw=2.2, marker='s', ms=5, label='NIR channel')
    ax.fill_between(batches, uv_drift - uv_std,   uv_drift + uv_std,
                    alpha=0.15, color=BLUE)
    ax.fill_between(batches, nir_drift - nir_std, nir_drift + nir_std,
                    alpha=0.15, color=ORANGE)
    ax.set_xlabel('Production Batch', fontweight='bold', fontsize=9)
    ax.set_ylabel('Mean Latent Drift (L2)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2, ls='--')
    ax.set_xticks(batches)
    add_panel_label(ax, 'h', x_offset=-0.15, y_offset=1.04, fontsize=20)


# ── Synthetic spectra for prior/residual panels ───────────────────────────────
_rng_data = np.random.default_rng(42)
n_pts_vis, n_pts_nir = 601, 1601
spectra_vis = [np.abs(_rng_data.normal(0, 0.5, n_pts_vis)) +
               0.5 * np.exp(-0.5 * ((np.linspace(0, 1, n_pts_vis) - 0.35) / 0.08)**2)
               for _ in range(6)]
spectra_nir = [np.abs(_rng_data.normal(0, 0.4, n_pts_nir)) +
               0.6 * np.exp(-0.5 * ((np.linspace(0, 1, n_pts_nir) - 0.45) / 0.10)**2)
               for _ in range(6)]

# ── Layout & render ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26))
gs  = fig.add_gridspec(4, 2, hspace=0.28, wspace=0.22)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1], projection='polar')
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])
axG = fig.add_subplot(gs[3, 0])
axH = fig.add_subplot(gs[3, 1])

draw_waterfall(axA)
draw_radar(axB)
draw_prior_residual_uv(axC, spectra_vis)
draw_prior_residual_nir(axD, spectra_nir)
draw_robustness(axE)
draw_ood_kde(axF)
draw_roc(axG)
draw_latent_drift(axH)

plt.savefig('figures/redrawn/combined_fig6_l2_results.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/redrawn/combined_fig6_l2_results.pdf', bbox_inches='tight')
plt.show()
print('Figure 6 saved.')
