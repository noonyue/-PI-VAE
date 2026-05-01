#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: PI-VAE Architecture – 3-section horizontal layout.
No LaTeX math (Unicode only). Windows-safe fonts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from plotting_style import setup_style
setup_style()

CUV  = '#2166AC'
CNIR = '#C0392B'
CLAT = '#27AE60'
CFUS = '#D4860B'
CL1  = '#6C3483'
CL2  = '#1A5276'
CLOS = '#7F8C8D'
CRAW = '#34495E'


def rbox(ax, x, y, w, h, txt, fc, ec='#2C3E50', tc='white',
         fs=7.5, fw='bold', alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle='round,pad=0.09', facecolor=fc,
                                edgecolor=ec, lw=1.1, alpha=alpha, zorder=z))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, zorder=z+1, linespacing=1.35)


def rarr(ax, x0, y0, x1, y1, c='#555', lw=1.3):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))


def nn_layer(ax, cx, cy, n, r=0.06, fc='white', ec='#444'):
    gap = r * 2.7
    xs = np.linspace(cx - gap*(n-1)/2, cx + gap*(n-1)/2, n)
    for xi in xs:
        ax.add_patch(plt.Circle((xi, cy), r, color=fc, ec=ec, lw=0.9, zorder=5))
    return xs


def draw_mlp(ax, cx, y_top, clr, n_list=(5, 4, 3, 2), gap=0.5):
    prev_xs, ys = None, [y_top - i*gap for i in range(len(n_list))]
    for i, (n, y) in enumerate(zip(n_list, ys)):
        fc = CLAT if i == len(n_list)-1 else clr
        xs = nn_layer(ax, cx, y, n, fc=fc, ec='#333')
        if prev_xs is not None:
            for px in prev_xs:
                for nx in xs:
                    ax.plot([px, nx], [ys[i-1], y], color='#ccc', lw=0.35, zorder=2)
        prev_xs = xs
    return ys  # list of y coords per layer

def draw_section_a(ax):
    """Left section: dual-modal input spectra."""
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 4.8, 9.6,
                                boxstyle='round,pad=0.1', fc='#EEF5FB',
                                ec='#AAC4DF', lw=1.5, zorder=0))
    ax.text(2.55, 9.65, 'Input Spectra', ha='center', fontsize=10,
            fontweight='bold', color='#1a1a1a')

    lam = np.linspace(0, 1, 200)

    # UV-Vis (top)
    yuv = (0.55*np.exp(-((lam-0.28)/0.08)**2) +
           0.90*np.exp(-((lam-0.55)/0.10)**2) +
           0.40*np.exp(-((lam-0.82)/0.07)**2))
    px = 0.4 + lam*4.1
    ax.plot(px, 6.5 + yuv*1.8, color=CUV, lw=2.0, zorder=3)
    ax.fill_between(px, 6.5, 6.5+yuv*1.8, alpha=0.22, color=CUV)
    ax.text(2.55, 6.15, 'UV-Vis Spectrum', ha='center', fontsize=8,
            color=CUV, fontweight='bold')
    ax.text(2.55, 5.78, '200-700 nm  |  Gaussian peaks', ha='center',
            fontsize=6.5, color='#555')
    rarr(ax, 4.42, 8.0, 4.95, 8.0, CUV, lw=1.8)

    # NIR (bottom)
    ynir = (0.65/(1+((lam-0.25)/0.07)**2) +
            0.90/(1+((lam-0.55)/0.09)**2) +
            0.45/(1+((lam-0.82)/0.06)**2))
    ax.plot(px, 2.3+ynir*1.8, color=CNIR, lw=2.0, zorder=3)
    ax.fill_between(px, 2.3, 2.3+ynir*1.8, alpha=0.22, color=CNIR)
    ax.text(2.55, 1.95, 'NIR Spectrum', ha='center', fontsize=8,
            color=CNIR, fontweight='bold')
    ax.text(2.55, 1.58, '700-2500 nm  |  Lorentzian peaks', ha='center',
            fontsize=6.5, color='#555')
    rarr(ax, 4.42, 3.7, 4.95, 3.7, CNIR, lw=1.8)

    rbox(ax, 2.55, 4.6, 2.8, 0.55, 'Preprocessing: SNV Normalisation',
         '#546E7A', tc='white', fs=7, fw='normal')


def draw_section_b(ax):
    """Middle section: PI-VAE encoders + decoders + loss."""
    ax.add_patch(FancyBboxPatch((5.1, 0.2), 10.8, 9.6,
                                boxstyle='round,pad=0.1', fc='#F8FDF3',
                                ec='#99BB88', lw=1.5, zorder=0))
    ax.text(10.5, 9.65,
            'Physics-Informed Variational Autoencoder (PI-VAE)',
            ha='center', fontsize=9.5, fontweight='bold', color='#1a1a1a')

    # ── UV lane background
    ax.add_patch(FancyBboxPatch((5.25, 5.4), 10.5, 4.1,
                                boxstyle='round,pad=0.05', fc='#E8F4FD',
                                ec='none', alpha=0.6, zorder=1))
    ax.text(5.5, 9.3, 'UV-Vis Path', fontsize=7.5,
            color=CUV, fontweight='bold')

    # UV Encoder MLP
    ys_uv = draw_mlp(ax, cx=7.0, y_top=8.8, clr=CUV)
    ax.text(7.0, 8.8+0.22, 'UV Encoder', ha='center', fontsize=7,
            color=CUV, fontweight='bold')
    ax.text(7.0, ys_uv[0]-0.18, 'Input (400-d)', ha='center',
            fontsize=6, color='#555')
    ax.text(7.0, ys_uv[1]-0.18, 'FC-256', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_uv[2]-0.18, 'FC-128', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_uv[3]-0.18, u'\u03bc,\u03c3\u00b2 (32-d)',
            ha='center', fontsize=6, color=CLAT, fontweight='bold')

    # Reparameterisation + latent
    rbox(ax, 9.2, 7.3, 1.6, 0.5, u'z ~ N(\u03bc,\u03c3\u00b2)\nReparam.',
         '#FFFACD', tc='#333', fs=6.5)
    rarr(ax, 7.35, ys_uv[3], 8.4, 7.3, CLAT)
    rbox(ax, 11.0, 7.3, 1.3, 0.5, 'z_UV\n(32-dim)', CLAT, fs=7)
    rarr(ax, 10.0, 7.3, 10.35, 7.3, CLAT)

    # Gaussian decoder
    rbox(ax, 12.9, 7.3, 1.7, 0.5, 'Gaussian\nPeak Decoder', CUV, fs=7)
    rarr(ax, 11.65, 7.3, 12.05, 7.3, CUV)

    # Reconstructed UV
    lx = np.linspace(14.1, 15.7, 100)
    ry = 7.0 + 0.75*np.exp(-((lx-14.9)/0.28)**2) + 0.45*np.exp(-((lx-15.4)/0.22)**2)
    ax.plot(lx, ry, color=CUV, lw=1.8)
    ax.fill_between(lx, 7.0, ry, alpha=0.2, color=CUV)
    ax.text(14.9, 6.72, 'x\u0302_UV', ha='center', fontsize=7.5,
            color=CUV, fontweight='bold')
    rarr(ax, 13.75, 7.3, 14.1, 7.5, CUV)

    # UV physics formula box
    ax.text(14.9, 6.3,
            u'f(\u03bb) = \u03a3 A\u2096 exp[-(\u03bb-\u03bc\u2096)\u00b2 / 2\u03c3\u2096\u00b2]',
            ha='center', fontsize=6.5, color='#222',
            bbox=dict(fc='white', ec=CUV, lw=0.8, boxstyle='round,pad=0.25'))

    # beta-VAE loss UV
    rbox(ax, 11.0, 6.1, 2.0, 0.48,
         u'\u03b2-VAE Loss: L_rec + \u03b2 D_KL', CLOS, fs=6.5)
    rarr(ax, 11.0, 7.05, 11.0, 6.34, CLOS, lw=0.9)
    rarr(ax, 12.9, 7.05, 12.9, 6.34, CLOS, lw=0.9)

    # ── NIR lane background
    ax.add_patch(FancyBboxPatch((5.25, 0.5), 10.5, 4.6,
                                boxstyle='round,pad=0.05', fc='#FDEDEC',
                                ec='none', alpha=0.6, zorder=1))
    ax.text(5.5, 4.85, 'NIR Path', fontsize=7.5, color=CNIR, fontweight='bold')

    # NIR Encoder MLP
    ys_nir = draw_mlp(ax, cx=7.0, y_top=4.3, clr=CNIR)
    ax.text(7.0, 4.3+0.22, 'NIR Encoder', ha='center', fontsize=7,
            color=CNIR, fontweight='bold')
    ax.text(7.0, ys_nir[0]-0.18, 'Input (200-d)', ha='center',
            fontsize=6, color='#555')
    ax.text(7.0, ys_nir[1]-0.18, 'FC-256', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_nir[2]-0.18, 'FC-128', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_nir[3]-0.18, u'\u03bc,\u03c3\u00b2 (32-d)',
            ha='center', fontsize=6, color=CLAT, fontweight='bold')

    rbox(ax, 9.2, 2.8, 1.6, 0.5, u'z ~ N(\u03bc,\u03c3\u00b2)\nReparam.',
         '#FFFACD', tc='#333', fs=6.5)
    rarr(ax, 7.35, ys_nir[3], 8.4, 2.8, CLAT)
    rbox(ax, 11.0, 2.8, 1.3, 0.5, 'z_NIR\n(32-dim)', CLAT, fs=7)
    rarr(ax, 10.0, 2.8, 10.35, 2.8, CLAT)

    rbox(ax, 12.9, 2.8, 1.7, 0.5, 'Lorentzian\nPeak Decoder', CNIR, fs=7)
    rarr(ax, 11.65, 2.8, 12.05, 2.8, CNIR)

    lx2 = np.linspace(14.1, 15.7, 100)
    ry2 = 2.5 + 0.70/(1+((lx2-14.9)/0.22)**2) + 0.42/(1+((lx2-15.4)/0.18)**2)
    ax.plot(lx2, ry2, color=CNIR, lw=1.8)
    ax.fill_between(lx2, 2.5, ry2, alpha=0.2, color=CNIR)
    ax.text(14.9, 2.22, 'x\u0302_NIR', ha='center', fontsize=7.5,
            color=CNIR, fontweight='bold')
    rarr(ax, 13.75, 2.8, 14.1, 3.0, CNIR)

    ax.text(14.9, 1.78,
            u'f(\u03bb) = \u03a3 A\u2096 / [1 + ((\u03bb-\u03bb\u2096)/\u0393\u2096)\u00b2]',
            ha='center', fontsize=6.5, color='#222',
            bbox=dict(fc='white', ec=CNIR, lw=0.8, boxstyle='round,pad=0.25'))

    rbox(ax, 11.0, 1.55, 2.0, 0.48,
         u'\u03b2-VAE Loss: L_rec + \u03b2 D_KL', CLOS, fs=6.5)
    rarr(ax, 11.0, 2.55, 11.0, 1.79, CLOS, lw=0.9)
    rarr(ax, 12.9, 2.55, 12.9, 1.79, CLOS, lw=0.9)

    # arrows from Section A
    rarr(ax, 4.97, 8.0, 6.4, 8.5, CUV, lw=1.6)
    rarr(ax, 4.97, 3.7, 6.4, 3.8, CNIR, lw=1.6)

    # z_UV / z_NIR → Section C
    rarr(ax, 11.65, 7.3, 15.85, 7.8, CLAT, lw=1.6)
    rarr(ax, 11.65, 2.8, 15.85, 6.5, CLAT, lw=1.6)


def draw_section_c(ax):
    """Right section: feature fusion + cascade classifier."""
    ax.add_patch(FancyBboxPatch((16.0, 0.2), 7.9, 9.6,
                                boxstyle='round,pad=0.1', fc='#F5F0FF',
                                ec='#9B59B6', lw=1.5, zorder=0))
    ax.text(19.95, 9.65, 'Cascade Classifier', ha='center',
            fontsize=10, fontweight='bold', color='#1a1a1a')

    # Feature fusion
    rbox(ax, 19.95, 8.7, 6.8, 0.72,
         'Feature Fusion:  z_UV  \u2295  z_NIR  \u2295  Raw Spectra  (64+64+raw)',
         CFUS, tc='black', fs=8)
    rbox(ax, 17.2, 7.5, 1.5, 0.5, 'Raw\nFeatures', CRAW, fs=6.5)
    rarr(ax, 17.2, 7.75, 17.8, 8.34, CRAW, lw=0.9)

    # L1 SVM
    rbox(ax, 19.95, 7.6, 5.2, 0.65,
         'L1: SVM (RBF Kernel)  -  Drug-type Classification', CL1, fs=8)
    rarr(ax, 19.95, 8.34, 19.95, 7.93, CL1, lw=1.6)
    ax.text(19.95, 7.15, '\u2713 L1 Accuracy: 100%', ha='center',
            fontsize=8, color=CL1, fontweight='bold')

    # 9 drug nodes
    drugs = ['CIM','FMD','GLD','GSR','HCT','IBU','MHE','MHL','MHR']
    xs9 = np.linspace(16.5, 23.4, 9)
    y_d = 6.45
    for xi, d in zip(xs9, drugs):
        rbox(ax, xi, y_d, 0.82, 0.46, d, CL1, fs=6.5, alpha=0.8)
        rarr(ax, 19.95, 6.83, xi, y_d+0.23, CL1, lw=0.75)

    # L2 per-drug model
    l2m = ['SVM','RF','PLS','RF','SVM','PLS','SVM','RF','SVM']
    accs = ['100%','100%','100%','100%','100%','83%','100%','100%','92%']
    y_l2 = 5.35
    for xi, m, a in zip(xs9, l2m, accs):
        rbox(ax, xi, y_l2, 0.82, 0.46, f'L2\n{m}', CL2, fs=6, alpha=0.85)
        rarr(ax, xi, y_d-0.23, xi, y_l2+0.23, CL2, lw=0.85)
        col = CL2 if a == '100%' else CNIR
        ax.text(xi, y_l2-0.38, a, ha='center', fontsize=6,
                color=col, fontweight='bold')

    # 28 manufacturers (compact dots)
    y_mfr = 4.2
    n_mfrs = [3,3,2,2,3,4,3,3,5]
    for xi, nm in zip(xs9, n_mfrs):
        show = min(nm, 3)
        mxs = np.linspace(xi-0.35, xi+0.35, show)
        for mx in mxs:
            rbox(ax, mx, y_mfr, 0.26, 0.32, 'M', '#2471A3',
                 tc='white', fs=5.5, alpha=0.75)
            rarr(ax, xi, y_l2-0.23, mx, y_mfr+0.16, CL2, lw=0.5)
        if nm > 3:
            ax.text(xi+0.48, y_mfr, f'+{nm-3}', fontsize=5,
                    color='#555', va='center')

    ax.text(19.95, 3.7, '28 Manufacturers Total', ha='center',
            fontsize=7, color='#555', style='italic')

    # Overall badge
    rbox(ax, 19.95, 3.0, 6.0, 0.7,
         '\u2605  L2 Overall Accuracy: 97.22%  (7/9 drugs: 100%)', CL2, fs=9)
    ax.text(19.95, 2.45,
            'vs. Baseline Raw+SVM: 88.89%   Improvement: +8.33 pp',
            ha='center', fontsize=7.5, color='#333',
            bbox=dict(fc='#FFFDE7', ec='#F0A500', lw=0.8,
                      boxstyle='round,pad=0.25'))

def main():
    os.makedirs('figures/redrawn', exist_ok=True)
    fig, ax = plt.subplots(figsize=(24, 10), facecolor='white')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')

    draw_section_a(ax)
    draw_section_b(ax)
    draw_section_c(ax)

    # Section dividers
    ax.axvline(5.05, color='#CCC', lw=1.0, ls='--', ymin=0.03, ymax=0.97)
    ax.axvline(15.95, color='#CCC', lw=1.0, ls='--', ymin=0.03, ymax=0.97)

    # Section header labels
    for xc, lbl in [(2.55, 'A. Input'), (10.5, 'B. PI-VAE Core'), (19.95, 'C. Cascade Classifier')]:
        ax.text(xc, 0.08, lbl, ha='center', fontsize=8,
                color='#555', style='italic')

    fig.suptitle('Figure 2.  PI-VAE System Architecture & Physics-Informed Decoder Design',
                 fontsize=12, fontweight='bold', y=1.01)

    out = 'figures/redrawn/combined_fig2_arch_flow.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'[OK] {out}')
    plt.close()


if __name__ == '__main__':
    main()



