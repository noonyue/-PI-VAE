"""Matplotlib graphical abstract for PI-VAE paper."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import os

np.random.seed(42)
OUT = r"d:\work\class\GEN_MODEL\figures\paper\ai_graphical_abstract.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────
C_UV   = '#1F77B4'   # blue   – UV-Vis
C_NIR  = '#D62728'   # red    – NIR
C_VAE  = '#2CA02C'   # green  – PI-VAE box
C_L1   = '#FF7F0E'   # orange – L1
C_L2   = '#9467BD'   # purple – L2
C_BG   = '#F8F9FA'
C_DARK = '#2C3E50'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig = plt.figure(figsize=(16, 5), facecolor='white')
fig.patch.set_facecolor('white')

# Five columns: spectra-UV | spectra-NIR | PI-VAE | L1 | L2
gs = GridSpec(1, 5, figure=fig, wspace=0.55,
              left=0.03, right=0.97, top=0.88, bottom=0.14)

# ── helpers ─────────────────────────────────────────────────────────
def panel_box(ax, title, color):
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(2)
    ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=6)

def arrow_between(fig, ax_left, ax_right, label=''):
    """Draw arrow in figure coords from right edge of ax_left to left edge of ax_right."""
    xl = ax_left.get_position().x1
    xr = ax_right.get_position().x0
    yc = (ax_left.get_position().y0 + ax_left.get_position().y1) / 2
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.annotate('', xy=(xr + 0.005, yc), xytext=(xl - 0.005, yc),
                arrowprops=dict(arrowstyle='->', color=C_DARK,
                                lw=2, mutation_scale=16))
    if label:
        ax.text((xl + xr) / 2, yc + 0.06, label, ha='center',
                fontsize=8, color=C_DARK, style='italic')

# ── Panel A: UV-Vis spectrum ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
panel_box(ax1, 'A  UV-Vis Spectrum', C_UV)
wl = np.linspace(200, 400, 300)
spec = (0.6 * np.exp(-((wl - 254)**2) / (2 * 12**2)) +
        0.35 * np.exp(-((wl - 310)**2) / (2 * 18**2)) +
        0.04 * np.random.randn(300))
ax1.plot(wl, spec, color=C_UV, lw=2)
ax1.fill_between(wl, spec, alpha=0.18, color=C_UV)
ax1.set_xlabel('Wavelength (nm)', fontsize=8)
ax1.set_ylabel('Absorbance', fontsize=8)
ax1.tick_params(labelsize=7)
ax1.set_xlim(200, 400)
ax1.annotate('Gaussian\nprior', xy=(254, 0.6), xytext=(290, 0.72),
             fontsize=7, color=C_UV,
             arrowprops=dict(arrowstyle='->', color=C_UV, lw=1))

# ── Panel B: NIR spectrum ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
panel_box(ax2, 'B  NIR Spectrum', C_NIR)
wn = np.linspace(4000, 10000, 400)
nir = (0.5 * np.exp(-((wn - 5900)**2) / (2 * 300**2)) +
       0.4 * np.exp(-((wn - 7100)**2) / (2 * 250**2)) +
       0.25 * np.exp(-((wn - 8600)**2) / (2 * 350**2)) +
       0.03 * np.random.randn(400))
ax2.plot(wn, nir, color=C_NIR, lw=2)
ax2.fill_between(wn, nir, alpha=0.18, color=C_NIR)
ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=8)
ax2.set_ylabel('Absorbance', fontsize=8)
ax2.tick_params(labelsize=7)
ax2.set_xlim(4000, 10000)
ax2.invert_xaxis()
ax2.annotate('Lorentzian\nprior', xy=(5900, 0.5), xytext=(7200, 0.65),
             fontsize=7, color=C_NIR,
             arrowprops=dict(arrowstyle='->', color=C_NIR, lw=1))

# ── Panel C: PI-VAE latent space (scatter) ──────────────────────────
ax3 = fig.add_subplot(gs[2])
panel_box(ax3, 'C  PI-VAE Latent Space', C_VAE)
drug_labels = ['CIM', 'FMD', 'GLD', 'GSR', 'IBU', 'MHE', 'MHL', 'MHR', 'HCT']
colors9 = plt.cm.tab10(np.linspace(0, 0.9, 9))
centers = [(-2.5, 1.5), (1.8, 2.2), (-0.5, -2.0), (2.5, -1.5), (-2.0, -0.5),
           (0.5, 2.8), (2.0, 0.0), (-1.5, -2.5), (0.0, 0.5)]
for (cx, cy), col, lbl in zip(centers, colors9, drug_labels):
    xs = cx + 0.35 * np.random.randn(12)
    ys = cy + 0.35 * np.random.randn(12)
    ax3.scatter(xs, ys, c=[col], s=22, alpha=0.85, label=lbl)
ax3.legend(fontsize=5.5, ncol=3, loc='lower right',
           handlelength=0.8, labelspacing=0.3, columnspacing=0.5)
ax3.set_xlabel('Latent dim 1', fontsize=8)
ax3.set_ylabel('Latent dim 2', fontsize=8)
ax3.tick_params(labelsize=7)
ax3.text(0.03, 0.97, '9 drugs\n64-D fused', transform=ax3.transAxes,
         va='top', fontsize=7.5, color=C_DARK)

# ── Panel D: L1 classification (confusion-style bar) ─────────────────
ax4 = fig.add_subplot(gs[3])
panel_box(ax4, 'D  L1 Drug ID (100%)', C_L1)
drugs = ['CIM', 'FMD', 'GLD', 'GSR', 'IBU', 'MHE', 'MHL', 'MHR', 'HCT']
acc = [1.0] * 9
ax4.barh(drugs, acc, color=C_L1, alpha=0.8, edgecolor='white', height=0.6)
ax4.set_xlim(0, 1.12)
ax4.set_xlabel('Accuracy', fontsize=8)
ax4.tick_params(labelsize=7.5)
for i, v in enumerate(acc):
    ax4.text(v + 0.01, i, '100%', va='center', fontsize=7.5, fontweight='bold', color=C_DARK)
ax4.axvline(1.0, color=C_DARK, lw=1, ls='--', alpha=0.5)

# ── Panel E: L2 manufacturer ID (waterfall-style) ────────────────────
ax5 = fig.add_subplot(gs[4])
panel_box(ax5, 'E  L2 Manufacturer ID', C_L2)
drugs5  = ['CIM', 'FMD', 'GLD', 'GSR', 'IBU', 'MHE', 'MHL', 'MHR', 'HCT']
acc5    = [1.00, 1.00, 1.00, 1.00, 0.889, 1.00, 1.00, 0.917, 1.00]
bars = ax5.barh(drugs5, acc5, color=[C_L2 if a == 1.0 else '#E07B54' for a in acc5],
               alpha=0.82, edgecolor='white', height=0.6)
ax5.set_xlim(0, 1.18)
ax5.set_xlabel('Accuracy', fontsize=8)
ax5.tick_params(labelsize=7.5)
for i, v in enumerate(acc5):
    ax5.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=7.5,
             fontweight='bold', color=C_DARK)
ax5.axvline(0.9722, color=C_DARK, lw=1.5, ls='--', alpha=0.7)
ax5.text(0.9722, -0.7, 'Avg\n97.2%', fontsize=6.5, ha='center',
         color=C_DARK, fontweight='bold')

# ── Arrows between panels ────────────────────────────────────────────
for al, ar, lbl in [(ax1, ax2, ''), (ax2, ax3, 'Encode'), (ax3, ax4, 'Classify'), (ax4, ax5, 'Cascade')]:
    arrow_between(fig, al, ar, lbl)

# ── Title ────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         'PI-VAE: Physics-Informed VAE for Drug Spectral Analysis'
         '  |  L1: 100% Drug ID  •  L2: 97.2% Manufacturer ID  •  OOD AUC = 1.0',
         ha='center', va='top', fontsize=11, fontweight='bold', color=C_DARK)

plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
