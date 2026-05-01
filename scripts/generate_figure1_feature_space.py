"""
Figure 1: Feature Space Evolution
Combined: t-SNE (Row 1) + PCA vs PI-VAE scatter (Row 2, 2x2)

Layout:
- Row 1 (3 panels): t-SNE of Raw / PCA / PI-VAE features
- Divider line with label
- Row 2 (2x2): Left col = PCA (UV, NIR), Right col = VAE (UV, NIR)
- Shared legend on the far right

Output: figures/figure1_feature_space_evolution.png
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi_vae_pipeline import (
    load_data, preprocess_spectra, SpectralDataset,
    UV_VAE, NIR_VAE, train_vae, extract_latent_features,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label, format_axes

setup_style()

DRUG_NAMES = ['CIM', 'FMD', 'GLD', 'GSR', 'HCT', 'IBU', 'MHE', 'MHL', 'MHR']
CMAP = 'tab10'
N_DRUGS = len(DRUG_NAMES)


def tsne_embed(X, perplexity=30, seed=42):
    perplexity = min(perplexity, len(X) - 1)
    return TSNE(n_components=2, random_state=seed,
                perplexity=perplexity).fit_transform(X)


def scatter_panel(ax, X2d, labels, title, panel_label):
    """Draw a scatter panel, return scatter object for shared legend."""
    sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=labels,
                    cmap=CMAP, vmin=0, vmax=N_DRUGS - 1,
                    s=16, alpha=0.78, edgecolors='none')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(length=3, width=0.8)
    add_panel_label(ax, panel_label, x_offset=-0.14, y_offset=1.04)
    return sc


def build_shared_legend(fig, x_pos=0.92):
    """Build shared drug legend on the right side of the figure."""
    cmap = plt.cm.tab10
    handles = [
        mlines.Line2D([], [], color=cmap(i / N_DRUGS), marker='o',
                      linestyle='None', markersize=7, label=name)
        for i, name in enumerate(DRUG_NAMES)
    ]
    fig.legend(handles=handles, title='Drug Type', title_fontsize=10,
               fontsize=9, loc='center right',
               bbox_to_anchor=(x_pos + 0.065, 0.50),
               framealpha=0.9, edgecolor='gray',
               markerscale=1.3, borderpad=0.8)


def main():
    os.makedirs('figures', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load & preprocess ──────────────────────────────────────────────────
    uv_raw, nir_raw, drug_labels, _ = load_data()
    uv  = preprocess_spectra(uv_raw, method='snv')
    nir = preprocess_spectra(nir_raw, method='snv')

    le = LabelEncoder()
    drug_y = le.fit_transform(drug_labels)

    # ── Feature extraction ─────────────────────────────────────────────────
    X_raw = np.hstack([nir, uv])

    X_pca64 = PCA(n_components=64, random_state=42).fit_transform(X_raw)

    pca2_uv  = PCA(n_components=2, random_state=42).fit_transform(uv)
    pca2_nir = PCA(n_components=2, random_state=42).fit_transform(nir)

    # Train VAE
    uv_loader  = torch.utils.data.DataLoader(
        SpectralDataset(uv),  batch_size=64, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(
        SpectralDataset(nir), batch_size=64, shuffle=True)

    uv_vae  = UV_VAE( input_dim=uv.shape[1],  latent_dim=32, n_peaks=10)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=32, n_peaks=10)
    uv_vae,  _ = train_vae(uv_vae,  uv_loader,  epochs=80,
                            device=device, model_name='UV-VAE')
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80,
                            device=device, model_name='NIR-VAE')

    uv_dl  = torch.utils.data.DataLoader(
        SpectralDataset(uv),  batch_size=64, shuffle=False)
    nir_dl = torch.utils.data.DataLoader(
        SpectralDataset(nir), batch_size=64, shuffle=False)

    z_uv  = extract_latent_features(uv_vae,  uv_dl,  device)
    z_nir = extract_latent_features(nir_vae, nir_dl, device)
    X_vae = np.hstack([z_nir, z_uv])

    vae2_uv  = PCA(n_components=2, random_state=42).fit_transform(z_uv)
    vae2_nir = PCA(n_components=2, random_state=42).fit_transform(z_nir)

    print('Computing t-SNE...')
    tsne_raw = tsne_embed(X_raw)
    tsne_pca = tsne_embed(X_pca64)
    tsne_vae = tsne_embed(X_vae)

    # ── Figure layout ──────────────────────────────────────────────────────
    # Height=14 for comfortable row spacing; right margin for legend
    fig = plt.figure(figsize=(18, 14))

    # Main title — anchored to very top, well above all subplots
    fig.text(0.46, 0.985,
             'Figure 1.  Feature Space Evolution: Raw \u2192 PCA \u2192 PI-VAE',
             ha='center', va='top', fontsize=14, fontweight='bold')

    # ── Row 1: t-SNE (top=0.95, bottom=0.54) ──
    gs_top = gridspec.GridSpec(
        1, 3, figure=fig,
        top=0.94, bottom=0.55,
        left=0.05, right=0.89,
        wspace=0.30)

    ax_tsne_raw = fig.add_subplot(gs_top[0])
    ax_tsne_pca = fig.add_subplot(gs_top[1])
    ax_tsne_vae = fig.add_subplot(gs_top[2])

    scatter_panel(ax_tsne_raw, tsne_raw, drug_y, 'Raw (SNV)  —  t-SNE',       '(a)')
    scatter_panel(ax_tsne_pca, tsne_pca, drug_y, 'PCA (64-dim)  —  t-SNE',    '(b)')
    scatter_panel(ax_tsne_vae, tsne_vae, drug_y, 'PI-VAE (64-dim)  —  t-SNE', '(c)')

    # ── Row 2: 2×2 grid (top=0.50, bottom=0.04) ──
    gs_bot = gridspec.GridSpec(
        2, 2, figure=fig,
        top=0.50, bottom=0.05,
        left=0.05, right=0.89,
        wspace=0.30, hspace=0.48)

    ax_pca_uv  = fig.add_subplot(gs_bot[0, 0])
    ax_pca_nir = fig.add_subplot(gs_bot[1, 0])
    ax_vae_uv  = fig.add_subplot(gs_bot[0, 1])
    ax_vae_nir = fig.add_subplot(gs_bot[1, 1])

    scatter_panel(ax_pca_uv,  pca2_uv,  drug_y, 'PCA  —  UV-Vis Latent',    '(d)')
    scatter_panel(ax_pca_nir, pca2_nir, drug_y, 'PCA  —  NIR Latent',       '(e)')
    scatter_panel(ax_vae_uv,  vae2_uv,  drug_y, 'PI-VAE  —  UV-Vis Latent', '(f)')
    scatter_panel(ax_vae_nir, vae2_nir, drug_y, 'PI-VAE  —  NIR Latent',    '(g)')

    # Column header labels removed — subplot titles already convey PCA vs PI-VAE

    # ── Shared legend (far right, vertically centered) ──
    build_shared_legend(fig, x_pos=0.90)

    out_path = 'figures/figure1_feature_space_evolution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
