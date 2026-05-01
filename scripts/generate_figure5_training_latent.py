"""
Figure 5: Training Dynamics & Latent Space Stability
Combined: UV/NIR training curves (top) + UV/NIR latent perturbation (bottom)

Layout: 2x2
- (a) UV-VAE training loss curves  (b) NIR-VAE training loss curves
- (c) UV latent perturbation       (d) NIR latent perturbation

Output: figures/figure5_training_latent.png
"""

import os
import sys
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi_vae_pipeline import (
    load_data, preprocess_spectra, SpectralDataset,
    UV_VAE, NIR_VAE, train_vae, vae_loss,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label, format_axes, COLOR_UV, COLOR_NIR, COLOR_PRED

setup_style()

COLOR_STEPS = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']


# ── Training with full history logging ────────────────────────────────────────
def train_with_logging(model, loader, epochs, device):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    history = {'total': [], 'recon': [], 'kl': []}
    best, patience, counter = float('inf'), 20, 0
    for ep in range(epochs):
        model.train()
        tot, rec, kl = 0., 0., 0.
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = model(x)
            loss, rl, kll = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            tot += loss.item()
            rec += rl.item()
            kl  += kll.item()
        n = len(loader.dataset)
        history['total'].append(tot / n)
        history['recon'].append(rec / n)
        history['kl'].append(kl  / n)
        if history['total'][-1] < best:
            best, counter = history['total'][-1], 0
        else:
            counter += 1
            if counter >= patience:
                print(f'  Early stop at epoch {ep+1}')
                break
    return model, history


# ── Latent perturbation ───────────────────────────────────────────────────────
def perturb_reconstruct(model, sample, dim, steps, device):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(sample[None, :]).to(device)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)[0].cpu().numpy()
        mu  = mu[0].cpu().numpy()

    recons = []
    for s in steps:
        z = torch.FloatTensor(mu.copy()).to(device)
        z[dim] = float(mu[dim] + s * std[dim])
        with torch.no_grad():
            rec = model.decode(z.unsqueeze(0)).cpu().numpy().flatten()
        recons.append((s, rec))
    return recons, mu, std


def draw_training_panel(ax, history, color, title, panel_label):
    ep = np.arange(1, len(history['total']) + 1)
    ax.plot(ep, history['total'], color=color,     lw=2.0, label='Total loss')
    ax.plot(ep, history['recon'], color=color,     lw=1.5, linestyle='--',
            alpha=0.75, label='Reconstruction')
    ax.plot(ep, history['kl'],   color='#888888',  lw=1.5, linestyle=':',
            alpha=0.85, label='KL divergence')
    # Annotate final values
    ax.annotate(f"Final: {history['total'][-1]:.2f}",
                xy=(ep[-1], history['total'][-1]),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1))
    format_axes(ax, xlabel='Epoch', ylabel='Loss / sample', title=title)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(length=3)
    add_panel_label(ax, panel_label, x_offset=-0.13, y_offset=1.04)


def draw_latent_panel(ax, baseline, recon_list, modality, dim, panel_label):
    x = np.arange(len(baseline))
    ax.plot(x, baseline, color='black', lw=2.0, label='Baseline (z=μ)', zorder=5)
    for (s, rec), col in zip(recon_list, COLOR_STEPS):
        sign = '+' if s >= 0 else ''
        ax.plot(x, rec, color=col, lw=1.4, alpha=0.82,
                linestyle='--', label=f'z[{dim}] {sign}{s:.1f}σ')
    ax.set_title(f'{modality} Latent Perturbation  (dim {dim})',
                 fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel('Wavelength Index', fontsize=9)
    ax.set_ylabel('Intensity (SNV)', fontsize=9)
    ax.legend(fontsize=7.5, loc='best', ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(length=3)
    add_panel_label(ax, panel_label, x_offset=-0.13, y_offset=1.04)


def main():
    os.makedirs('figures', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    uv_raw, nir_raw, drug_labels, _ = load_data()
    uv  = preprocess_spectra(uv_raw,  method='snv')
    nir = preprocess_spectra(nir_raw, method='snv')

    uv_loader  = torch.utils.data.DataLoader(
        SpectralDataset(uv),  batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(
        SpectralDataset(nir), batch_size=32, shuffle=True)

    print('Training UV-VAE...')
    uv_vae, hist_uv = train_with_logging(
        UV_VAE(input_dim=uv.shape[1],  latent_dim=16, n_peaks=8),
        uv_loader, epochs=150, device=device)

    print('Training NIR-VAE...')
    nir_vae, hist_nir = train_with_logging(
        NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8),
        nir_loader, epochs=150, device=device)

    # Pick a median-quality sample for perturbation
    sample_idx = len(uv) // 3
    uv_sample  = uv[sample_idx]
    nir_sample = nir[sample_idx]

    steps = [-2, -1, 0, 1, 2]
    PERTURB_DIM = 0   # most informative dim by convention

    # baseline reconstructions
    uv_vae.eval()
    nir_vae.eval()
    with torch.no_grad():
        uv_base = uv_vae.decode(
            uv_vae.encode(torch.FloatTensor(uv_sample[None, :]).to(device))[0]
        ).cpu().numpy().flatten()
        nir_base = nir_vae.decode(
            nir_vae.encode(torch.FloatTensor(nir_sample[None, :]).to(device))[0]
        ).cpu().numpy().flatten()

    uv_recons,  _, _ = perturb_reconstruct(uv_vae.to(device),  uv_sample,  PERTURB_DIM, steps, device)
    nir_recons, _, _ = perturb_reconstruct(nir_vae.to(device), nir_sample, PERTURB_DIM, steps, device)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.50, 0.995,
             'Figure 5.  Training Dynamics & Latent Space Stability',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.93, bottom=0.08,
                           left=0.07, right=0.97,
                           hspace=0.32, wspace=0.28)

    ax_uv_train  = fig.add_subplot(gs[0, 0])
    ax_nir_train = fig.add_subplot(gs[0, 1])
    ax_uv_lat    = fig.add_subplot(gs[1, 0])
    ax_nir_lat   = fig.add_subplot(gs[1, 1])

    draw_training_panel(ax_uv_train,  hist_uv,  COLOR_UV,  'UV-VAE Training Curve',  '(a)')
    draw_training_panel(ax_nir_train, hist_nir, COLOR_NIR, 'NIR-VAE Training Curve', '(b)')
    draw_latent_panel(ax_uv_lat,  uv_base,  uv_recons,  'UV-Vis', PERTURB_DIM, '(c)')
    draw_latent_panel(ax_nir_lat, nir_base, nir_recons, 'NIR',    PERTURB_DIM, '(d)')

    out_path = 'figures/figure5_training_latent.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
