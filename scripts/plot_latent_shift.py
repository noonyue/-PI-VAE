"""
Latent disentanglement visualization for PI-VAE.

This script perturbs selected latent dimensions of trained UV/NIR VAEs and
plots how reconstructed spectra shift or broaden, helping interpret the
physical meaning of latent variables.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_vae_pipeline import (
    load_data,
    preprocess_spectra,
    SpectralDataset,
    UV_VAE,
    NIR_VAE,
    train_vae,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    COLOR_UV, COLOR_NIR, COLOR_PRED
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def prepare_data():
    uv_spectra, nir_spectra, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_spectra, method="snv")
    nir = preprocess_spectra(nir_spectra, method="snv")
    return uv, nir, drug_labels, manuf_labels


def train_vaes(uv, nir, latent_dim=16, n_peaks=8, epochs=80, device="cpu"):
    uv_ds = SpectralDataset(uv)
    nir_ds = SpectralDataset(nir)
    uv_loader = torch.utils.data.DataLoader(uv_ds, batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(nir_ds, batch_size=32, shuffle=True)

    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)

    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=epochs, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=epochs, device=device, model_name="NIR-VAE")
    return uv_vae, nir_vae


def perturb_and_reconstruct(model, sample, dim_indices, steps=(-3, -1.5, 0, 1.5, 3)):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.FloatTensor(sample[None, :]).to(device)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)
        mu = mu[0].cpu().numpy()
        std = std[0].cpu().numpy()

    reconstructions = {}
    for dim in dim_indices:
        z_list = []
        for s in steps:
            z = torch.FloatTensor(mu.copy()).to(device)
            z[dim] = mu[dim] + s * std[dim]
            with torch.no_grad():
                recon = model.decode(z.unsqueeze(0)).cpu().numpy().flatten()
            z_list.append((s, recon))
        reconstructions[dim] = z_list
    return reconstructions


def plot_reconstructions(recons, baseline, title, save_path):
    # Create figure with reference style - multi-panel layout
    n_panels = len(recons)
    fig, gs = create_multi_panel_figure(nrows=1, ncols=n_panels, figsize=(4*n_panels, 4), wspace=0.3)
    
    # Color scheme adapted to reference style
    colors = [COLOR_PRED, COLOR_NIR, "#2CA02C", "#9467BD", "#FF7F0E"]
    
    for idx, (dim, z_list) in enumerate(recons.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(baseline, color="black", lw=2, label="Baseline")
        for (s, recon), c in zip(z_list, colors[:len(z_list)]):
            ax.plot(recon, lw=1.5, color=c, alpha=0.8, label=f"z{dim} + {s:.1f}σ")
        label_letter = f"({chr(97+idx)})" if idx < 26 else f"({idx+1})"
        format_axes(ax, xlabel="Wavelength idx", ylabel="Intensity",
                   title=f"{title} | z{dim}")
        ax.legend(fontsize=9, loc='best')
        add_panel_label(ax, label_letter, x_offset=-0.12, y_offset=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    uv, nir, _, _ = prepare_data()
    uv_vae, nir_vae = train_vaes(uv, nir, latent_dim=16, n_peaks=8, epochs=60, device=device)

    sample_idx = 0
    uv_sample = uv[sample_idx]
    nir_sample = nir[sample_idx]

    uv_recons = perturb_and_reconstruct(uv_vae.to(device), uv_sample, dim_indices=[0, 1, 2])
    nir_recons = perturb_and_reconstruct(nir_vae.to(device), nir_sample, dim_indices=[0, 1, 2])

    # Baseline reconstructions
    with torch.no_grad():
        uv_base = uv_vae.decode(torch.FloatTensor(uv_vae.encode(torch.FloatTensor(uv_sample[None, :]).to(device))[0])).cpu().numpy().flatten()
        nir_base = nir_vae.decode(torch.FloatTensor(nir_vae.encode(torch.FloatTensor(nir_sample[None, :]).to(device))[0])).cpu().numpy().flatten()

    plot_reconstructions(uv_recons, uv_base, "UV-Vis latent perturb", "figures/latent_shift_uv.png")
    plot_reconstructions(nir_recons, nir_base, "NIR latent perturb", "figures/latent_shift_nir.png")


if __name__ == "__main__":
    main()

