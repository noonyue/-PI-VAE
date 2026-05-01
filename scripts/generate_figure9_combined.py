"""
Generate Figure 9 (New): Physical Interpretability Analysis
Combines original Fig 13 (UV latent shift) + Fig 14 (NIR latent shift)

Layout: 2 rows × 3 columns (6 panels)
- Row 1: UV-Vis latent perturbation (z0, z1, z2)
- Row 2: NIR latent perturbation (z0, z1, z2)

Each panel shows how perturbing a specific latent dimension affects the reconstructed spectrum.

Outputs:
- figures_new/figure9_physical_interpretability.png
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
    COLOR_PRED, COLOR_NIR, COLOR_TRUE, setup_style
)
from output_path_helper import get_figures_dir

# Setup style
setup_style()


def ensure_dirs():
    """Ensure output directories exist"""
    figures_dir = get_figures_dir()
    os.makedirs(figures_dir, exist_ok=True)


def train_vaes(uv, nir, latent_dim=16, n_peaks=8, epochs=80, device="cpu"):
    """Train UV and NIR VAE models"""
    uv_ds = SpectralDataset(uv)
    nir_ds = SpectralDataset(nir)
    uv_loader = torch.utils.data.DataLoader(uv_ds, batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(nir_ds, batch_size=32, shuffle=True)

    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)

    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=epochs, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=epochs, device=device, model_name="NIR-VAE")
    return uv_vae.to(device), nir_vae.to(device)


def perturb_and_reconstruct(model, sample, dim_indices, steps=(-3.0, -1.5, 0.0, 1.5, 3.0)):
    """Perturb latent dimensions and reconstruct spectra"""
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
    return reconstructions, mu, std


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    uv_spectra, nir_spectra, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_spectra, method="snv")
    nir = preprocess_spectra(nir_spectra, method="snv")

    # Train VAEs
    print("Training VAE models...")
    uv_vae, nir_vae = train_vaes(uv, nir, latent_dim=16, n_peaks=8, epochs=60, device=device)

    # Select a representative sample
    sample_idx = 0
    uv_sample = uv[sample_idx]
    nir_sample = nir[sample_idx]

    # Perturb latent dimensions
    print("Computing latent perturbations...")
    uv_recons, uv_mu, uv_std = perturb_and_reconstruct(uv_vae, uv_sample, dim_indices=[0, 1, 2])
    nir_recons, nir_mu, nir_std = perturb_and_reconstruct(nir_vae, nir_sample, dim_indices=[0, 1, 2])

    # Get baseline reconstructions
    with torch.no_grad():
        uv_base_z = uv_vae.encode(torch.FloatTensor(uv_sample[None, :]).to(device))[0]
        uv_base = uv_vae.decode(uv_base_z).cpu().numpy().flatten()
        nir_base_z = nir_vae.encode(torch.FloatTensor(nir_sample[None, :]).to(device))[0]
        nir_base = nir_vae.decode(nir_base_z).cpu().numpy().flatten()

    # ========== Create combined figure ==========
    print("Creating combined figure...")
    
    # Create 2×3 grid (6 panels)
    # Row 1: UV-Vis (z0, z1, z2)
    # Row 2: NIR (z0, z1, z2)
    fig, gs = create_multi_panel_figure(nrows=2, ncols=3, figsize=(18, 10), 
                                        hspace=0.35, wspace=0.3)
    
    # Color scheme for perturbation steps
    step_colors = {
        -3.0: '#1f77b4',  # Blue
        -1.5: '#7fb3d3',  # Light blue
        0.0: '#2ca02c',   # Green (baseline, but we'll use black)
        1.5: '#ff7f0e',   # Orange
        3.0: '#d62728'    # Red
    }
    
    # Row 1: UV-Vis latent perturbations
    for col, (dim, z_list) in enumerate(uv_recons.items()):
        ax = fig.add_subplot(gs[0, col])
        
        # Plot baseline
        ax.plot(uv_base, color="black", lw=2.5, label="baseline", zorder=10)
        
        # Plot perturbed reconstructions
        for s, recon in z_list:
            if s == 0.0:
                continue  # Skip baseline (already plotted)
            color = step_colors.get(s, '#808080')
            ax.plot(recon, lw=2, color=color, alpha=0.8, 
                   label=f"{s:+.1f}σ", zorder=5)
        
        format_axes(ax, xlabel="Wavelength idx", ylabel="Intensity",
                   title=f"UV-Vis latent perturb | z{dim}")
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        add_panel_label(ax, f"({chr(65+col)})", x_offset=-0.12, y_offset=1.02)
    
    # Row 2: NIR latent perturbations
    for col, (dim, z_list) in enumerate(nir_recons.items()):
        ax = fig.add_subplot(gs[1, col])
        
        # Plot baseline
        ax.plot(nir_base, color="black", lw=2.5, label="baseline", zorder=10)
        
        # Plot perturbed reconstructions
        for s, recon in z_list:
            if s == 0.0:
                continue  # Skip baseline (already plotted)
            color = step_colors.get(s, '#808080')
            ax.plot(recon, lw=2, color=color, alpha=0.8, 
                   label=f"{s:+.1f}σ", zorder=5)
        
        format_axes(ax, xlabel="Wavelength idx", ylabel="Intensity",
                   title=f"NIR latent perturb | z{dim}")
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        add_panel_label(ax, f"({chr(68+col)})", x_offset=-0.12, y_offset=1.02)
    
    # Save figure
    figures_dir = get_figures_dir()
    save_path = os.path.join(figures_dir, "figure9_physical_interpretability.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
