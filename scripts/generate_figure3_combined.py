"""
Generate Figure 3 (New): Physical Mechanism Validation
Spectral Reconstruction and Residual Statistics
Combines original Fig 16 (UV reconstruction) + Fig 17 (NIR reconstruction) + Fig 8 (residual analysis)

Layout: 3 rows × 2 columns (6 panels: A-F)
- Row 1: Residual analysis (A: Residual curves, B: RMSE distribution)
- Row 2: UV-Vis reconstruction (C: Original vs Reconstructed, D: Residual)
- Row 3: NIR reconstruction (E: Original vs Reconstructed, F: Residual)

Outputs:
- figures_new/figure3_physical_validation.png
- results_new/prior_fitting_stats.csv
"""
import os
import sys
import numpy as np
import pandas as pd
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
    COLOR_TRUE, COLOR_PRED, COLOR_NIR, setup_style
)
from output_path_helper import get_figures_dir, get_results_dir

# Setup style
setup_style()


def ensure_dirs():
    """Ensure output directories exist"""
    figures_dir = get_figures_dir()
    results_dir = get_results_dir()
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)


def train_models(nir, device, latent_dim=16, n_peaks=8, epochs=80):
    """Train both Lorentzian and Gaussian VAE models for NIR"""
    loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=32, shuffle=True)
    lorentz_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    gauss_vae = UV_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    lorentz_vae, _ = train_vae(lorentz_vae, loader, epochs=epochs, device=device, model_name="Lorentz-NIR")
    gauss_vae, _ = train_vae(gauss_vae, loader, epochs=epochs, device=device, model_name="Gaussian-NIR")
    return lorentz_vae.to(device), gauss_vae.to(device)


def reconstruction(model, data, device):
    """Reconstruct spectra using trained model"""
    loader = torch.utils.data.DataLoader(SpectralDataset(data), batch_size=64, shuffle=False)
    recons = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            rec, _, _, _ = model(x)
            recons.append(rec.cpu().numpy())
    return np.vstack(recons)


def rmse(a, b):
    """Calculate RMSE for each sample"""
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    uv_raw, nir_raw, drug_labels, _ = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    # ========== Row 1: Residual Analysis (Lorentzian vs Gaussian) ==========
    print("Training models for residual analysis...")
    lorentz_vae, gauss_vae = train_models(nir, device=device)
    rec_lorentz = reconstruction(lorentz_vae, nir, device)
    rec_gauss = reconstruction(gauss_vae, nir, device)

    resid_lorentz = nir - rec_lorentz
    resid_gauss = nir - rec_gauss
    rmse_l = rmse(nir, rec_lorentz)
    rmse_g = rmse(nir, rec_gauss)

    # ========== Row 2 & 3: UV-Vis and NIR Reconstruction ==========
    print("Training UV-VAE and NIR-VAE models...")
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=64, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=64, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=32, n_peaks=10)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=32, n_peaks=10)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    # Select representative samples for visualization
    # Use fixed indices for reproducibility (or select based on reconstruction quality)
    np.random.seed(42)
    uv_sample_idx = 45  # Fixed sample for UV
    nir_sample_idx = 11  # Fixed sample for NIR
    
    # Reconstruct selected samples
    uv_vae.eval()
    nir_vae.eval()
    with torch.no_grad():
        # UV-Vis reconstruction
        uv_x = torch.FloatTensor(uv[uv_sample_idx:uv_sample_idx+1]).to(device)
        uv_recon, _, _, _ = uv_vae(uv_x)
        uv_original = uv_x.cpu().numpy().flatten()
        uv_reconstructed = uv_recon.cpu().numpy().flatten()
        uv_residual = uv_original - uv_reconstructed
        
        # NIR reconstruction
        nir_x = torch.FloatTensor(nir[nir_sample_idx:nir_sample_idx+1]).to(device)
        nir_recon, _, _, _ = nir_vae(nir_x)
        nir_original = nir_x.cpu().numpy().flatten()
        nir_reconstructed = nir_recon.cpu().numpy().flatten()
        nir_residual = nir_original - nir_reconstructed

    # ========== Create combined figure ==========
    print("Creating combined figure...")
    
    # Create 3×2 grid (6 panels: A-F)
    # Layout:
    # [A: Residual curves]  [B: RMSE distribution]
    # [C: UV Original/Recon] [D: UV Residual]
    # [E: NIR Original/Recon] [F: NIR Residual]
    
    fig, gs = create_multi_panel_figure(nrows=3, ncols=2, figsize=(16, 14), 
                                        hspace=0.35, wspace=0.3)
    
    # ========== Row 1: Residual Analysis ==========
    # Panel A: Residual curves (Lorentzian vs Gaussian)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(resid_lorentz[0], label="Lorentzian residual", color=COLOR_PRED, linewidth=2)
    ax_a.plot(resid_gauss[0], label="Gaussian residual", color=COLOR_NIR, alpha=0.8, linewidth=2)
    ax_a.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    format_axes(ax_a, xlabel="Wavelength idx", ylabel="Residual",
               title="Residual (sample 0)")
    ax_a.legend(fontsize=10)
    add_panel_label(ax_a, "(A)", x_offset=-0.12, y_offset=1.02)
    
    # Panel B: RMSE distribution
    ax_b = fig.add_subplot(gs[0, 1])
    bp = ax_b.boxplot([rmse_l, rmse_g], labels=["Lorentzian", "Gaussian"], 
                     patch_artist=True, widths=0.6)
    # Color boxes
    bp['boxes'][0].set_facecolor(COLOR_PRED)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_NIR)
    bp['boxes'][1].set_alpha(0.7)
    format_axes(ax_b, ylabel="RMSE",
               title="RMSE distribution (all samples)")
    add_panel_label(ax_b, "(B)", x_offset=-0.12, y_offset=1.02)
    
    # ========== Row 2: UV-Vis Reconstruction ==========
    # Panel C: UV-Vis Original vs Reconstructed
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(uv_original, label='Original', color=COLOR_TRUE, alpha=0.8, linewidth=2)
    ax_c.plot(uv_reconstructed, label='Reconstructed', color=COLOR_PRED, 
             alpha=0.8, linestyle='--', linewidth=2)
    format_axes(ax_c, xlabel='Wavelength Index', ylabel='Intensity',
               title=f'Sample {uv_sample_idx} (Label: {drug_labels[uv_sample_idx]})')
    ax_c.legend(fontsize=10)
    add_panel_label(ax_c, "(C)", x_offset=-0.12, y_offset=1.02)
    
    # Panel D: UV-Vis Residual
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(uv_residual, color=COLOR_NIR, alpha=0.8, linewidth=2)
    ax_d.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    format_axes(ax_d, xlabel='Wavelength Index', ylabel='Residual',
               title='Residual (Noise)')
    add_panel_label(ax_d, "(D)", x_offset=-0.12, y_offset=1.02)
    
    # ========== Row 3: NIR Reconstruction ==========
    # Panel E: NIR Original vs Reconstructed
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.plot(nir_original, label='Original', color=COLOR_TRUE, alpha=0.8, linewidth=2)
    ax_e.plot(nir_reconstructed, label='Reconstructed', color=COLOR_PRED, 
             alpha=0.8, linestyle='--', linewidth=2)
    format_axes(ax_e, xlabel='Wavelength Index', ylabel='Intensity',
               title=f'Sample {nir_sample_idx} (Label: {drug_labels[nir_sample_idx]})')
    ax_e.legend(fontsize=10)
    add_panel_label(ax_e, "(E)", x_offset=-0.12, y_offset=1.02)
    
    # Panel F: NIR Residual
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.plot(nir_residual, color=COLOR_NIR, alpha=0.8, linewidth=2)
    ax_f.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    format_axes(ax_f, xlabel='Wavelength Index', ylabel='Residual',
               title='Residual (Noise)')
    add_panel_label(ax_f, "(F)", x_offset=-0.12, y_offset=1.02)
    
    # Save figure
    figures_dir = get_figures_dir()
    save_path = os.path.join(figures_dir, "figure3_physical_validation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Save statistics
    print("Saving statistics...")
    results_dir = get_results_dir()
    stats = pd.DataFrame(
        {
            "Model": ["Lorentzian", "Gaussian"],
            "RMSE_mean": [rmse_l.mean(), rmse_g.mean()],
            "RMSE_median": [np.median(rmse_l), np.median(rmse_g)],
            "RMSE_std": [rmse_l.std(), rmse_g.std()],
        }
    )
    stats_path = os.path.join(results_dir, "prior_fitting_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
