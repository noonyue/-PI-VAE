"""
Residual analysis for NIR: Lorentzian vs Gaussian decoders.

Outputs:
- figures/spectral_residual_analysis.png
- results/prior_fitting_stats.csv
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
    COLOR_TRUE, COLOR_PRED, COLOR_NIR
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def train_models(nir, device, latent_dim=16, n_peaks=8, epochs=80):
    loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=32, shuffle=True)
    lorentz_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    gauss_vae = UV_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    lorentz_vae, _ = train_vae(lorentz_vae, loader, epochs=epochs, device=device, model_name="Lorentz-NIR")
    gauss_vae, _ = train_vae(gauss_vae, loader, epochs=epochs, device=device, model_name="Gaussian-NIR")
    return lorentz_vae.to(device), gauss_vae.to(device)


def reconstruction(model, data, device):
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
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, nir_raw, _, _ = load_data()
    nir = preprocess_spectra(nir_raw, method="snv")

    lorentz_vae, gauss_vae = train_models(nir, device=device)
    rec_lorentz = reconstruction(lorentz_vae, nir, device)
    rec_gauss = reconstruction(gauss_vae, nir, device)

    resid_lorentz = nir - rec_lorentz
    resid_gauss = nir - rec_gauss
    rmse_l = rmse(nir, rec_lorentz)
    rmse_g = rmse(nir, rec_gauss)

    # Plot: residual curves (first sample) + RMSE boxplots with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=2, figsize=(12, 5), wspace=0.3)
    
    # Panel (a): Residual curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid_lorentz[0], label="Lorentzian residual", color=COLOR_PRED, linewidth=2)
    ax1.plot(resid_gauss[0], label="Gaussian residual", color=COLOR_NIR, alpha=0.8, linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    format_axes(ax1, xlabel="Wavelength idx", ylabel="Residual",
               title="Residual (sample 0)")
    ax1.legend(fontsize=10)
    add_panel_label(ax1, "(a)", x_offset=-0.12, y_offset=1.02)

    # Panel (b): RMSE distribution
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([rmse_l, rmse_g], labels=["Lorentzian", "Gaussian"], 
                     patch_artist=True, widths=0.6)
    # Color boxes
    bp['boxes'][0].set_facecolor(COLOR_PRED)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_NIR)
    bp['boxes'][1].set_alpha(0.7)
    format_axes(ax2, ylabel="RMSE",
               title="RMSE distribution (all samples)")
    add_panel_label(ax2, "(b)", x_offset=-0.12, y_offset=1.02)
    
    plt.savefig("figures/spectral_residual_analysis.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/spectral_residual_analysis.png")
    plt.close()

    stats = pd.DataFrame(
        {
            "Model": ["Lorentzian", "Gaussian"],
            "RMSE_mean": [rmse_l.mean(), rmse_g.mean()],
            "RMSE_median": [np.median(rmse_l), np.median(rmse_g)],
            "RMSE_std": [rmse_l.std(), rmse_g.std()],
        }
    )
    stats.to_csv("results/prior_fitting_stats.csv", index=False)
    print("Saved: results/prior_fitting_stats.csv")


if __name__ == "__main__":
    main()

