"""
Plot VAE training curves (reconstruction + KL) for UV and NIR.
Output: figures/training_loss_curve.png
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_vae_pipeline import (
    load_data,
    preprocess_spectra,
    SpectralDataset,
    UV_VAE,
    NIR_VAE,
    vae_loss,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    plot_with_confidence_interval, COLOR_UV, COLOR_NIR, COLOR_PRED
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)


def train_with_logging(model, loader, epochs, device):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    history = {"loss": [], "recon": [], "kl": []}
    best = float("inf")
    patience, counter = 15, 0
    for _ in range(epochs):
        model.train()
        total, recon_t, kl_t = 0.0, 0.0, 0.0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item()
            recon_t += recon_loss.item()
            kl_t += kl_loss.item()
        n = len(loader.dataset)
        history["loss"].append(total / n)
        history["recon"].append(recon_t / n)
        history["kl"].append(kl_t / n)
        if history["loss"][-1] < best:
            best = history["loss"][-1]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return history


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_raw, nir_raw, _, _ = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    uv_loader = DataLoader(SpectralDataset(uv), batch_size=32, shuffle=True)
    nir_loader = DataLoader(SpectralDataset(nir), batch_size=32, shuffle=True)

    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)

    hist_uv = train_with_logging(uv_vae, uv_loader, epochs=120, device=device)
    hist_nir = train_with_logging(nir_vae, nir_loader, epochs=120, device=device)

    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=2, figsize=(14, 5), wspace=0.3)
    
    # Panel (a): UV-VAE training curves
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = np.arange(1, len(hist_uv["loss"]) + 1)
    ax1.plot(epochs, hist_uv["loss"], label="UV total", color=COLOR_UV, linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, hist_uv["recon"], label="UV recon", color=COLOR_UV, linestyle="--", linewidth=1.5)
    ax1.plot(epochs, hist_uv["kl"], label="UV KL", color=COLOR_UV, linestyle=":", linewidth=1.5)
    format_axes(ax1, xlabel="Epoch", ylabel="Loss per sample",
               title="UV-VAE Training Curves")
    ax1.legend(fontsize=10)
    add_panel_label(ax1, "(a)", x_offset=-0.12, y_offset=1.02)

    # Panel (b): NIR-VAE training curves
    ax2 = fig.add_subplot(gs[0, 1])
    epochs = np.arange(1, len(hist_nir["loss"]) + 1)
    ax2.plot(epochs, hist_nir["loss"], label="NIR total", color=COLOR_NIR, linewidth=2, marker='s', markersize=3)
    ax2.plot(epochs, hist_nir["recon"], label="NIR recon", color=COLOR_NIR, linestyle="--", linewidth=1.5)
    ax2.plot(epochs, hist_nir["kl"], label="NIR KL", color=COLOR_NIR, linestyle=":", linewidth=1.5)
    format_axes(ax2, xlabel="Epoch", ylabel="Loss per sample",
               title="NIR-VAE Training Curves")
    ax2.legend(fontsize=10)
    add_panel_label(ax2, "(b)", x_offset=-0.12, y_offset=1.02)
    
    plt.savefig("figures/training_loss_curve.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/training_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

