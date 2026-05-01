"""
OOD detection via reconstruction error histogram.

Outputs:
- figures/ood_detection_histogram.png
- results/ood_performance_metrics.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

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
    COLOR_PRED, COLOR_NIR
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def prepare_data():
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")
    return uv, nir, drug_labels, manuf_labels


def train_vaes(uv, nir, device, latent_dim=16, n_peaks=8, epochs=60):
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=32, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=latent_dim, n_peaks=n_peaks)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=epochs, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=epochs, device=device, model_name="NIR-VAE")
    return uv_vae.to(device), nir_vae.to(device)


def reconstruction_error(model, data, device):
    loader = torch.utils.data.DataLoader(SpectralDataset(data), batch_size=64, shuffle=False)
    errs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            recon, _, _, _ = model(x)
            mse = torch.mean((recon - x) ** 2, dim=1)
            errs.append(mse.cpu().numpy())
    return np.concatenate(errs)


def simulate_ood(data, noise_scale=0.5, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data, axis=1)
    noise = rng.normal(0, noise_scale, size=data.shape)
    return perm + noise


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    uv, nir, _, _ = prepare_data()
    # Use NIR for OOD (can be swapped to UV similarly)
    nir_vae = train_vaes(uv, nir, device=device)[1]

    idx = np.arange(len(nir))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
    nir_test = nir[test_idx]

    in_err = reconstruction_error(nir_vae, nir_test, device)
    ood_data = simulate_ood(nir_test, noise_scale=0.8)
    ood_err = reconstruction_error(nir_vae, ood_data, device)

    labels = np.concatenate([np.zeros_like(in_err), np.ones_like(ood_err)])
    scores = np.concatenate([in_err, ood_err])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    youden_idx = np.argmax(tpr - fpr)
    best_thr = thr[youden_idx]
    best_fpr, best_tpr = fpr[youden_idx], tpr[youden_idx]

    # Histogram with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(7, 5))
    ax = fig.add_subplot(gs[0, 0])
    
    ax.hist(in_err, bins=40, alpha=0.6, label="In-distribution", 
           color=COLOR_PRED, density=True, edgecolor='black', linewidth=0.5)
    ax.hist(ood_err, bins=40, alpha=0.6, label="Simulated OOD", 
           color=COLOR_NIR, density=True, edgecolor='black', linewidth=0.5)
    ax.axvline(best_thr, color="black", linestyle="--", linewidth=2, 
              label=f"Threshold={best_thr:.4f}")
    format_axes(ax, xlabel="Reconstruction error (MSE)", ylabel="Density",
               title="OOD Detection by Reconstruction Error")
    ax.legend(fontsize=10, loc='upper right')
    add_panel_label(ax, "(a)", x_offset=-0.12, y_offset=1.02)
    
    plt.savefig("figures/ood_detection_histogram.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/ood_detection_histogram.png")
    plt.close()

    # Metrics table
    df = pd.DataFrame(
        [
            {
                "AUC": auc,
                "Best_Threshold": best_thr,
                "TPR_at_best": best_tpr,
                "FPR_at_best": best_fpr,
            }
        ]
    )
    df.to_csv("results/ood_performance_metrics.csv", index=False)
    print("Saved: results/ood_performance_metrics.csv")


if __name__ == "__main__":
    main()

