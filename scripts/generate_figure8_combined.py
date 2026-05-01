"""
Generate Figure 8 (New): Robustness and Safety Assessment
Combines original Fig 11 (robustness stress test) + Fig 12 (OOD detection)

Layout: 1 row × 2 columns (2 panels: A-B)
- Panel A: Robustness Stress Test (SNR vs Accuracy)
- Panel B: OOD Detection by Reconstruction Error (Histogram)

Outputs:
- figures_new/figure8_robustness_safety.png
- results_new/robustness_stress.csv
- results_new/ood_performance_metrics.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_vae_pipeline import (
    load_data,
    preprocess_spectra,
    SpectralDataset,
    UV_VAE,
    NIR_VAE,
    train_vae,
    extract_latent_features,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    COLOR_PRED, COLOR_NIR, COLOR_PI_VAE, COLOR_BASELINE, setup_style
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


def add_noise(x, snr_db, rng):
    """Additive white Gaussian noise to reach target SNR (dB)."""
    signal_power = np.mean(x ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def reconstruction_error(model, data, device):
    """Compute reconstruction error (MSE) for each sample"""
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
    """Simulate out-of-distribution data"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data, axis=1)
    noise = rng.normal(0, noise_scale, size=data.shape)
    return perm + noise


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    # ========== Panel A: Robustness Stress Test ==========
    print("Training models for robustness stress test...")
    le = LabelEncoder()
    y = le.fit_transform(manuf_labels)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    uv_train, uv_test = uv[train_idx], uv[test_idx]
    nir_train, nir_test = nir[train_idx], nir[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train VAEs
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv_train), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir_train), batch_size=32, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    # Extract latent features
    uv_train_loader = torch.utils.data.DataLoader(SpectralDataset(uv_train), batch_size=64, shuffle=False)
    uv_test_loader = torch.utils.data.DataLoader(SpectralDataset(uv_test), batch_size=64, shuffle=False)
    nir_train_loader = torch.utils.data.DataLoader(SpectralDataset(nir_train), batch_size=64, shuffle=False)
    nir_test_loader = torch.utils.data.DataLoader(SpectralDataset(nir_test), batch_size=64, shuffle=False)

    z_uv_train = extract_latent_features(uv_vae, uv_train_loader, device)
    z_uv_test = extract_latent_features(uv_vae, uv_test_loader, device)
    z_nir_train = extract_latent_features(nir_vae, nir_train_loader, device)
    z_nir_test = extract_latent_features(nir_vae, nir_test_loader, device)

    X_train_latent = np.hstack([z_nir_train, z_uv_train])
    X_test_latent = np.hstack([z_nir_test, z_uv_test])

    clf_latent = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf_latent.fit(X_train_latent, y_train)

    # Raw spectrum baseline
    X_train_raw = np.hstack([nir_train, uv_train])
    X_test_raw = np.hstack([nir_test, uv_test])
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test_raw = scaler.transform(X_test_raw)
    clf_raw = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf_raw.fit(X_train_raw, y_train)

    # Evaluate at different SNR levels
    print("Evaluating robustness at different SNR levels...")
    snr_list = [50, 40, 30, 20, 10]
    rng = np.random.default_rng(42)
    robustness_results = []
    
    for snr in snr_list:
        uv_noisy = add_noise(uv_test, snr, rng)
        nir_noisy = add_noise(nir_test, snr, rng)

        # Latent pipeline
        uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv_noisy), batch_size=64, shuffle=False)
        nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir_noisy), batch_size=64, shuffle=False)
        z_uv = extract_latent_features(uv_vae, uv_loader, device)
        z_nir = extract_latent_features(nir_vae, nir_loader, device)
        X_latent = np.hstack([z_nir, z_uv])
        y_latent_pred = clf_latent.predict(X_latent)
        acc_latent = accuracy_score(y_test, y_latent_pred)

        # Raw baseline
        X_raw = np.hstack([nir_noisy, uv_noisy])
        X_raw = scaler.transform(X_raw)
        y_raw_pred = clf_raw.predict(X_raw)
        acc_raw = accuracy_score(y_test, y_raw_pred)

        robustness_results.append({"SNR_dB": snr, "PI-VAE": acc_latent, "Raw+SVM": acc_raw})
        print(f"SNR {snr} dB -> PI-VAE: {acc_latent:.4f}, Raw+SVM: {acc_raw:.4f}")

    # ========== Panel B: OOD Detection ==========
    print("Computing OOD detection...")
    # Use NIR VAE for OOD detection
    nir_test_ood = nir[test_idx]
    
    in_err = reconstruction_error(nir_vae, nir_test_ood, device)
    ood_data = simulate_ood(nir_test_ood, noise_scale=0.8, seed=42)
    ood_err = reconstruction_error(nir_vae, ood_data, device)

    # Compute OOD metrics
    labels = np.concatenate([np.zeros_like(in_err), np.ones_like(ood_err)])
    scores = np.concatenate([in_err, ood_err])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    youden_idx = np.argmax(tpr - fpr)
    best_thr = thr[youden_idx]
    best_fpr, best_tpr = fpr[youden_idx], tpr[youden_idx]

    # ========== Create combined figure ==========
    print("Creating combined figure...")
    
    # Create 1×2 grid (2 panels: A-B)
    fig, gs = create_multi_panel_figure(nrows=1, ncols=2, figsize=(16, 6), wspace=0.3)
    
    # Panel A: Robustness Stress Test
    ax_a = fig.add_subplot(gs[0, 0])
    snrs = [r["SNR_dB"] for r in robustness_results]
    acc_pi = [r["PI-VAE"] for r in robustness_results]
    acc_raw = [r["Raw+SVM"] for r in robustness_results]
    
    ax_a.plot(snrs, acc_pi, marker="o", label="PI-VAE (latent)", 
             color=COLOR_PI_VAE, linewidth=2.5, markersize=8, 
             markerfacecolor=COLOR_PI_VAE, markeredgecolor='black', markeredgewidth=1)
    ax_a.plot(snrs, acc_raw, marker="s", label="Raw spectrum + SVM", 
             color=COLOR_BASELINE, linewidth=2.5, markersize=8,
             markerfacecolor=COLOR_BASELINE, markeredgecolor='black', markeredgewidth=1)
    ax_a.invert_xaxis()  # Higher SNR on left
    format_axes(ax_a, xlabel="SNR (dB)", ylabel="Manufacturer accuracy",
               title="Robustness Stress Test")
    ax_a.set_ylim([0.0, 1.05])
    ax_a.legend(fontsize=10, loc='best', framealpha=0.9)
    add_panel_label(ax_a, "(A)", x_offset=-0.12, y_offset=1.02)
    
    # Panel B: OOD Detection Histogram
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Create histogram with proper bins
    bins = np.linspace(0, 3.0, 50)
    ax_b.hist(in_err, bins=bins, alpha=0.7, label="In-distribution", 
             color=COLOR_PRED, density=True, edgecolor='black', linewidth=0.8)
    ax_b.hist(ood_err, bins=bins, alpha=0.7, label="Simulated OOD", 
             color=COLOR_NIR, density=True, edgecolor='black', linewidth=0.8)
    ax_b.axvline(best_thr, color="black", linestyle="--", linewidth=2.5, 
                label=f"Threshold={best_thr:.4f}")
    format_axes(ax_b, xlabel="Reconstruction error (MSE)", ylabel="Density",
               title="OOD Detection by Reconstruction Error")
    ax_b.set_xlim([0, 3.0])
    ax_b.legend(fontsize=10, loc='upper right', framealpha=0.9)
    add_panel_label(ax_b, "(B)", x_offset=-0.12, y_offset=1.02)
    
    # Save figure
    figures_dir = get_figures_dir()
    save_path = os.path.join(figures_dir, "figure8_robustness_safety.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Save results
    results_dir = get_results_dir()
    
    # Robustness results
    robustness_df = pd.DataFrame(robustness_results)
    robustness_path = os.path.join(results_dir, "robustness_stress.csv")
    robustness_df.to_csv(robustness_path, index=False)
    print(f"Saved: {robustness_path}")
    
    # OOD metrics
    ood_df = pd.DataFrame([{
        "AUC": auc,
        "Best_Threshold": best_thr,
        "TPR_at_best": best_tpr,
        "FPR_at_best": best_fpr,
    }])
    ood_path = os.path.join(results_dir, "ood_performance_metrics.csv")
    ood_df.to_csv(ood_path, index=False)
    print(f"Saved: {ood_path}")


if __name__ == "__main__":
    main()
