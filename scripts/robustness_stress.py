"""
Robustness stress test: evaluates manufacturer classification accuracy under
increasing noise levels (SNR 50→10 dB) for PI-VAE latent features vs. raw
spectra + SVM.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
    plot_with_confidence_interval, COLOR_PI_VAE, COLOR_BASELINE, COLOR_PRED
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def add_noise(x, snr_db, rng):
    """Additive white Gaussian noise to reach target SNR (dB)."""
    signal_power = np.mean(x ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def prepare_data():
    uv, nir, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv, method="snv")
    nir = preprocess_spectra(nir, method="snv")
    return uv, nir, drug_labels, manuf_labels


def train_models(uv, nir, manuf_labels, device):
    le = LabelEncoder()
    y = le.fit_transform(manuf_labels)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    uv_train, uv_test = uv[train_idx], uv[test_idx]
    nir_train, nir_test = nir[train_idx], nir[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train VAEs (shorter epochs for demo)
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv_train), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir_train), batch_size=32, shuffle=True)

    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    # Latent features
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

    # Raw spectrum baseline (concatenate uv+nir)
    X_train_raw = np.hstack([nir_train, uv_train])
    X_test_raw = np.hstack([nir_test, uv_test])
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test_raw = scaler.transform(X_test_raw)
    clf_raw = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf_raw.fit(X_train_raw, y_train)

    state = {
        "uv_vae": uv_vae.to(device),
        "nir_vae": nir_vae.to(device),
        "clf_latent": clf_latent,
        "clf_raw": clf_raw,
        "scaler_raw": scaler,
        "splits": (uv_test, nir_test, y_test),
        "le": le,
    }
    return state


def evaluate_noise_curve(state, device, snr_list=(50, 40, 30, 20, 10), seed=42):
    rng = np.random.default_rng(seed)
    uv_test, nir_test, y_test = state["splits"]
    results = []
    # Baseline latents on clean data for reference
    uv_vae = state["uv_vae"]
    nir_vae = state["nir_vae"]

    for snr in snr_list:
        uv_noisy = add_noise(uv_test, snr, rng)
        nir_noisy = add_noise(nir_test, snr, rng)

        # Latent pipeline
        uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv_noisy), batch_size=64, shuffle=False)
        nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir_noisy), batch_size=64, shuffle=False)
        z_uv = extract_latent_features(uv_vae, uv_loader, device)
        z_nir = extract_latent_features(nir_vae, nir_loader, device)
        X_latent = np.hstack([z_nir, z_uv])
        y_latent_pred = state["clf_latent"].predict(X_latent)
        acc_latent = accuracy_score(y_test, y_latent_pred)

        # Raw baseline
        X_raw = np.hstack([nir_noisy, uv_noisy])
        X_raw = state["scaler_raw"].transform(X_raw)
        y_raw_pred = state["clf_raw"].predict(X_raw)
        acc_raw = accuracy_score(y_test, y_raw_pred)

        results.append({"SNR_dB": snr, "PI-VAE": acc_latent, "Raw+SVM": acc_raw})
        print(f"SNR {snr} dB -> PI-VAE: {acc_latent:.4f}, Raw+SVM: {acc_raw:.4f}")
    return results


def plot_results(results, save_path):
    snrs = [r["SNR_dB"] for r in results]
    acc_pi = [r["PI-VAE"] for r in results]
    acc_raw = [r["Raw+SVM"] for r in results]

    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(7, 5))
    ax = fig.add_subplot(gs[0, 0])
    
    ax.plot(snrs, acc_pi, marker="o", label="PI-VAE (latent)", 
           color=COLOR_PI_VAE, linewidth=2, markersize=6)
    ax.plot(snrs, acc_raw, marker="s", label="Raw spectrum + SVM", 
           color=COLOR_BASELINE, linewidth=2, markersize=6)
    ax.invert_xaxis()  # higher SNR on left
    format_axes(ax, xlabel="SNR (dB)", ylabel="Manufacturer accuracy",
               title="Robustness Stress Test")
    ax.legend(fontsize=10, loc='best')
    add_panel_label(ax, "(a)", x_offset=-0.12, y_offset=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    uv, nir, _, manuf_labels = prepare_data()
    state = train_models(uv, nir, manuf_labels, device=device)
    results = evaluate_noise_curve(state, device=device)

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = "results/robustness_stress.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plot_results(results, "figures/robustness_stress.png")


if __name__ == "__main__":
    main()

