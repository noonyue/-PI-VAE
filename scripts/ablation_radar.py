"""
Ablation radar chart comparing:
- Standard AE (no physical prior)
- Gaussian VAE (both modalities use Gaussian peak decoder)
- PI-VAE (UV Gaussian + NIR Lorentzian)

Metrics: Accuracy, F1, convergence speed, low-sample stability, reconstruction quality (1-RMSE).
Outputs: figures/ablation_radar.png and results/ablation_radar.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

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
    COLOR_PRED, COLOR_DEFAULT, COLOR_OTHER, COLOR_TRUE
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


class StandardAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def encode(self, x):
        return self.encoder(x)


class GaussianOnlyVAE(UV_VAE):
    """
    Reuse UV_VAE architecture (Gaussian decoder) for NIR to approximate a full
    Gaussian-prior variant.
    """


def train_ae(model, loader, epochs=80, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()
    losses = []
    for _ in range(epochs):
        total = 0.0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            opt.zero_grad()
            recon, _ = model(x)
            loss = crit(recon, x)
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(loader.dataset))
    return model, losses


def reconstruction_rmse(model, data, device):
    loader = torch.utils.data.DataLoader(SpectralDataset(data), batch_size=64, shuffle=False)
    crit = torch.nn.MSELoss(reduction="mean")
    model.eval()
    rmses = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            recon = model(x)[0] if isinstance(model, StandardAE) else model(x)[0]
            mse = crit(recon, x).item()
            rmses.append(np.sqrt(mse))
    return float(np.mean(rmses))


def classifier_metrics(features_train, features_test, y_train, y_test):
    clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf.fit(features_train, y_train)
    y_pred = clf.predict(features_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return acc, f1


def low_sample_stability(feature_fn, y, repeats=3, train_frac=0.5, seed=42):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(repeats):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        train_idx, test_idx = idx[:cut], idx[cut:]
        X_train, X_test, y_train, y_test = feature_fn(train_idx, test_idx)
        acc, _ = classifier_metrics(X_train, X_test, y_train, y_test)
        accs.append(acc)
    std = np.std(accs)
    return 1.0 / (1.0 + std)  # higher = more stable


def normalize_scores(rows):
    # Per metric min-max to [0,1]
    metrics = ["Accuracy", "F1", "Speed", "Stability", "Recon"]
    vals = {m: [r[m] for r in rows] for m in metrics}
    norm = []
    for r in rows:
        entry = r.copy()
        for m in metrics:
            v = r[m]
            vmin, vmax = min(vals[m]), max(vals[m])
            entry[m] = 1.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        norm.append(entry)
    return norm


def plot_radar(rows, save_path):
    labels = ["Accuracy", "F1", "Speed", "Stability", "Recon"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Color mapping
    color_map = {
        "Standard AE": COLOR_TRUE,
        "Gaussian VAE": COLOR_OTHER,
        "PI-VAE (Ours)": COLOR_PRED
    }

    for r in rows:
        scores = [r[m] for m in labels]
        scores += scores[:1]
        color = color_map.get(r["Model"], COLOR_DEFAULT)
        ax.plot(angles, scores, label=r["Model"], linewidth=2, color=color)
        ax.fill(angles, scores, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("(A) Ablation Radar Chart", pad=20, fontsize=13, fontweight='bold')
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    uv_raw, nir_raw, _, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    le = LabelEncoder()
    y = le.fit_transform(manuf_labels)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]

    # Common scalers for raw baseline
    raw_scaler = StandardScaler()
    raw_train = np.hstack([nir[train_idx], uv[train_idx]])
    raw_test = np.hstack([nir[test_idx], uv[test_idx]])
    raw_train = raw_scaler.fit_transform(raw_train)
    raw_test = raw_scaler.transform(raw_test)

    results = []

    # ---- Standard AE ----
    ae_uv = StandardAE(uv.shape[1], latent_dim=16)
    ae_nir = StandardAE(nir.shape[1], latent_dim=16)
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv[train_idx]), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir[train_idx]), batch_size=32, shuffle=True)
    ae_uv, ae_uv_losses = train_ae(ae_uv, uv_loader, epochs=80, device=device)
    ae_nir, ae_nir_losses = train_ae(ae_nir, nir_loader, epochs=80, device=device)

    def ae_feats(train_ids, test_ids):
        uv_train_ds = torch.utils.data.DataLoader(SpectralDataset(uv[train_ids]), batch_size=64, shuffle=False)
        uv_test_ds = torch.utils.data.DataLoader(SpectralDataset(uv[test_ids]), batch_size=64, shuffle=False)
        nir_train_ds = torch.utils.data.DataLoader(SpectralDataset(nir[train_ids]), batch_size=64, shuffle=False)
        nir_test_ds = torch.utils.data.DataLoader(SpectralDataset(nir[test_ids]), batch_size=64, shuffle=False)
        def encode_all(model, loader):
            feats = []
            model.eval()
            with torch.no_grad():
                for b in loader:
                    x = b[0] if isinstance(b, (list, tuple)) else b
                    x = x.to(device)
                    z = model.encode(x).cpu().numpy()
                    feats.append(z)
            return np.vstack(feats)
        z_uv_tr = encode_all(ae_uv, uv_train_ds)
        z_uv_te = encode_all(ae_uv, uv_test_ds)
        z_nir_tr = encode_all(ae_nir, nir_train_ds)
        z_nir_te = encode_all(ae_nir, nir_test_ds)
        return np.hstack([z_nir_tr, z_uv_tr]), np.hstack([z_nir_te, z_uv_te]), y[train_ids], y[test_ids]

    ae_acc, ae_f1 = classifier_metrics(*ae_feats(train_idx, test_idx))
    ae_rmse = 0.5 * (reconstruction_rmse(ae_uv, uv[test_idx], device) + reconstruction_rmse(ae_nir, nir[test_idx], device))
    ae_speed = 1.0 / len(ae_uv_losses)
    ae_stab = low_sample_stability(ae_feats, y, repeats=3)
    results.append({"Model": "Standard AE", "Accuracy": ae_acc, "F1": ae_f1, "Speed": ae_speed, "Stability": ae_stab, "Recon": 1 - ae_rmse})

    # ---- Gaussian VAE (both Gaussian decoders) ----
    gauss_uv = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    gauss_nir = GaussianOnlyVAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    gauss_uv, gauss_uv_losses = train_vae(gauss_uv, uv_loader, epochs=80, device=device, model_name="Gaussian-UV")
    gauss_nir, gauss_nir_losses = train_vae(gauss_nir, nir_loader, epochs=80, device=device, model_name="Gaussian-NIR")

    def gauss_feats(train_ids, test_ids):
        uv_tr = torch.utils.data.DataLoader(SpectralDataset(uv[train_ids]), batch_size=64, shuffle=False)
        uv_te = torch.utils.data.DataLoader(SpectralDataset(uv[test_ids]), batch_size=64, shuffle=False)
        nir_tr = torch.utils.data.DataLoader(SpectralDataset(nir[train_ids]), batch_size=64, shuffle=False)
        nir_te = torch.utils.data.DataLoader(SpectralDataset(nir[test_ids]), batch_size=64, shuffle=False)
        z_uv_tr = extract_latent_features(gauss_uv, uv_tr, device)
        z_uv_te = extract_latent_features(gauss_uv, uv_te, device)
        z_nir_tr = extract_latent_features(gauss_nir, nir_tr, device)
        z_nir_te = extract_latent_features(gauss_nir, nir_te, device)
        return np.hstack([z_nir_tr, z_uv_tr]), np.hstack([z_nir_te, z_uv_te]), y[train_ids], y[test_ids]

    gauss_acc, gauss_f1 = classifier_metrics(*gauss_feats(train_idx, test_idx))
    gauss_rmse = 0.5 * (reconstruction_rmse(gauss_uv, uv[test_idx], device) + reconstruction_rmse(gauss_nir, nir[test_idx], device))
    gauss_speed = 1.0 / (len(gauss_uv_losses) + len(gauss_nir_losses))
    gauss_stab = low_sample_stability(gauss_feats, y, repeats=3)
    results.append({"Model": "Gaussian VAE", "Accuracy": gauss_acc, "F1": gauss_f1, "Speed": gauss_speed, "Stability": gauss_stab, "Recon": 1 - gauss_rmse})

    # ---- PI-VAE (hybrid) ----
    pi_uv = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    pi_nir = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    pi_uv, pi_uv_losses = train_vae(pi_uv, uv_loader, epochs=80, device=device, model_name="PI-UV")
    pi_nir, pi_nir_losses = train_vae(pi_nir, nir_loader, epochs=80, device=device, model_name="PI-NIR")

    def pi_feats(train_ids, test_ids):
        uv_tr = torch.utils.data.DataLoader(SpectralDataset(uv[train_ids]), batch_size=64, shuffle=False)
        uv_te = torch.utils.data.DataLoader(SpectralDataset(uv[test_ids]), batch_size=64, shuffle=False)
        nir_tr = torch.utils.data.DataLoader(SpectralDataset(nir[train_ids]), batch_size=64, shuffle=False)
        nir_te = torch.utils.data.DataLoader(SpectralDataset(nir[test_ids]), batch_size=64, shuffle=False)
        z_uv_tr = extract_latent_features(pi_uv, uv_tr, device)
        z_uv_te = extract_latent_features(pi_uv, uv_te, device)
        z_nir_tr = extract_latent_features(pi_nir, nir_tr, device)
        z_nir_te = extract_latent_features(pi_nir, nir_te, device)
        return np.hstack([z_nir_tr, z_uv_tr]), np.hstack([z_nir_te, z_uv_te]), y[train_ids], y[test_ids]

    pi_acc, pi_f1 = classifier_metrics(*pi_feats(train_idx, test_idx))
    pi_rmse = 0.5 * (reconstruction_rmse(pi_uv, uv[test_idx], device) + reconstruction_rmse(pi_nir, nir[test_idx], device))
    pi_speed = 1.0 / (len(pi_uv_losses) + len(pi_nir_losses))
    pi_stab = low_sample_stability(pi_feats, y, repeats=3)
    results.append({"Model": "PI-VAE (Ours)", "Accuracy": pi_acc, "F1": pi_f1, "Speed": pi_speed, "Stability": pi_stab, "Recon": 1 - pi_rmse})

    # Normalize for radar plot
    norm_rows = normalize_scores(results)
    plot_radar(norm_rows, "figures/ablation_radar.png")

    df = pd.DataFrame(results)
    df.to_csv("results/ablation_radar.csv", index=False)
    print("Saved: results/ablation_radar.csv")


if __name__ == "__main__":
    main()

