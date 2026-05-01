"""
Visualize L2 decision boundary on a hard drug subset using t-SNE + SVM.
Output: figures/l2_decision_boundary_zoom.png
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE

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
    COLOR_PRED
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)


def pick_hard_drug(drug_labels, manuf_labels):
    # Pick drug with max manufacturer count
    unique_drugs = np.unique(drug_labels)
    best = None
    best_count = -1
    for d in unique_drugs:
        cnt = len(np.unique(manuf_labels[drug_labels == d]))
        if cnt > best_count:
            best = d
            best_count = cnt
    return best


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    target_drug = pick_hard_drug(drug_labels, manuf_labels)
    mask = drug_labels == target_drug
    uv = uv[mask]
    nir = nir[mask]
    manuf_labels = manuf_labels[mask]

    le = LabelEncoder()
    y = le.fit_transform(manuf_labels)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv[train_idx]), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir[train_idx]), batch_size=32, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    uv_tr = torch.utils.data.DataLoader(SpectralDataset(uv[train_idx]), batch_size=64, shuffle=False)
    uv_te = torch.utils.data.DataLoader(SpectralDataset(uv[test_idx]), batch_size=64, shuffle=False)
    nir_tr = torch.utils.data.DataLoader(SpectralDataset(nir[train_idx]), batch_size=64, shuffle=False)
    nir_te = torch.utils.data.DataLoader(SpectralDataset(nir[test_idx]), batch_size=64, shuffle=False)

    z_uv_tr = extract_latent_features(uv_vae, uv_tr, device)
    z_uv_te = extract_latent_features(uv_vae, uv_te, device)
    z_nir_tr = extract_latent_features(nir_vae, nir_tr, device)
    z_nir_te = extract_latent_features(nir_vae, nir_te, device)

    X_train = np.hstack([z_nir_tr, z_uv_tr])
    X_test = np.hstack([z_nir_te, z_uv_te])
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit 2D embedding on all points
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1))
    X_2d = tsne.fit_transform(X_all)

    # Train SVM on 2D
    X_train_2d = X_2d[: len(X_train)]
    X_test_2d = X_2d[len(X_train) :]
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=False)
    clf.fit(X_train_2d, y_train)

    # Mesh for boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(8, 6))
    ax = fig.add_subplot(gs[0, 0])
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="tab10")
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, cmap="tab10", 
                        edgecolor="k", s=40, alpha=0.8)
    format_axes(ax, xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
               title=f"L2 Decision Boundary (drug={target_drug})")
    ax.legend(*scatter.legend_elements(), title="Manufacturer", 
             fontsize=10, loc='upper right')
    
    plt.savefig("figures/l2_decision_boundary_zoom.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/l2_decision_boundary_zoom.png")
    plt.close()


if __name__ == "__main__":
    main()

