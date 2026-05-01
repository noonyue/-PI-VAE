"""
Feature space evolution visualization and clustering metrics.

Outputs:
- figures/feature_space_evolution.png (t-SNE of Raw / PCA / PI-VAE)
- results/clustering_metrics.csv (Silhouette, DBI, CHI)
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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
    COLOR_TRUE, COLOR_PRED
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def tsne_embed(X, perplexity=30, seed=42):
    perplexity = min(perplexity, len(X) - 1)
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(X)


def clustering_metrics(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "DBI": davies_bouldin_score(X, labels),
        "CHI": calinski_harabasz_score(X, labels),
    }


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_raw, nir_raw, drug_labels, _ = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    # Encode drug labels for colors/metrics
    from sklearn.preprocessing import LabelEncoder

    drug_le = LabelEncoder()
    drug_y = drug_le.fit_transform(drug_labels)

    # Raw features (concat)
    X_raw = np.hstack([nir, uv])

    # PCA features (keep same dim as VAE latent: 64)
    pca = PCA(n_components=64, random_state=42)
    X_pca = pca.fit_transform(X_raw)

    # VAE features
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=64, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=64, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=32, n_peaks=10)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=32, n_peaks=10)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    uv_data_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=64, shuffle=False)
    nir_data_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=64, shuffle=False)
    z_uv = extract_latent_features(uv_vae, uv_data_loader, device)
    z_nir = extract_latent_features(nir_vae, nir_data_loader, device)
    X_vae = np.hstack([z_nir, z_uv])  # 64 dims

    # t-SNE embeddings
    X_raw_2d = tsne_embed(X_raw, perplexity=30)
    X_pca_2d = tsne_embed(X_pca, perplexity=30)
    X_vae_2d = tsne_embed(X_vae, perplexity=30)

    # Plot with reference style - 1×3 multi-panel layout
    fig, gs = create_multi_panel_figure(nrows=1, ncols=3, figsize=(15, 4.5), wspace=0.3)
    
    panel_labels = [("Raw", X_raw_2d, COLOR_TRUE), 
                    ("PCA", X_pca_2d, "#FFA500"), 
                    ("PI-VAE", X_vae_2d, COLOR_PRED)]
    
    for i, (name, X2d, color) in enumerate(panel_labels):
        ax = fig.add_subplot(gs[0, i])
        scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=drug_y, cmap="tab10", 
                            s=10, alpha=0.7, edgecolors='none')
        format_axes(ax, xlabel="Dim 1", ylabel="Dim 2", title=f"{name} t-SNE")
        add_panel_label(ax, f"({chr(97+i)})", x_offset=-0.12, y_offset=1.02)
    
    plt.savefig("figures/feature_space_evolution.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/feature_space_evolution.png")
    plt.close()

    # Clustering metrics
    rows = []
    for name, X in [("Raw", X_raw), ("PCA", X_pca), ("PI-VAE", X_vae)]:
        m = clustering_metrics(X, drug_y)
        rows.append({"Method": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv("results/clustering_metrics.csv", index=False)
    print("Saved: results/clustering_metrics.csv")


if __name__ == "__main__":
    main()

