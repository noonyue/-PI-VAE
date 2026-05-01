"""
Generate Figure 5 (New): L2 Dynamic Model Selection Strategy Mechanism
Combines original Fig 6 (heatmap) + Fig 5 (decision boundary)

Layout: 1 row × 2 columns (2 panels: A-B)
- Panel A: L2 Model CV Accuracy (per drug) - Heatmap
- Panel B: L2 Decision Boundary (drug=MHR) - Decision boundary visualization

Outputs:
- figures_new/figure5_l2_selection_strategy.png
- results_new/l2_model_performance.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
    get_heatmap_colormap, setup_style
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


class PLSDAClassifier:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pls = None
        self.unique_classes = None

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        y_numeric = np.array([np.where(self.unique_classes == label)[0][0] for label in y])
        self.pls = PLSRegression(n_components=min(self.n_components, len(self.unique_classes) - 1))
        self.pls.fit(X, y_numeric)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X).flatten()
        return np.array([self.unique_classes[np.argmin(np.abs(self.unique_classes - val))] for val in y_pred_continuous])


def cv_scores(X, y, model_name):
    """Compute cross-validation scores for a model"""
    if len(X) <= 50:
        cv = LeaveOneOut()
    else:
        n_splits = min(5, len(np.unique(y)), len(X) // 2)
        if n_splits < 2:
            n_splits = 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if model_name == "SVM":
            m = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        elif model_name == "RF":
            m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            n_comp = min(5, len(np.unique(y_tr)) - 1)
            n_comp = max(n_comp, 1)
            m = PLSDAClassifier(n_components=n_comp)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    return float(np.mean(scores)) if scores else 0.0


def pick_hard_drug(drug_labels, manuf_labels):
    """Pick drug with max manufacturer count"""
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
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    # Encode labels
    drug_le = LabelEncoder()
    manuf_le = LabelEncoder()
    drug_y = drug_le.fit_transform(drug_labels)
    manuf_y = manuf_le.fit_transform(manuf_labels)

    # Train VAEs for latent features (used for both panels)
    print("Training VAE models...")
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=64, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=64, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=32, n_peaks=10)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=32, n_peaks=10)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    # Extract features
    uv_data_loader = torch.utils.data.DataLoader(SpectralDataset(uv), batch_size=64, shuffle=False)
    nir_data_loader = torch.utils.data.DataLoader(SpectralDataset(nir), batch_size=64, shuffle=False)
    z_uv = extract_latent_features(uv_vae, uv_data_loader, device)
    z_nir = extract_latent_features(nir_vae, nir_data_loader, device)
    X = np.hstack([z_nir, z_uv])

    # ========== Panel A: L2 Model Performance Heatmap ==========
    print("Computing L2 model performance (heatmap)...")
    models = ["SVM", "RF", "PLS"]
    records = []
    heatmap_data = []
    for d in np.unique(drug_y):
        mask = drug_y == d
        X_d = X[mask]
        y_d = manuf_y[mask]
        row = {"Drug": drug_le.classes_[d]}
        heat_row = []
        for m in models:
            score = cv_scores(X_d, y_d, m)
            row[f"{m}_Acc"] = score
            heat_row.append(score)
        records.append(row)
        heatmap_data.append(heat_row)

    df = pd.DataFrame(records)
    results_dir = get_results_dir()
    df.to_csv(os.path.join(results_dir, "l2_model_performance.csv"), index=False)
    print(f"Saved: {os.path.join(results_dir, 'l2_model_performance.csv')}")

    # ========== Panel B: L2 Decision Boundary ==========
    print("Computing L2 decision boundary...")
    target_drug = pick_hard_drug(drug_labels, manuf_labels)
    print(f"Selected drug for decision boundary: {target_drug}")
    
    mask = drug_labels == target_drug
    uv_drug = uv[mask]
    nir_drug = nir[mask]
    manuf_labels_drug = manuf_labels[mask]

    le = LabelEncoder()
    y = le.fit_transform(manuf_labels_drug)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    # Train drug-specific VAEs
    uv_loader_drug = torch.utils.data.DataLoader(SpectralDataset(uv_drug[train_idx]), batch_size=32, shuffle=True)
    nir_loader_drug = torch.utils.data.DataLoader(SpectralDataset(nir_drug[train_idx]), batch_size=32, shuffle=True)
    uv_vae_drug = UV_VAE(input_dim=uv_drug.shape[1], latent_dim=16, n_peaks=8)
    nir_vae_drug = NIR_VAE(input_dim=nir_drug.shape[1], latent_dim=16, n_peaks=8)
    uv_vae_drug, _ = train_vae(uv_vae_drug, uv_loader_drug, epochs=80, device=device, model_name="UV-VAE-drug")
    nir_vae_drug, _ = train_vae(nir_vae_drug, nir_loader_drug, epochs=80, device=device, model_name="NIR-VAE-drug")

    # Extract features for decision boundary
    uv_tr = torch.utils.data.DataLoader(SpectralDataset(uv_drug[train_idx]), batch_size=64, shuffle=False)
    uv_te = torch.utils.data.DataLoader(SpectralDataset(uv_drug[test_idx]), batch_size=64, shuffle=False)
    nir_tr = torch.utils.data.DataLoader(SpectralDataset(nir_drug[train_idx]), batch_size=64, shuffle=False)
    nir_te = torch.utils.data.DataLoader(SpectralDataset(nir_drug[test_idx]), batch_size=64, shuffle=False)

    z_uv_tr = extract_latent_features(uv_vae_drug, uv_tr, device)
    z_uv_te = extract_latent_features(uv_vae_drug, uv_te, device)
    z_nir_tr = extract_latent_features(nir_vae_drug, nir_tr, device)
    z_nir_te = extract_latent_features(nir_vae_drug, nir_te, device)

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

    # ========== Create combined figure ==========
    print("Creating combined figure...")
    
    # Create 1×2 grid (2 panels: A-B)
    fig, gs = create_multi_panel_figure(nrows=1, ncols=2, figsize=(16, 7), wspace=0.3)
    
    # Panel A: L2 Model Performance Heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    hm = np.array(heatmap_data)
    
    # Use consistent colormap for heatmap
    sns.heatmap(
        hm,
        annot=True,
        fmt=".2f",
        cmap=get_heatmap_colormap(),  # Use reference style colormap
        xticklabels=models,
        yticklabels=df["Drug"].tolist(),
        ax=ax_a,
        cbar_kws={'label': 'Accuracy'},
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0
    )
    format_axes(ax_a, xlabel="Model", ylabel="Drug",
               title="L2 Model CV Accuracy (per drug)")
    add_panel_label(ax_a, "(A)", x_offset=-0.12, y_offset=1.02)
    
    # Panel B: L2 Decision Boundary
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Use same colormap for decision boundary regions and scatter points
    # Use tab10 colormap for consistency with other figures
    contour = ax_b.contourf(xx, yy, Z, alpha=0.2, cmap="tab10", levels=len(np.unique(y_all)))
    scatter = ax_b.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, cmap="tab10", 
                          edgecolor="k", s=40, alpha=0.8, linewidths=0.5)
    format_axes(ax_b, xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
               title=f"L2 Decision Boundary (drug={target_drug})")
    add_panel_label(ax_b, "(B)", x_offset=-0.12, y_offset=1.02)
    
    # Add legend for manufacturers
    handles, labels = scatter.legend_elements()
    ax_b.legend(handles, [f"Manufacturer {i}" for i in range(len(handles))], 
               title="Manufacturer", fontsize=9, loc='upper right', framealpha=0.9)
    
    # Save figure
    figures_dir = get_figures_dir()
    save_path = os.path.join(figures_dir, "figure5_l2_selection_strategy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
