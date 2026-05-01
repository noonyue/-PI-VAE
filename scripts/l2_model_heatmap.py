"""
L2 model performance heatmap and per-model accuracies.

Outputs:
- figures/l2_model_performance_heatmap.png
- results/l2_classification_results.csv (augmented with SVM_Acc/RF_Acc/PLS_Acc)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
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
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    get_heatmap_colormap
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


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


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data()
    uv = preprocess_spectra(uv_raw, method="snv")
    nir = preprocess_spectra(nir_raw, method="snv")

    # Encode labels
    drug_le = LabelEncoder()
    manuf_le = LabelEncoder()
    drug_y = drug_le.fit_transform(drug_labels)
    manuf_y = manuf_le.fit_transform(manuf_labels)

    # Train VAEs for latent features
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

    # For each drug, compute CV scores for SVM/RF/PLS on manufacturers
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
    df.to_csv("results/l2_model_performance.csv", index=False)
    print("Saved: results/l2_model_performance.csv")

    # Heatmap with reference style
    hm = np.array(heatmap_data)
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(6, 6))
    ax = fig.add_subplot(gs[0, 0])
    
    sns.heatmap(
        hm,
        annot=True,
        fmt=".2f",
        cmap=get_heatmap_colormap(),
        xticklabels=models,
        yticklabels=df["Drug"].tolist(),
        ax=ax,
        cbar_kws={'label': 'Accuracy'},
        linewidths=0.5
    )
    format_axes(ax, xlabel="Model", ylabel="Drug",
               title="L2 Model CV Accuracy (per drug)")
    
    plt.savefig("figures/l2_model_performance_heatmap.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/l2_model_performance_heatmap.png")
    plt.close()

    # Merge into existing l2_classification_results.csv if present
    l2_path = "results/l2_classification_results.csv"
    if os.path.exists(l2_path):
        l2_df = pd.read_csv(l2_path)
        # Convert numeric Drug to drug names if needed
        if l2_df["Drug"].dtype in [np.int64, np.int32, int]:
            drug_map = {i: name for i, name in enumerate(drug_le.classes_)}
            l2_df["Drug"] = l2_df["Drug"].map(drug_map)
        merged = l2_df.merge(df, on="Drug", how="left")
        merged.to_csv(l2_path, index=False)
        print("Updated:", l2_path)


if __name__ == "__main__":
    main()

