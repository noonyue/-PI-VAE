"""
Sample-size sensitivity analysis.

Groups manufacturers by sample size:
- Rich: >15
- Medium: 10-15
- Low: <10

Compares mean per-class accuracy of:
1) End-to-end 1D CNN on raw spectra
2) PI-VAE latent + SVM (cascade-style)
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, fusion, labels):
        self.x = torch.FloatTensor(fusion)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Simple1DCNN(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out = input_len // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        x = self.net(x)
        return self.head(x)


def train_cnn(model, train_loader, val_loader, epochs=50, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())
    acc = accuracy_score(all_true, all_preds)
    return model, acc, np.array(all_true), np.array(all_preds)


def per_class_accuracy(y_true, y_pred):
    accs = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        accs[int(cls)] = accuracy_score(y_true[mask], y_pred[mask])
    return accs


def group_by_size(label_encoder, y_true, accs, manuf_labels):
    counts = pd.Series(manuf_labels).value_counts()
    inv_labels = {i: lbl for i, lbl in enumerate(label_encoder.classes_)}

    groups = {"rich": [], "medium": [], "low": []}
    for cls, acc in accs.items():
        manuf = inv_labels[cls]
        n = counts[manuf]
        if n > 15:
            groups["rich"].append(acc)
        elif n >= 10:
            groups["medium"].append(acc)
        else:
            groups["low"].append(acc)
    summary = {}
    for g, vals in groups.items():
        summary[g] = np.mean(vals) if len(vals) else np.nan
    summary["counts"] = {g: len(groups[g]) for g in groups}
    return summary


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

    # ---------- PI-VAE latent + SVM ----------
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv[train_idx]), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir[train_idx]), batch_size=32, shuffle=True)
    uv_vae = UV_VAE(input_dim=uv.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir.shape[1], latent_dim=16, n_peaks=8)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=80, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=80, device=device, model_name="NIR-VAE")

    uv_train_loader = torch.utils.data.DataLoader(SpectralDataset(uv[train_idx]), batch_size=64, shuffle=False)
    uv_test_loader = torch.utils.data.DataLoader(SpectralDataset(uv[test_idx]), batch_size=64, shuffle=False)
    nir_train_loader = torch.utils.data.DataLoader(SpectralDataset(nir[train_idx]), batch_size=64, shuffle=False)
    nir_test_loader = torch.utils.data.DataLoader(SpectralDataset(nir[test_idx]), batch_size=64, shuffle=False)

    z_uv_train = extract_latent_features(uv_vae, uv_train_loader, device)
    z_uv_test = extract_latent_features(uv_vae, uv_test_loader, device)
    z_nir_train = extract_latent_features(nir_vae, nir_train_loader, device)
    z_nir_test = extract_latent_features(nir_vae, nir_test_loader, device)

    X_train_latent = np.hstack([z_nir_train, z_uv_train])
    X_test_latent = np.hstack([z_nir_test, z_uv_test])

    clf_latent = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf_latent.fit(X_train_latent, y_train)
    y_pred_latent = clf_latent.predict(X_test_latent)
    latent_accs = per_class_accuracy(y_test, y_pred_latent)

    # ---------- CNN end-to-end ----------
    fusion_train = np.hstack([nir[train_idx], uv[train_idx]])
    fusion_test = np.hstack([nir[test_idx], uv[test_idx]])
    scaler = StandardScaler()
    fusion_train = scaler.fit_transform(fusion_train)
    fusion_test = scaler.transform(fusion_test)

    cnn_ds_train = FusionDataset(fusion_train, y_train)
    cnn_ds_test = FusionDataset(fusion_test, y_test)
    train_loader = torch.utils.data.DataLoader(cnn_ds_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(cnn_ds_test, batch_size=64, shuffle=False)

    cnn = Simple1DCNN(input_len=fusion_train.shape[1], n_classes=len(le.classes_))
    cnn, cnn_acc, y_true_cnn, y_pred_cnn = train_cnn(cnn, train_loader, test_loader, epochs=50, device=device)
    cnn_accs = per_class_accuracy(y_true_cnn, y_pred_cnn)

    # ---------- Group summary ----------
    latent_summary = group_by_size(le, y_test, latent_accs, manuf_labels)
    cnn_summary = group_by_size(le, y_true_cnn, cnn_accs, manuf_labels)

    rows = []
    for group in ["rich", "medium", "low"]:
        rows.append(
            {
                "Group": group,
                "Manufacturers": latent_summary["counts"][group],
                "CNN_acc": cnn_summary[group],
                "PI-VAE_acc": latent_summary[group],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = "results/sample_size_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

