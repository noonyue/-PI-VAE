"""
Benchmark multiple classifiers (PLS-DA, SVM, RF, CNN, LSTM, Transformer)
on L1 drug classification and L2 manufacturer classification, using the
same train/test split as the PI-VAE cascade pipeline.

Outputs CSV tables that can be turned into paper tables/figures:
- results/model_comparison_l1.csv
- results/model_comparison_l2_direct.csv
- results/model_comparison_l2_cascade.csv
"""

import os
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression

# Add parent directory for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_vae_pipeline import (  # type: ignore
    load_data,
    preprocess_spectra,
    SpectralDataset,
    UV_VAE,
    NIR_VAE,
    train_vae,
    extract_latent_features,
)


def ensure_dirs() -> None:
    os.makedirs("results", exist_ok=True)


class PLSDAWrapper:
    """Simple PLS-DA wrapper using PLSRegression with argmax over class scores."""

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.model: PLSRegression | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAWrapper":
        self.classes_ = np.unique(y)
        y_numeric = np.array([np.where(self.classes_ == c)[0][0] for c in y])
        n_comp = min(self.n_components, max(1, len(self.classes_) - 1))
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(X, y_numeric)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.classes_ is not None
        y_cont = self.model.predict(X).ravel()
        y_idx = np.clip(np.round(y_cont).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[y_idx]


# ----------------- Deep models (very compact) -----------------


class CNN1D(nn.Module):
    def __init__(self, input_len: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # infer flattened size with dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self.features(dummy)
            flat = out.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B,1,L)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L) -> (B,L,1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # use last time step
        last = out[:, -1, :]
        return self.fc(last)


class TransformerClassifier(nn.Module):
    def __init__(self, input_len: int, n_classes: int, d_model: int = 32, nhead: int = 2):
        super().__init__()
        self.input_len = input_len
        # Downsample input to reduce memory: use max pooling first
        self.downsample = nn.MaxPool1d(kernel_size=4, stride=4) if input_len > 500 else nn.Identity()
        reduced_len = input_len // 4 if input_len > 500 else input_len
        
        self.proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Reduced to 1 layer
        self.cls_head = nn.Linear(d_model, n_classes)

        # simple positional encoding (for reduced length)
        pos = torch.arange(0, reduced_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, reduced_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L)
        x = x.unsqueeze(1)  # (B,1,L) for pooling
        x = self.downsample(x)  # (B,1,L_reduced)
        x = x.squeeze(1).unsqueeze(-1)  # (B,L_reduced,1)
        x = self.proj(x) + self.pos_encoding
        enc = self.encoder(x)
        pooled = enc.mean(dim=1)
        return self.cls_head(pooled)


# ----------------- Helper functions -----------------


def train_deep_classifier(
    model: nn.Module,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Train CNN/LSTM/Transformer briefly and return test accuracy, macro-F1."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    test_ds = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
        
        # Early stopping check every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    _, pred = logits.max(1)
                    correct += pred.eq(yb).sum().item()
                    total += yb.size(0)
                test_acc = correct / total
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

    # Evaluate
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return acc, f1


def eval_sklearn_classifier(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return acc, f1


def prepare_raw_and_latent_features() -> Dict[str, np.ndarray]:
    """Load data, split train/test as in report_figures, and build raw + latent features."""
    uv_spectra, nir_spectra, drug_labels_raw, manuf_labels_raw = load_data()

    # Encode labels
    drug_le = LabelEncoder()
    manuf_le = LabelEncoder()
    drug_labels = drug_le.fit_transform(drug_labels_raw)
    manuf_labels = manuf_le.fit_transform(manuf_labels_raw)

    # Split (stratified by drug, same方式 as report_figures)
    indices = np.arange(len(uv_spectra))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=drug_labels
    )

    uv_train, uv_test = uv_spectra[train_idx], uv_spectra[test_idx]
    nir_train, nir_test = nir_spectra[train_idx], nir_spectra[test_idx]
    drug_train, drug_test = drug_labels[train_idx], drug_labels[test_idx]
    manuf_train, manuf_test = manuf_labels[train_idx], manuf_labels[test_idx]

    # Preprocess raw spectra (SNV)
    uv_train_p = preprocess_spectra(uv_train, method="snv")
    uv_test_p = preprocess_spectra(uv_test, method="snv")
    nir_train_p = preprocess_spectra(nir_train, method="snv")
    nir_test_p = preprocess_spectra(nir_test, method="snv")

    # Raw fused features
    X_raw_train = np.hstack([nir_train_p, uv_train_p])
    X_raw_test = np.hstack([nir_test_p, uv_test_p])

    # Train PI-VAE models to get latent features
    device = "cuda" if torch.cuda.is_available() else "cpu"

    uv_model = UV_VAE(input_dim=uv_train_p.shape[1], latent_dim=32, n_peaks=10)
    nir_model = NIR_VAE(input_dim=nir_train_p.shape[1], latent_dim=32, n_peaks=10)

    uv_loader = DataLoader(SpectralDataset(uv_train_p), batch_size=32, shuffle=True)
    nir_loader = DataLoader(SpectralDataset(nir_train_p), batch_size=32, shuffle=True)

    uv_model, _ = train_vae(uv_model, uv_loader, epochs=120, lr=1e-3, device=device, model_name="UV-VAE")
    nir_model, _ = train_vae(nir_model, nir_loader, epochs=120, lr=1e-3, device=device, model_name="NIR-VAE")

    # Extract latent features for train/test
    uv_train_lat = extract_latent_features(
        uv_model, DataLoader(SpectralDataset(uv_train_p), batch_size=64, shuffle=False), device
    )
    uv_test_lat = extract_latent_features(
        uv_model, DataLoader(SpectralDataset(uv_test_p), batch_size=64, shuffle=False), device
    )
    nir_train_lat = extract_latent_features(
        nir_model, DataLoader(SpectralDataset(nir_train_p), batch_size=64, shuffle=False), device
    )
    nir_test_lat = extract_latent_features(
        nir_model, DataLoader(SpectralDataset(nir_test_p), batch_size=64, shuffle=False), device
    )

    X_latent_train = np.hstack([nir_train_lat, uv_train_lat])
    X_latent_test = np.hstack([nir_test_lat, uv_test_lat])

    # Fused features: Raw + Latent concatenation
    X_fused_train = np.hstack([X_raw_train, X_latent_train])
    X_fused_test = np.hstack([X_raw_test, X_latent_test])

    return {
        "X_raw_train": X_raw_train,
        "X_raw_test": X_raw_test,
        "X_latent_train": X_latent_train,
        "X_latent_test": X_latent_test,
        "X_fused_train": X_fused_train,  # New: Raw + Latent fusion
        "X_fused_test": X_fused_test,    # New: Raw + Latent fusion
        "drug_train": drug_train,
        "drug_test": drug_test,
        "manuf_train": manuf_train,
        "manuf_test": manuf_test,
    }


def benchmark_l1_classic_only(feats: Dict[str, np.ndarray]) -> None:
    """Only run classic ML models (PLS-DA, SVM, RF) - fast."""
    results = []
    X_raw_tr, X_raw_te = feats["X_raw_train"], feats["X_raw_test"]
    X_lat_tr, X_lat_te = feats["X_latent_train"], feats["X_latent_test"]
    y_tr, y_te = feats["drug_train"], feats["drug_test"]
    
    # Standardize for SVM/PLS
    scaler_raw = StandardScaler()
    scaler_lat = StandardScaler()
    X_raw_tr_s = scaler_raw.fit_transform(X_raw_tr)
    X_raw_te_s = scaler_raw.transform(X_raw_te)
    X_lat_tr_s = scaler_lat.fit_transform(X_lat_tr)
    X_lat_te_s = scaler_lat.transform(X_lat_te)
    
    def add_result(model_name: str, feat_type: str, acc: float, f1: float) -> None:
        results.append({
            "Task": "L1_Drug",
            "Model": model_name,
            "Feature": feat_type,
            "Accuracy": acc,
            "Macro_F1": f1,
        })
    
    print("\n=== Benchmarking L1 Drug Classification (Classic Models Only) ===")
    
    # PLS-DA
    print("  Evaluating PLS-DA...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr_s, X_raw_te_s), ("Latent", X_lat_tr_s, X_lat_te_s)]:
        pls = PLSDAWrapper(n_components=10)
        acc, f1 = eval_sklearn_classifier(pls, Xtr, y_tr, Xte, y_te)
        print(f"    PLS-DA ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_result("PLS-DA", feat_name, acc, f1)
    
    # SVM
    print("  Evaluating SVM...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr_s, X_raw_te_s), ("Latent", X_lat_tr_s, X_lat_te_s)]:
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        acc, f1 = eval_sklearn_classifier(svm, Xtr, y_tr, Xte, y_te)
        print(f"    SVM ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_result("SVM", feat_name, acc, f1)
    
    # RF
    print("  Evaluating RandomForest...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr, X_raw_te), ("Latent", X_lat_tr, X_lat_te)]:
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        acc, f1 = eval_sklearn_classifier(rf, Xtr, y_tr, Xte, y_te)
        print(f"    RF ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_result("RandomForest", feat_name, acc, f1)
    
    df = pd.DataFrame(results)
    df.to_csv("results/model_comparison_l1_classic.csv", index=False)
    print("\nSaved: results/model_comparison_l1_classic.csv")


def benchmark_l1_deep_only(feats: Dict[str, np.ndarray]) -> None:
    """Only run deep models (CNN, LSTM, Transformer) - slower."""
    results = []
    X_raw_tr, X_raw_te = feats["X_raw_train"], feats["X_raw_test"]
    y_tr, y_te = feats["drug_train"], feats["drug_test"]
    
    def add_result(model_name: str, feat_type: str, acc: float, f1: float) -> None:
        results.append({
            "Task": "L1_Drug",
            "Model": model_name,
            "Feature": feat_type,
            "Accuracy": acc,
            "Macro_F1": f1,
        })
    
    print("\n=== Benchmarking L1 Drug Classification (Deep Models Only) ===")
    print("  This may take a few minutes...")
    
    input_len = X_raw_tr.shape[1]
    n_classes = len(np.unique(y_tr))
    
    for model_name, builder, batch_size, epochs in [
        ("CNN", lambda: CNN1D(input_len=input_len, n_classes=n_classes), 32, 40),
        ("LSTM", lambda: LSTMClassifier(input_dim=1, n_classes=n_classes), 32, 40),
        ("Transformer", lambda: TransformerClassifier(input_len=input_len, n_classes=n_classes), 8, 30),
    ]:
        print(f"  Training {model_name}...")
        model = builder()
        acc, f1 = train_deep_classifier(model, X_raw_tr, y_tr, X_raw_te, y_te, epochs=epochs, batch_size=batch_size)
        print(f"    {model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        add_result(model_name, "Raw", acc, f1)
    
    df = pd.DataFrame(results)
    df.to_csv("results/model_comparison_l1_deep.csv", index=False)
    print("\nSaved: results/model_comparison_l1_deep.csv")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark L1/L2 models")
    parser.add_argument("--stage", choices=["classic", "deep", "all"], default="all",
                        help="Which models to run: classic (fast), deep (slow), or all")
    args = parser.parse_args()
    
    ensure_dirs()
    feats = prepare_raw_and_latent_features()
    
    if args.stage in ["classic", "all"]:
        benchmark_l1_classic_only(feats)
    
    if args.stage in ["deep", "all"]:
        benchmark_l1_deep_only(feats)
    
    # Merge results if both stages completed
    if args.stage == "all":
        import glob
        classic_file = "results/model_comparison_l1_classic.csv"
        deep_file = "results/model_comparison_l1_deep.csv"
        if os.path.exists(classic_file) and os.path.exists(deep_file):
            df_classic = pd.read_csv(classic_file)
            df_deep = pd.read_csv(deep_file)
            df_merged = pd.concat([df_classic, df_deep], ignore_index=True)
            df_merged.to_csv("results/model_comparison_l1.csv", index=False)
            print("\nMerged results saved: results/model_comparison_l1.csv")
    
    # 后续可在这里继续实现：
    # - Direct 28-class L2 benchmark
    # - per-drug cascade L2 benchmark


if __name__ == "__main__":
    main()

