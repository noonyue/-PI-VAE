"""
Benchmark classifiers on L2 manufacturer classification.

Tasks:
- Direct 28-class manufacturer classification (whole dataset)
- Cascade per-drug manufacturer classification (RF + latent features)

Models:
- Classic: PLS-DA, SVM, RandomForest
- Deep: CNN, LSTM, Transformer (Raw spectra only, direct 28-class)

Outputs:
- results/model_comparison_l2_direct_classic.csv
- results/model_comparison_l2_direct_deep.csv
- results/model_comparison_l2_cascade_per_drug.csv
- results/model_comparison_l2_cascade_summary.csv
"""

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Reuse helpers from L1 benchmarking script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_l1_l2_models import (  # type: ignore
    ensure_dirs,
    PLSDAWrapper,
    CNN1D,
    LSTMClassifier,
    TransformerClassifier,
    train_deep_classifier,
    eval_sklearn_classifier,
    prepare_raw_and_latent_features,
)


def benchmark_l2_direct_classic(feats: Dict[str, np.ndarray]) -> None:
    """Direct 28-class manufacturer classification with classic models."""
    ensure_dirs()
    X_raw_tr, X_raw_te = feats["X_raw_train"], feats["X_raw_test"]
    X_lat_tr, X_lat_te = feats["X_latent_train"], feats["X_latent_test"]
    y_tr, y_te = feats["manuf_train"], feats["manuf_test"]

    results = []

    # Standardize for SVM / PLS
    scaler_raw = StandardScaler()
    scaler_lat = StandardScaler()
    X_raw_tr_s = scaler_raw.fit_transform(X_raw_tr)
    X_raw_te_s = scaler_raw.transform(X_raw_te)
    X_lat_tr_s = scaler_lat.fit_transform(X_lat_tr)
    X_lat_te_s = scaler_lat.transform(X_lat_te)

    def add_row(model: str, feat: str, acc: float, f1: float) -> None:
        results.append(
            {
                "Task": "L2_Manufacturer_Direct",
                "Model": model,
                "Feature": feat,
                "Accuracy": acc,
                "Macro_F1": f1,
            }
        )

    print("\n=== L2 Direct 28-class Manufacturer Classification (Classic Models) ===")

    # PLS-DA
    print("  Evaluating PLS-DA...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr_s, X_raw_te_s), ("Latent", X_lat_tr_s, X_lat_te_s)]:
        pls = PLSDAWrapper(n_components=10)
        acc, f1 = eval_sklearn_classifier(pls, Xtr, y_tr, Xte, y_te)
        print(f"    PLS-DA ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_row("PLS-DA", feat_name, acc, f1)

    # SVM
    print("  Evaluating SVM...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr_s, X_raw_te_s), ("Latent", X_lat_tr_s, X_lat_te_s)]:
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        acc, f1 = eval_sklearn_classifier(svm, Xtr, y_tr, Xte, y_te)
        print(f"    SVM ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_row("SVM", feat_name, acc, f1)

    # RandomForest
    print("  Evaluating RandomForest...")
    for feat_name, Xtr, Xte in [("Raw", X_raw_tr, X_raw_te), ("Latent", X_lat_tr, X_lat_te)]:
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        acc, f1 = eval_sklearn_classifier(rf, Xtr, y_tr, Xte, y_te)
        print(f"    RF ({feat_name}): Acc={acc:.4f}, F1={f1:.4f}")
        add_row("RandomForest", feat_name, acc, f1)

    df = pd.DataFrame(results)
    out_path = "results/model_comparison_l2_direct_classic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def benchmark_l2_direct_deep(feats: Dict[str, np.ndarray]) -> None:
    """Direct 28-class manufacturer classification with deep models (Raw only)."""
    ensure_dirs()
    X_raw_tr, X_raw_te = feats["X_raw_train"], feats["X_raw_test"]
    y_tr, y_te = feats["manuf_train"], feats["manuf_test"]

    results = []

    def add_row(model: str, acc: float, f1: float) -> None:
        results.append(
            {
                "Task": "L2_Manufacturer_Direct",
                "Model": model,
                "Feature": "Raw",
                "Accuracy": acc,
                "Macro_F1": f1,
            }
        )

    print("\n=== L2 Direct 28-class Manufacturer Classification (Deep Models) ===")
    input_len = X_raw_tr.shape[1]
    n_classes = len(np.unique(y_tr))

    for model_name, builder, batch_size, epochs in [
        ("CNN", lambda: CNN1D(input_len=input_len, n_classes=n_classes), 32, 40),
        ("LSTM", lambda: LSTMClassifier(input_dim=1, n_classes=n_classes), 32, 40),
        ("Transformer", lambda: TransformerClassifier(input_len=input_len, n_classes=n_classes), 8, 30),
    ]:
        print(f"  Training {model_name}...")
        model = builder()
        acc, f1 = train_deep_classifier(
            model,
            X_raw_tr,
            y_tr,
            X_raw_te,
            y_te,
            epochs=epochs,
            batch_size=batch_size,
        )
        print(f"    {model_name}: Acc={acc:.4f}, F1={f1:.4f}")
        add_row(model_name, acc, f1)

    df = pd.DataFrame(results)
    out_path = "results/model_comparison_l2_direct_deep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def benchmark_l2_cascade_rf_fused(feats: Dict[str, np.ndarray]) -> None:
    """
    Cascade per-drug L2 classification with FUSED features (Raw + Latent):
    - For each drug, train an RF classifier on fused features to predict manufacturer.
    - Report per-drug accuracy and weighted average over all drugs.
    """
    ensure_dirs()
    X_fused_tr, X_fused_te = feats["X_fused_train"], feats["X_fused_test"]
    drug_tr, drug_te = feats["drug_train"], feats["drug_test"]
    manuf_tr, manuf_te = feats["manuf_train"], feats["manuf_test"]

    rows = []
    all_true = []
    all_pred = []

    unique_drugs = np.unique(drug_tr)
    print("\n=== L2 Cascade per-drug (RF + Fused: Raw+Latent) ===")

    for d in unique_drugs:
        mask_tr = drug_tr == d
        mask_te = drug_te == d
        if not np.any(mask_te) or np.sum(mask_tr) < 2:
            continue

        X_tr_d = X_fused_tr[mask_tr]
        y_tr_d = manuf_tr[mask_tr]
        X_te_d = X_fused_te[mask_te]
        y_te_d = manuf_te[mask_te]

        rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        rf.fit(X_tr_d, y_tr_d)
        y_pred_d = rf.predict(X_te_d)

        acc = accuracy_score(y_te_d, y_pred_d)
        f1 = f1_score(y_te_d, y_pred_d, average="macro")

        print(
            f"  Drug {d}: Train={len(X_tr_d)}, Test={len(X_te_d)}, "
            f"Acc={acc:.4f}, F1={f1:.4f}"
        )

        rows.append(
            {
                "Drug": int(d),
                "Train_Samples": int(len(X_tr_d)),
                "Test_Samples": int(len(X_te_d)),
                "Accuracy": acc,
                "Macro_F1": f1,
            }
        )

        all_true.extend(y_te_d.tolist())
        all_pred.extend(y_pred_d.tolist())

    df_per_drug = pd.DataFrame(rows)
    per_drug_path = "results/model_comparison_l2_cascade_per_drug.csv"
    df_per_drug.to_csv(per_drug_path, index=False)
    print(f"\nSaved per-drug results: {per_drug_path}")

    # Weighted average over drugs
    if rows:
        total_test = sum(r["Test_Samples"] for r in rows)
        weighted_acc = sum(r["Accuracy"] * r["Test_Samples"] for r in rows) / total_test
        weighted_f1 = sum(r["Macro_F1"] * r["Test_Samples"] for r in rows) / total_test
    else:
        weighted_acc = float("nan")
        weighted_f1 = float("nan")

    overall_acc = accuracy_score(all_true, all_pred) if all_true else float("nan")
    overall_f1 = f1_score(all_true, all_pred, average="macro") if all_true else float("nan")

    summary = pd.DataFrame(
        [
            {
                "Task": "L2_Manufacturer_Cascade",
                "Model": "RandomForest",
                "Feature": "Fused",  # Changed from "Latent" to "Fused"
                "Weighted_Accuracy": weighted_acc,
                "Weighted_Macro_F1": weighted_f1,
                "Overall_Accuracy": overall_acc,
                "Overall_Macro_F1": overall_f1,
            }
        ]
    )
    summary_path = "results/model_comparison_l2_cascade_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved cascade summary: {summary_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark L2 manufacturer models")
    parser.add_argument(
        "--stage",
        choices=["classic", "deep", "cascade", "all"],
        default="all",
        help="Which part to run",
    )
    args = parser.parse_args()

    ensure_dirs()
    feats = prepare_raw_and_latent_features()

    if args.stage in ("classic", "all"):
        benchmark_l2_direct_classic(feats)
    if args.stage in ("deep", "all"):
        benchmark_l2_direct_deep(feats)
    if args.stage in ("cascade", "all"):
        benchmark_l2_cascade_rf_fused(feats)  # Changed to use fused features


if __name__ == "__main__":
    main()

