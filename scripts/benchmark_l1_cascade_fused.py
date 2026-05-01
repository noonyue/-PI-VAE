"""
Evaluate L1 drug classification using Cascade method with FUSED features (Raw + Latent).
"""
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_l1_l2_models import ensure_dirs, prepare_raw_and_latent_features

def benchmark_l1_cascade_fused() -> None:
    """Evaluate L1 drug classification using fused features (Raw + Latent) with SVM."""
    ensure_dirs()
    feats = prepare_raw_and_latent_features()
    
    X_fused_tr = feats["X_fused_train"]
    X_fused_te = feats["X_fused_test"]
    y_tr = feats["drug_train"]
    y_te = feats["drug_test"]
    
    # Standardize fused features
    scaler = StandardScaler()
    X_fused_tr_s = scaler.fit_transform(X_fused_tr)
    X_fused_te_s = scaler.transform(X_fused_te)
    
    # Train SVM on fused features
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm.fit(X_fused_tr_s, y_tr)
    y_pred = svm.predict(X_fused_te_s)
    
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    
    print(f"\n=== L1 Cascade with Fused Features (Raw+Latent) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    
    # Save result
    result = pd.DataFrame([{
        "Task": "L1_Drug",
        "Model": "PI-VAE+SVM (Cascade)",
        "Feature": "Fused",
        "Accuracy": acc,
        "Macro_F1": f1
    }])
    
    result.to_csv("results/model_comparison_l1_cascade_fused.csv", index=False)
    print("Saved: results/model_comparison_l1_cascade_fused.csv")
    
    return acc, f1

if __name__ == "__main__":
    benchmark_l1_cascade_fused()
