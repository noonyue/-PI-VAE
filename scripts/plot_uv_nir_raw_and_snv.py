#!/usr/bin/env python
"""
Standalone script: Plot UV-Vis and NIR spectra (raw and SNV-processed)
in a single figure with 4 subplots:
  (A) UV-Vis Raw Spectra
  (B) NIR Raw Spectra
  (C) UV-Vis SNV Spectra
  (D) NIR SNV Spectra

Reads data directly from Sampedata0.xlsx (VIS_0 and NIR_0).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "Sampedata0.xlsx"
UV_SHEET = "VIS_0"
NIR_SHEET = "NIR_0"

OUT_DIR = "figures_new"
os.makedirs(OUT_DIR, exist_ok=True)


def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate preprocessing: per-sample (x - mean) / std."""
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True) + 1e-8
    return (spectra - mean) / std


def autoscale_ylim(ax, data, lower=1, upper=99):
    """Set y-limits based on percentiles to avoid being dominated by outliers."""
    y_min, y_max = np.percentile(data, [lower, upper])
    ax.set_ylim(y_min, y_max)


def main():
    print("Loading data from Excel...")
    df_uv = pd.read_excel(DATA_FILE, sheet_name=UV_SHEET, header=None)
    df_nir = pd.read_excel(DATA_FILE, sheet_name=NIR_SHEET, header=None)

    # Assume column 0 = drug, column 1 = manufacturer, from column 2 onward = spectra
    uv_raw = df_uv.iloc[:, 2:].to_numpy(dtype=float)
    nir_raw = df_nir.iloc[:, 2:].to_numpy(dtype=float)

    # SNV preprocessing
    uv_snv = snv(uv_raw)
    nir_snv = snv(nir_raw)

    # X-axis: index-based. Replace with real wavelength if available.
    uv_x = np.arange(uv_raw.shape[1])
    nir_x = np.arange(nir_raw.shape[1])

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_uv_raw, ax_nir_raw, ax_uv_snv, ax_nir_snv = axes.ravel()

    # ---------- (A) UV-Vis Raw ----------
    for spec in uv_raw:
        ax_uv_raw.plot(uv_x, spec, color="tab:blue", alpha=0.08, linewidth=0.6)
    ax_uv_raw.set_title("(A) UV-Vis Raw Spectra")
    ax_uv_raw.set_xlabel("Wavelength index")
    ax_uv_raw.set_ylabel("Intensity")
    ax_uv_raw.set_xlim(uv_x.min(), uv_x.max())
    autoscale_ylim(ax_uv_raw, uv_raw)

    # ---------- (B) NIR Raw ----------
    for spec in nir_raw:
        ax_nir_raw.plot(nir_x, spec, color="tab:orange", alpha=0.08, linewidth=0.6)
    ax_nir_raw.set_title("(B) NIR Raw Spectra")
    ax_nir_raw.set_xlabel("Wavelength index")
    ax_nir_raw.set_ylabel("Intensity")
    ax_nir_raw.set_xlim(nir_x.min(), nir_x.max())
    autoscale_ylim(ax_nir_raw, nir_raw)

    # ---------- (C) UV-Vis SNV ----------
    for spec in uv_snv:
        ax_uv_snv.plot(uv_x, spec, color="tab:blue", alpha=0.08, linewidth=0.6)
    ax_uv_snv.set_title("(C) UV-Vis SNV Spectra")
    ax_uv_snv.set_xlabel("Wavelength index")
    ax_uv_snv.set_ylabel("SNV intensity")
    ax_uv_snv.set_xlim(uv_x.min(), uv_x.max())
    autoscale_ylim(ax_uv_snv, uv_snv)

    # ---------- (D) NIR SNV ----------
    for spec in nir_snv:
        ax_nir_snv.plot(nir_x, spec, color="tab:orange", alpha=0.08, linewidth=0.6)
    ax_nir_snv.set_title("(D) NIR SNV Spectra")
    ax_nir_snv.set_xlabel("Wavelength index")
    ax_nir_snv.set_ylabel("SNV intensity")
    ax_nir_snv.set_xlim(nir_x.min(), nir_x.max())
    autoscale_ylim(ax_nir_snv, nir_snv)

    fig.suptitle("UV-Vis & NIR Spectra (Raw and SNV)", fontsize=14, fontweight="bold")

    save_path = os.path.join(OUT_DIR, "uv_nir_raw_and_snv.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()

