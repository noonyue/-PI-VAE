#!/usr/bin/env python
"""
Plot UV-Vis and NIR spectra (raw and SNV-processed) colored by
drug or manufacturer. One figure has 4 subplots:
  (A) UV-Vis Raw Spectra
  (B) NIR Raw Spectra
  (C) UV-Vis SNV Spectra
  (D) NIR SNV Spectra

This script produces two figures:
  - uv_nir_raw_and_snv_by_drug.png
  - uv_nir_raw_and_snv_by_manufacturer.png
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


def plot_by_group(group_type: str):
    """
    group_type: 'drug' or 'manufacturer'
    """
    assert group_type in {"drug", "manufacturer"}

    print(f"Loading data for group_type={group_type}...")
    df_uv = pd.read_excel(DATA_FILE, sheet_name=UV_SHEET, header=None)
    df_nir = pd.read_excel(DATA_FILE, sheet_name=NIR_SHEET, header=None)

    # Column 0 = drug, column 1 = manufacturer, from column 2 onward = spectra
    if group_type == "drug":
        uv_group = df_uv.iloc[:, 0].astype(str).to_numpy()
        nir_group = df_nir.iloc[:, 0].astype(str).to_numpy()
        group_label = "Drug"
        suffix = "by_drug"
    else:
        uv_group = df_uv.iloc[:, 1].astype(str).to_numpy()
        nir_group = df_nir.iloc[:, 1].astype(str).to_numpy()
        group_label = "Manufacturer"
        suffix = "by_manufacturer"

    uv_raw = df_uv.iloc[:, 2:].to_numpy(dtype=float)
    nir_raw = df_nir.iloc[:, 2:].to_numpy(dtype=float)

    uv_snv = snv(uv_raw)
    nir_snv = snv(nir_raw)

    # X 轴改为真实波长：UV-Vis 300–1000 nm, NIR 1750–2150 nm（线性采样）
    uv_x = np.linspace(300, 1000, uv_raw.shape[1])
    nir_x = np.linspace(1750, 2150, nir_raw.shape[1])

    # Bold fonts globally
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_uv_raw, ax_nir_raw, ax_uv_snv, ax_nir_snv = axes.ravel()

    # Color map per group
    unique_groups = sorted(np.unique(uv_group))
    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % 10) for i, g in enumerate(unique_groups)}

    # Helper to plot grouped spectra
    def plot_grouped(ax, x, specs, groups, title, ylabel):
        # 多条淡线 + 每个分组一条粗均值线
        for g in unique_groups:
            mask = groups == g
            if not np.any(mask):
                continue
            color = color_map[g]
            # 所有样本（淡）
            for spec in specs[mask]:
                ax.plot(x, spec, color=color, alpha=0.08, linewidth=0.6)
            # 均值谱（粗实线）
            mean_spec = specs[mask].mean(axis=0)
            ax.plot(x, mean_spec, color=color, linewidth=2.0, label=str(g))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_xlim(x.min(), x.max())

    # ---------- (A) UV-Vis Raw ----------
    plot_grouped(ax_uv_raw, uv_x, uv_raw, uv_group,
                 f"(A) UV-Vis Raw Spectra\nColored by {group_label}", "Intensity")
    autoscale_ylim(ax_uv_raw, uv_raw)

    # ---------- (B) NIR Raw ----------
    plot_grouped(ax_nir_raw, nir_x, nir_raw, nir_group,
                 f"(B) NIR Raw Spectra\nColored by {group_label}", "Intensity")
    autoscale_ylim(ax_nir_raw, nir_raw)

    # ---------- (C) UV-Vis SNV ----------
    plot_grouped(ax_uv_snv, uv_x, uv_snv, uv_group,
                 f"(C) UV-Vis SNV Spectra\nColored by {group_label}", "SNV intensity")
    autoscale_ylim(ax_uv_snv, uv_snv)

    # ---------- (D) NIR SNV ----------
    plot_grouped(ax_nir_snv, nir_x, nir_snv, nir_group,
                 f"(D) NIR SNV Spectra\nColored by {group_label}", "SNV intensity")
    autoscale_ylim(ax_nir_snv, nir_snv)

    # 只在右下角子图放统一图例，避免重复
    handles, labels = ax_nir_snv.get_legend_handles_labels()
    ax_nir_snv.legend(handles, labels, title=group_label,
                      fontsize=8, title_fontsize=9, ncol=2, framealpha=0.9,
                      loc="upper left")

    fig.suptitle(f"UV-Vis & NIR Spectra (Raw and SNV) Colored by {group_label}",
                 fontsize=14, fontweight="bold")

    save_name = f"uv_nir_raw_and_snv_{suffix}.png"
    save_path = os.path.join(OUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # 按药物着色
    plot_by_group("drug")
    # 按厂家着色
    plot_by_group("manufacturer")


if __name__ == "__main__":
    main()

