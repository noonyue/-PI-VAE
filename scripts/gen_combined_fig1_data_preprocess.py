#!/usr/bin/env python3
"""Combined Figure 1: Dataset Overview & SNV Preprocessing (6 panels)"""
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotting_style import setup_style, add_panel_label, COLOR_UV, COLOR_NIR, COLOR_BASELINE, COLOR_PRED
setup_style()


def snv(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / 'figures' / 'redrawn'
    input_path = project_root / 'Sampedata0.xlsx'

    output_dir.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(input_path)
    df_vis = xl.parse('VIS_0', header=None)
    df_nir = xl.parse('NIR_0', header=None)
    drug_labels = df_vis.iloc[:, 0].values
    mfr_labels  = df_vis.iloc[:, 1].values
    spectra_vis = df_vis.iloc[:, 2:].values.astype(float)
    spectra_nir = df_nir.iloc[:, 2:].values.astype(float)
    drug_names  = ['CIM','FMD','GLD','GSR','HCT','IBU','MHE','MHL','MHR']
    colors9     = plt.cm.tab10(np.linspace(0, 0.9, 9))
    drug_counts, mfr_counts = [], []
    for d in drug_names:
        mask = drug_labels == d
        drug_counts.append(int(mask.sum()))
        mfr_counts.append(len(np.unique(mfr_labels[mask])))
    n_vis = spectra_vis.shape[1]
    n_nir = spectra_nir.shape[1]
    wl_vis = np.linspace(200, 800, n_vis)
    wl_nir = np.linspace(900, 2500, n_nir)

    fig = plt.figure(figsize=(18, 11), facecolor='white')
    fig.suptitle('Figure 1.  Dataset Overview & SNV Spectral Preprocessing',
                 fontsize=14, fontweight='bold', y=1.005)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    # (a) Sample count + manufacturer count
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(drug_names)); w = 0.38
    ax.bar(x - w/2, drug_counts, w, color=COLOR_BASELINE, alpha=0.85, edgecolor='black', lw=1.2, label='Total Samples')
    ax.bar(x + w/2, mfr_counts,  w, color=COLOR_PRED,     alpha=0.85, edgecolor='black', lw=1.2, label='# Manufacturers')
    for i, (s, m) in enumerate(zip(drug_counts, mfr_counts)):
        ax.text(i - w/2, s + 0.4, str(s), ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.text(i + w/2, m + 0.4, str(m), ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(drug_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_ylim(0, max(drug_counts) * 1.25)
    ax.set_title('a Dataset Distribution\n(9 drugs · 28 manufacturers · 357 samples)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, ls='--'); ax.set_axisbelow(True)
    add_panel_label(ax, 'a', x_offset=-0.16, y_offset=1.04)

    # (b) Stacked bar: samples per manufacturer per drug
    ax = fig.add_subplot(gs[0, 1])
    unique_mfrs = list(pd.unique(mfr_labels))
    mfr_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_mfrs)))
    mfr_color_map = {mfr: mfr_colors[i] for i, mfr in enumerate(unique_mfrs)}
    bottom = np.zeros(len(drug_names))
    for mfr in unique_mfrs:
        heights = []
        for d in drug_names:
            mask = drug_labels == d
            heights.append(int(np.sum(mfr_labels[mask] == mfr)))
        ax.bar(np.arange(len(drug_names)), heights, bottom=bottom,
               color=mfr_color_map[mfr], edgecolor='white', lw=0.4, width=0.6)
        bottom += np.array(heights, dtype=float)
    ax.set_xticks(np.arange(len(drug_names)))
    ax.set_xticklabels(drug_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Sample Count', fontweight='bold')
    ax.set_title('b Per-Manufacturer Sample Breakdown\n(Each color = 1 manufacturer)', fontsize=10.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls='--'); ax.set_axisbelow(True)
    add_panel_label(ax, 'b', x_offset=-0.16, y_offset=1.04)

    # (c) UV-Vis raw spectra (all drugs)
    ax = fig.add_subplot(gs[0, 2])
    for i, (d, col) in enumerate(zip(drug_names, colors9)):
        idx = np.where(drug_labels == d)[0]
        for j, k in enumerate(idx[:5]):
            ax.plot(wl_vis, spectra_vis[k], color=col, lw=0.9, alpha=0.55,
                    label=d if j == 0 else None)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Absorbance (raw)', fontweight='bold', fontsize=9)
    ax.set_title('c UV-Vis Raw Spectra\n(5 samples/drug, colored by drug type)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, framealpha=0.8, loc='upper right')
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'c', x_offset=-0.16, y_offset=1.04)

    # (d) UV-Vis after SNV
    ax = fig.add_subplot(gs[1, 0])
    for i, (d, col) in enumerate(zip(drug_names, colors9)):
        idx = np.where(drug_labels == d)[0]
        for j, k in enumerate(idx[:5]):
            ax.plot(wl_vis, snv(spectra_vis[k]), color=col, lw=0.9, alpha=0.55,
                    label=d if j == 0 else None)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Absorbance (SNV)', fontweight='bold', fontsize=9)
    ax.set_title('d UV-Vis After SNV Normalization\n(Baseline shift eliminated)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, framealpha=0.8, loc='upper right')
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'd', x_offset=-0.16, y_offset=1.04)

    # (e) NIR raw spectra
    ax = fig.add_subplot(gs[1, 1])
    for i, (d, col) in enumerate(zip(drug_names, colors9)):
        idx = np.where(drug_labels == d)[0]
        for j, k in enumerate(idx[:5]):
            ax.plot(wl_nir, spectra_nir[k], color=col, lw=0.9, alpha=0.55,
                    label=d if j == 0 else None)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Absorbance (raw)', fontweight='bold', fontsize=9)
    ax.set_title('e NIR Raw Spectra\n(5 samples/drug, colored by drug type)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, framealpha=0.8, loc='upper right')
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'e', x_offset=-0.16, y_offset=1.04)

    # (f) NIR after SNV
    ax = fig.add_subplot(gs[1, 2])
    for i, (d, col) in enumerate(zip(drug_names, colors9)):
        idx = np.where(drug_labels == d)[0]
        for j, k in enumerate(idx[:5]):
            ax.plot(wl_nir, snv(spectra_nir[k]), color=col, lw=0.9, alpha=0.55,
                    label=d if j == 0 else None)
    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Absorbance (SNV)', fontweight='bold', fontsize=9)
    ax.set_title('f NIR After SNV Normalization\n(Scatter effects corrected)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, framealpha=0.8, loc='upper right')
    ax.grid(alpha=0.2, ls='--')
    add_panel_label(ax, 'f', x_offset=-0.16, y_offset=1.04)

    output_path = output_dir / 'combined_fig1_data_preprocess.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'[OK] {output_path}')
    plt.close()

if __name__ == '__main__':
    main()
