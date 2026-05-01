#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "Sampedata0.xlsx"
UV_SHEET = "VIS_0"
NIR_SHEET = "NIR_0"

FIGURES_DIR = os.environ.get("FIGURES_DIR", "figures_new")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("Loading data...")
    uv_df = pd.read_excel(DATA_FILE, sheet_name=UV_SHEET, header=None)
    nir_df = pd.read_excel(DATA_FILE, sheet_name=NIR_SHEET, header=None)

    uv_specs = uv_df.iloc[:, 2:].values
    nir_specs = nir_df.iloc[:, 2:].values
    uv_x = np.arange(uv_specs.shape[1])
    nir_x = np.arange(nir_specs.shape[1])

    plt.style.use("seaborn-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax_uv, ax_nir = axes

    for i in range(uv_specs.shape[0]):
        ax_uv.plot(uv_x, uv_specs[i], color="tab:blue", alpha=0.08, linewidth=0.8)
    ax_uv.set_title("UV-Vis Raw Spectra (all samples)")
    ax_uv.set_xlabel("Wavelength idx")
    ax_uv.set_ylabel("Intensity")

    for i in range(nir_specs.shape[0]):
        ax_nir.plot(nir_x, nir_specs[i], color="tab:orange", alpha=0.08, linewidth=0.8)
    ax_nir.set_title("NIR Raw Spectra (all samples)")
    ax_nir.set_xlabel("Wavelength idx")
    ax_nir.set_ylabel("Intensity")

    save_path = os.path.join(FIGURES_DIR, "raw_spectra_overview.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
