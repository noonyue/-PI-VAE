"""
Visualize preprocessing effect (Raw vs SNV) and quantify data quality.
Outputs:
- figures/preprocessing_effect.png
- results/data_quality_metrics.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_vae_pipeline import load_data, preprocess_spectra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    COLOR_TRUE, COLOR_PRED, COLOR_GRID
)


def ensure_dirs():
    # Use output_path_helper if available
    try:
        from output_path_helper import get_figures_dir, get_results_dir
        os.makedirs(get_figures_dir(), exist_ok=True)
        os.makedirs(get_results_dir(), exist_ok=True)
    except ImportError:
        os.makedirs("figures", exist_ok=True)
        os.makedirs("results", exist_ok=True)


def baseline_drift(spectra):
    """Simple baseline drift metric: std over sample-wise means."""
    means = spectra.mean(axis=1)
    return np.std(means)


def intra_class_variance(spectra, labels):
    """Average within-class variance."""
    vars_per_class = []
    for cls in np.unique(labels):
        cls_data = spectra[labels == cls]
        if len(cls_data) > 1:
            vars_per_class.append(np.mean(np.var(cls_data, axis=0)))
    return float(np.mean(vars_per_class)) if vars_per_class else np.nan


def main():
    ensure_dirs()
    uv_raw, nir_raw, drug_labels, _ = load_data()

    uv_snv = preprocess_spectra(uv_raw, method="snv")
    nir_snv = preprocess_spectra(nir_raw, method="snv")

    # Plot raw vs SNV (UV for visualization) with reference style
    fig, gs = create_multi_panel_figure(nrows=2, ncols=1, figsize=(10, 6), hspace=0.3)
    
    # Panel (a): Raw spectra
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(uv_raw.T, alpha=0.1, color=COLOR_TRUE)
    format_axes(ax1, xlabel="Wavelength idx", ylabel="Intensity", 
               title="UV-Vis Raw Spectra")
    add_panel_label(ax1, "(a)", x_offset=-0.12, y_offset=1.02)
    
    # Panel (b): SNV spectra
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(uv_snv.T, alpha=0.1, color=COLOR_PRED)
    format_axes(ax2, xlabel="Wavelength idx", ylabel="Intensity (SNV)", 
               title="UV-Vis SNV Spectra")
    add_panel_label(ax2, "(b)", x_offset=-0.12, y_offset=1.02)
    
    try:
        from output_path_helper import get_figure_path
        save_path = get_figure_path("preprocessing_effect.png")
    except ImportError:
        save_path = "figures/preprocessing_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Metrics
    metrics = []
    for name, data in [("UV_Raw", uv_raw), ("UV_SNV", uv_snv), ("NIR_Raw", nir_raw), ("NIR_SNV", nir_snv)]:
        metrics.append(
            {
                "Name": name,
                "Baseline_Drift": baseline_drift(data),
                "Intra_Class_Var": intra_class_variance(data, drug_labels),
            }
        )
    df = pd.DataFrame(metrics)

    # Improvement rows (UV, NIR)
    def improvement(raw_name, snv_name, metric):
        raw = df.loc[df["Name"] == raw_name, metric].values[0]
        snv = df.loc[df["Name"] == snv_name, metric].values[0]
        return 100 * (raw - snv) / raw if raw != 0 else np.nan

    df_improve = pd.DataFrame(
        [
            {
                "Metric": "Baseline_Drift_UV",
                "Raw_Value": df.loc[df["Name"] == "UV_Raw", "Baseline_Drift"].values[0],
                "SNV_Value": df.loc[df["Name"] == "UV_SNV", "Baseline_Drift"].values[0],
                "Improvement_%": improvement("UV_Raw", "UV_SNV", "Baseline_Drift"),
            },
            {
                "Metric": "Baseline_Drift_NIR",
                "Raw_Value": df.loc[df["Name"] == "NIR_Raw", "Baseline_Drift"].values[0],
                "SNV_Value": df.loc[df["Name"] == "NIR_SNV", "Baseline_Drift"].values[0],
                "Improvement_%": improvement("NIR_Raw", "NIR_SNV", "Baseline_Drift"),
            },
            {
                "Metric": "Intra_Class_Var_UV",
                "Raw_Value": df.loc[df["Name"] == "UV_Raw", "Intra_Class_Var"].values[0],
                "SNV_Value": df.loc[df["Name"] == "UV_SNV", "Intra_Class_Var"].values[0],
                "Improvement_%": improvement("UV_Raw", "UV_SNV", "Intra_Class_Var"),
            },
            {
                "Metric": "Intra_Class_Var_NIR",
                "Raw_Value": df.loc[df["Name"] == "NIR_Raw", "Intra_Class_Var"].values[0],
                "SNV_Value": df.loc[df["Name"] == "NIR_SNV", "Intra_Class_Var"].values[0],
                "Improvement_%": improvement("NIR_Raw", "NIR_SNV", "Intra_Class_Var"),
            },
        ]
    )

    try:
        from output_path_helper import get_result_path
        out_path = get_result_path("data_quality_metrics.csv")
    except ImportError:
        out_path = "results/data_quality_metrics.csv"
    df_improve.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

