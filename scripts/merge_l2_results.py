"""
Merge L2 evaluation results into a single overview CSV for tables/figures.

Inputs:
- results/model_comparison_l2_direct_classic.csv
- results/model_comparison_l2_cascade_summary.csv

Output:
- results/model_comparison_l2_overview.csv
"""

import os
import pandas as pd


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    direct_path = os.path.join("results", "model_comparison_l2_direct_classic.csv")
    cascade_path = os.path.join("results", "model_comparison_l2_cascade_summary.csv")

    direct_df = pd.read_csv(direct_path)
    cascade_df = pd.read_csv(cascade_path)

    # Select key rows for direct methods
    keep_models = [
        ("PLS-DA", "Raw"),
        ("PLS-DA", "Latent"),
        ("SVM", "Raw"),
        ("SVM", "Latent"),
        ("RandomForest", "Raw"),
        ("RandomForest", "Latent"),
    ]
    rows = []
    for model, feat in keep_models:
        row = direct_df[(direct_df["Model"] == model) & (direct_df["Feature"] == feat)].iloc[0]
        rows.append(
            {
                "Strategy": "Direct",
                "Model": model,
                "Feature": feat,
                "Accuracy": row["Accuracy"],
                "Macro_F1": row["Macro_F1"],
            }
        )

    # Add cascade summary (RF + Latent)
    cas = cascade_df.iloc[0]
    rows.append(
        {
            "Strategy": "Cascade",
            "Model": cas["Model"],
            "Feature": cas["Feature"],
            "Accuracy": cas["Overall_Accuracy"],
            "Macro_F1": cas["Overall_Macro_F1"],
        }
    )

    out_df = pd.DataFrame(rows)
    out_path = os.path.join("results", "model_comparison_l2_overview.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved L2 overview to: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

