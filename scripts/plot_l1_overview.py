"""
Plot L1 drug classification performance overview.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    os.makedirs("figures", exist_ok=True)
    
    df = pd.read_csv("results/model_comparison_l1.csv")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Build labels
    labels = []
    colors = []
    for _, row in df.iterrows():
        if row["Model"] == "PI-VAE+SVM (Cascade)":
            lbl = f"{row['Model']} ({row['Feature']})"
            colors.append("#DD8452")  # Highlight cascade
        else:
            lbl = f"{row['Model']} ({row['Feature']})"
            colors.append("#4C72B0" if row["Feature"] == "Raw" else "#55A868")
        labels.append(lbl)
    
    df["Label"] = labels
    
    # Sort by accuracy
    df_sorted = df.sort_values("Accuracy", ascending=True)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted["Accuracy"], 
                   color=[colors[df.index.get_loc(idx)] for idx in df_sorted.index])
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["Label"])
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_title("L1 Drug Classification Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, df_sorted["Accuracy"])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join("figures", "l1_overview_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
