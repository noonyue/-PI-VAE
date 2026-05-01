"""
Create comparison figure for L1 drug classification and L2 manufacturer identification performance.

Outputs:
- figures/l1_l2_performance_comparison.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)


def load_l1_results():
    """Load L1 classification results from ablation_study.csv"""
    df = pd.read_csv("results/ablation_study.csv")
    l1_acc = df[df["Method"] == "PI-VAE (L1)"]["Accuracy"].values[0]
    return l1_acc


def load_l2_results():
    """Load L2 classification results"""
    df = pd.read_csv("results/l2_classification_results.csv")
    return df


def create_comparison_figure():
    """Create L1 vs L2 performance comparison figure"""
    ensure_dirs()
    
    # Load data
    l1_acc = load_l1_results()
    l2_df = load_l2_results()
    
    # Calculate L2 average accuracy
    l2_avg_acc = l2_df["Test_Accuracy"].mean()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Overall accuracy comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["L1: Drug\nClassification", "L2: Manufacturer\nIdentification"]
    accuracies = [l1_acc * 100, l2_avg_acc * 100]
    colors = ["#2E86AB", "#A23B72"]
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(A) Overall Classification Performance", fontsize=13, fontweight="bold", pad=15)
    ax1.set_ylim([85, 100])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 2: L2 per-drug accuracy (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    drugs = l2_df["Drug"].values
    l2_accs = l2_df["Test_Accuracy"].values * 100
    colors_drug = plt.cm.viridis(np.linspace(0.2, 0.8, len(drugs)))
    bars2 = ax2.barh(drugs, l2_accs, color=colors_drug, alpha=0.8, edgecolor="black", linewidth=1)
    ax2.set_xlabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title("(B) L2 Accuracy per Drug", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlim([75, 105])
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.axvline(x=l2_avg_acc * 100, color="red", linestyle="--", linewidth=2, label=f"Average: {l2_avg_acc*100:.2f}%")
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars2, l2_accs)):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}%', ha='left', va='center', fontsize=9)
    
    # Subplot 3: Model selection distribution (pie chart)
    ax3 = fig.add_subplot(gs[1, 0])
    model_counts = l2_df["Best_Model"].value_counts()
    colors_pie = {"RandomForest": "#2E86AB", "SVM": "#A23B72", "PLS-DA": "#F18F01"}
    pie_colors = [colors_pie.get(model, "#6C757D") for model in model_counts.index]
    wedges, texts, autotexts = ax3.pie(model_counts.values, labels=model_counts.index, 
                                       autopct='%1.1f%%', colors=pie_colors,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title("(C) Model Selection Distribution\n(Optimal Model per Drug)", 
                  fontsize=13, fontweight="bold", pad=15)
    
    # Subplot 4: Performance improvement (cascade vs direct)
    ax4 = fig.add_subplot(gs[1, 1])
    ablation_df = pd.read_csv("results/ablation_study.csv")
    direct_acc = ablation_df[ablation_df["Method"] == "Direct 28-class"]["Accuracy"].values[0] * 100
    cascade_acc = ablation_df[ablation_df["Method"] == "Cascade"]["Accuracy"].values[0] * 100
    
    methods = ["Direct\n28-class", "Cascade\n(L1+L2)"]
    accs = [direct_acc, cascade_acc]
    improvement = cascade_acc - direct_acc
    
    bars4 = ax4.bar(methods, accs, color=["#DC3545", "#28A745"], alpha=0.8, 
                    edgecolor="black", linewidth=1.5)
    ax4.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax4.set_title("(D) Cascade Strategy Advantage", fontsize=13, fontweight="bold", pad=15)
    ax4.set_ylim([50, 105])
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels
    for bar, acc in zip(bars4, accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    ax4.annotate('', xy=(1, cascade_acc), xytext=(0, direct_acc),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax4.text(0.5, (direct_acc + cascade_acc) / 2 + 2,
            f'+{improvement:.2f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Main title
    fig.suptitle("L1 Drug Classification vs L2 Manufacturer Identification Performance", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    plt.savefig("figures/l1_l2_performance_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/l1_l2_performance_comparison.png")
    plt.close()


if __name__ == "__main__":
    create_comparison_figure()

