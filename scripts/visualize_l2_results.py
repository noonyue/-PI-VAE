"""
Visualize Table 3: L2 Manufacturer Identification Results.

Creates comprehensive visualization of L2 classification performance including:
- Test accuracy per drug
- Best model selection
- Model performance comparison (SVM/RF/PLS)
- Sample size information
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    create_multi_panel_figure, add_panel_label, format_axes,
    add_diagonal_line, COLOR_PRED, COLOR_DEFAULT, COLOR_TUNED,
    COLOR_OTHER, COLOR_TRUE
)


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)


def create_l2_results_visualization():
    """Create comprehensive visualization for L2 classification results"""
    ensure_dirs()
    
    # Load data
    df = pd.read_csv("results/l2_classification_results.csv")
    
    # Create figure with subplots using reference style
    fig, gs = create_multi_panel_figure(nrows=2, ncols=3, figsize=(18, 12), 
                                        hspace=0.35, wspace=0.35)
    
    # ========== Subplot 1: Test Accuracy per Drug (Bar Chart) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    drugs = df["Drug"].values
    test_accs = df["Test_Accuracy"].values * 100
    best_models = df["Best_Model"].values
    
    # Color by best model - adapted to reference style
    colors = {"RandomForest": COLOR_DEFAULT, "SVM": COLOR_PRED, "PLS-DA": COLOR_OTHER}
    bar_colors = [colors.get(model, "#6C757D") for model in best_models]
    
    bars = ax1.barh(drugs, test_accs, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    format_axes(ax1, xlabel="Test Accuracy (%)", ylabel="Drug",
               title="L2 Test Accuracy per Drug\n(Color: Best Model)")
    ax1.set_xlim([75, 105])
    ax1.axvline(x=df["Test_Accuracy"].mean() * 100, color=COLOR_TRUE, linestyle="--", 
                linewidth=2, label=f"Average: {df['Test_Accuracy'].mean()*100:.2f}%")
    ax1.legend(fontsize=10, loc="lower right")
    add_panel_label(ax1, "(a)", x_offset=-0.12, y_offset=1.02)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ========== Subplot 2: Model Performance Comparison (Grouped Bar) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(drugs))
    width = 0.25
    
    svm_accs = df["SVM_Acc"].values * 100
    rf_accs = df["RF_Acc"].values * 100
    pls_accs = df["PLS_Acc"].values * 100
    
    bars1 = ax2.bar(x - width, svm_accs, width, label="SVM", 
                    color=COLOR_PRED, alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax2.bar(x, rf_accs, width, label="RandomForest", 
                    color=COLOR_DEFAULT, alpha=0.8, edgecolor="black", linewidth=1.5)
    bars3 = ax2.bar(x + width, pls_accs, width, label="PLS-DA", 
                    color=COLOR_OTHER, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    format_axes(ax2, xlabel="Drug", ylabel="LOOCV Accuracy (%)",
               title="Model Performance Comparison\n(LOOCV Accuracy)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(drugs, rotation=45, ha="right")
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=10, loc="upper left")
    add_panel_label(ax2, "(b)", x_offset=-0.12, y_offset=1.02)
    
    # Highlight best model for each drug
    for i, (drug, best_model) in enumerate(zip(drugs, best_models)):
        if best_model == "SVM":
            bars1[i].set_edgecolor("gold")
            bars1[i].set_linewidth(3)
        elif best_model == "RandomForest":
            bars2[i].set_edgecolor("gold")
            bars2[i].set_linewidth(3)
        elif best_model == "PLS-DA":
            bars3[i].set_edgecolor("gold")
            bars3[i].set_linewidth(3)
    
    # ========== Subplot 3: CV Score vs Test Accuracy (Scatter) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    cv_scores = df["CV_Score"].values * 100
    test_accs = df["Test_Accuracy"].values * 100
    
    # Color by best model
    scatter_colors = [colors.get(model, "#6C757D") for model in best_models]
    
    for i, (drug, cv, test, color) in enumerate(zip(drugs, cv_scores, test_accs, scatter_colors)):
        ax3.scatter(cv, test, s=200, c=color, alpha=0.7, edgecolor="black", linewidth=1.5, zorder=3)
        ax3.annotate(drug, (cv, test), xytext=(5, 5), textcoords="offset points", 
                    fontsize=9, fontweight="bold")
    
    # Perfect prediction line (diagonal)
    add_diagonal_line(ax3, xlim=[80, 105], ylim=[80, 105], 
                     color=COLOR_TRUE, label="Perfect Prediction")
    format_axes(ax3, xlabel="CV Score (%)", ylabel="Test Accuracy (%)",
               title="CV Score vs Test Accuracy\n(Generalization Assessment)")
    ax3.set_xlim([80, 105])
    ax3.set_ylim([80, 105])
    ax3.legend(fontsize=10)
    add_panel_label(ax3, "(c)", x_offset=-0.12, y_offset=1.02)
    
    # ========== Subplot 4: Sample Size vs Accuracy ==========
    ax4 = fig.add_subplot(gs[1, 0])
    train_samples = df["Train_Samples"].values
    test_samples = df["Test_Samples"].values
    total_samples = train_samples + test_samples
    
    scatter_colors = [colors.get(model, "#6C757D") for model in best_models]
    
    for i, (drug, total, acc, color) in enumerate(zip(drugs, total_samples, test_accs, scatter_colors)):
        ax4.scatter(total, acc, s=200, c=color, alpha=0.7, edgecolor="black", linewidth=1.5, zorder=3)
        ax4.annotate(drug, (total, acc), xytext=(5, 5), textcoords="offset points", 
                    fontsize=9, fontweight="bold")
    
    format_axes(ax4, xlabel="Total Samples (Train + Test)", ylabel="Test Accuracy (%)",
               title="Sample Size vs Test Accuracy")
    ax4.set_ylim([75, 105])
    add_panel_label(ax4, "(d)", x_offset=-0.12, y_offset=1.02)
    
    # ========== Subplot 5: Model Selection Distribution ==========
    ax5 = fig.add_subplot(gs[1, 1])
    model_counts = df["Best_Model"].value_counts()
    pie_colors = [colors.get(model, "#6C757D") for model in model_counts.index]
    
    wedges, texts, autotexts = ax5.pie(model_counts.values, labels=model_counts.index, 
                                       autopct='%1.1f%%', colors=pie_colors,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Add count labels
    for i, (wedge, count) in enumerate(zip(wedges, model_counts.values)):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = 0.6 * np.cos(np.radians(angle))
        y = 0.6 * np.sin(np.radians(angle))
        ax5.text(x, y, f'n={count}', ha='center', va='center', 
                fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', alpha=0.8))
    
    format_axes(ax5, title="Best Model Selection Distribution")
    add_panel_label(ax5, "(e)", x_offset=-0.12, y_offset=1.02)
    
    # ========== Subplot 6: Performance Summary Table ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary statistics
    summary_data = {
        "Metric": [
            "Average Test Accuracy",
            "Max Test Accuracy",
            "Min Test Accuracy",
            "Std Test Accuracy",
            "Drugs with 100% Accuracy",
            "Average CV Score",
            "RF Selected",
            "SVM Selected"
        ],
        "Value": [
            f"{df['Test_Accuracy'].mean()*100:.2f}%",
            f"{df['Test_Accuracy'].max()*100:.2f}%",
            f"{df['Test_Accuracy'].min()*100:.2f}%",
            f"{df['Test_Accuracy'].std()*100:.2f}%",
            f"{(df['Test_Accuracy'] == 1.0).sum()}/9",
            f"{df['CV_Score'].mean()*100:.2f}%",
            f"{(df['Best_Model'] == 'RandomForest').sum()}/9",
            f"{(df['Best_Model'] == 'SVM').sum()}/9"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    table = ax6.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor("#4A90E2")
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data cells
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")
            else:
                table[(i, j)].set_facecolor("white")
    
    format_axes(ax6, title="Performance Summary Statistics")
    add_panel_label(ax6, "(f)", x_offset=-0.12, y_offset=1.02)
    
    # Main title
    fig.suptitle("Table 3: L2 Manufacturer Identification Results - Comprehensive Visualization", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    plt.savefig("figures/table3_l2_results_visualization.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/table3_l2_results_visualization.png")
    plt.close()


if __name__ == "__main__":
    create_l2_results_visualization()

