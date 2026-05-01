"""
重新生成所有用于论文的图片和表格（融合特征级联模型）
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("tables_for_paper", exist_ok=True)

def generate_table_l1_paper():
    """生成L1对比表（论文格式）"""
    df = pd.read_csv("results/model_comparison_l1.csv")
    
    # 选择关键模型
    key_models = [
        ("PLS-DA", "Raw"),
        ("SVM", "Raw"),
        ("RandomForest", "Raw"),
        ("CNN", "Raw"),
        ("Transformer", "Raw"),
        ("PI-VAE+SVM (Cascade)", "Fused"),
    ]
    
    rows = []
    for model, feat in key_models:
        row = df[(df["Model"] == model) & (df["Feature"] == feat)]
        if not row.empty:
            r = row.iloc[0]
            rows.append({
                "Model": model.replace("PI-VAE+SVM (Cascade)", "PI-VAE Cascade"),
                "Feature": feat,
                "Accuracy (%)": f"{r['Accuracy']*100:.2f}",
                "Macro-F1 (%)": f"{r['Macro_F1']*100:.2f}"
            })
    
    out_df = pd.DataFrame(rows)
    out_path = "tables_for_paper/Table1_L1_Drug_Classification.csv"
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 生成: {out_path}")
    
    # 同时生成LaTeX格式
    latex_path = "tables_for_paper/Table1_L1_Drug_Classification.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{L1 Drug Classification Performance Comparison}\n")
        f.write("\\label{tab:l1_comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Feature & Accuracy (\\%) & Macro-F1 (\\%) \\\\\n")
        f.write("\\midrule\n")
        for _, r in out_df.iterrows():
            f.write(f"{r['Model']} & {r['Feature']} & {r['Accuracy (%)']} & {r['Macro-F1 (%)']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] 生成: {latex_path}")
    
    return out_df

def generate_table_l2_paper():
    """生成L2对比表（论文格式）"""
    df = pd.read_csv("results/model_comparison_l2_overview.csv")
    
    # 选择关键方法
    key_methods = [
        ("Direct", "PLS-DA", "Raw"),
        ("Direct", "SVM", "Raw"),
        ("Direct", "RandomForest", "Raw"),
        ("Cascade", "RandomForest", "Fused"),
    ]
    
    rows = []
    for strat, model, feat in key_methods:
        row = df[(df["Strategy"] == strat) & 
                 (df["Model"] == model) & 
                 (df["Feature"] == feat)]
        if not row.empty:
            r = row.iloc[0]
            rows.append({
                "Strategy": strat,
                "Model": model,
                "Feature": feat,
                "Accuracy (%)": f"{r['Accuracy']*100:.2f}",
                "Macro-F1 (%)": f"{r['Macro_F1']*100:.2f}"
            })
    
    out_df = pd.DataFrame(rows)
    out_path = "tables_for_paper/Table2_L2_Manufacturer_Classification.csv"
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 生成: {out_path}")
    
    # LaTeX格式
    latex_path = "tables_for_paper/Table2_L2_Manufacturer_Classification.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{L2 Manufacturer Classification: Direct vs Cascade}\n")
        f.write("\\label{tab:l2_comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Strategy & Model & Feature & Accuracy (\\%) & Macro-F1 (\\%) \\\\\n")
        f.write("\\midrule\n")
        for _, r in out_df.iterrows():
            f.write(f"{r['Strategy']} & {r['Model']} & {r['Feature']} & {r['Accuracy (%)']} & {r['Macro-F1 (%)']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] 生成: {latex_path}")
    
    return out_df

def generate_table_l2_per_drug_paper():
    """生成L2 per-drug表（论文格式）"""
    df = pd.read_csv("results/model_comparison_l2_cascade_per_drug.csv")
    
    # 添加药品名称映射（根据实际情况调整）
    drug_names = {0: "CIM", 1: "FMD", 2: "GLD", 3: "GSR", 4: "HCT", 
                  5: "IBU", 6: "MHE", 7: "MHL", 8: "MHR"}
    
    df["Drug_Name"] = df["Drug"].map(drug_names)
    df["Accuracy (%)"] = (df["Accuracy"] * 100).round(2)
    df["Macro-F1 (%)"] = (df["Macro_F1"] * 100).round(2)
    
    out_df = df[["Drug_Name", "Train_Samples", "Test_Samples", "Accuracy (%)", "Macro-F1 (%)"]]
    out_path = "tables_for_paper/Table3_L2_Cascade_Per_Drug.csv"
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 生成: {out_path}")
    
    # LaTeX格式
    latex_path = "tables_for_paper/Table3_L2_Cascade_Per_Drug.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{L2 Cascade Manufacturer Classification: Per-Drug Performance}\n")
        f.write("\\label{tab:l2_per_drug}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Drug & Train & Test & Accuracy (\\%) & Macro-F1 (\\%) \\\\\n")
        f.write("\\midrule\n")
        for _, r in out_df.iterrows():
            f.write(f"{r['Drug_Name']} & {int(r['Train_Samples'])} & {int(r['Test_Samples'])} & {r['Accuracy (%)']} & {r['Macro-F1 (%)']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] 生成: {latex_path}")
    
    return out_df

def generate_figure_l1_comparison():
    """生成L1对比图（论文质量）"""
    df = pd.read_csv("results/model_comparison_l1.csv")
    
    # 选择关键模型
    key_models = [
        ("PLS-DA", "Raw"),
        ("SVM", "Raw"),
        ("RandomForest", "Raw"),
        ("CNN", "Raw"),
        ("Transformer", "Raw"),
        ("PI-VAE+SVM (Cascade)", "Fused"),
    ]
    
    plot_data = []
    for model, feat in key_models:
        row = df[(df["Model"] == model) & (df["Feature"] == feat)]
        if not row.empty:
            r = row.iloc[0]
            label = f"{model.replace('PI-VAE+SVM (Cascade)', 'PI-VAE Cascade')} ({feat})"
            plot_data.append({
                "Label": label,
                "Accuracy": r["Accuracy"],
                "Model": model,
                "Feature": feat
            })
    
    plot_df = pd.DataFrame(plot_data).sort_values("Accuracy", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 颜色映射
    colors = []
    for _, row in plot_df.iterrows():
        if "PI-VAE Cascade" in row["Label"]:
            colors.append("#DD8452")  # 橙色突出级联方法
        elif row["Feature"] == "Raw":
            colors.append("#4C72B0")  # 蓝色
        else:
            colors.append("#55A868")  # 绿色
    
    bars = ax.barh(range(len(plot_df)), plot_df["Accuracy"], color=colors)
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Label"], fontsize=10)
    ax.set_xlabel("Test Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("L1 Drug Classification Performance Comparison", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, plot_df["Accuracy"])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    out_path = "figures/Figure_L1_Comparison_Paper.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] 生成: {out_path}")
    plt.close()

def generate_figure_l2_comparison():
    """生成L2对比图（论文质量）"""
    df = pd.read_csv("results/model_comparison_l2_overview.csv")
    per_drug = pd.read_csv("results/model_comparison_l2_cascade_per_drug.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (A) Direct vs Cascade 对比
    ax1 = axes[0]
    
    # 选择关键方法
    key_methods = [
        ("Direct", "PLS-DA", "Raw"),
        ("Direct", "SVM", "Raw"),
        ("Direct", "RandomForest", "Raw"),
        ("Cascade", "RandomForest", "Fused"),
    ]
    
    plot_data = []
    for strat, model, feat in key_methods:
        row = df[(df["Strategy"] == strat) & 
                 (df["Model"] == model) & 
                 (df["Feature"] == feat)]
        if not row.empty:
            r = row.iloc[0]
            if strat == "Cascade":
                label = f"Cascade RF (Fused)"
            else:
                label = f"{model} ({feat})"
            plot_data.append({
                "Label": label,
                "Accuracy": r["Accuracy"],
                "Strategy": strat
            })
    
    plot_df = pd.DataFrame(plot_data).sort_values("Accuracy", ascending=True)
    
    colors = ["#4C72B0" if s == "Direct" else "#DD8452" for s in plot_df["Strategy"]]
    bars1 = ax1.barh(range(len(plot_df)), plot_df["Accuracy"], color=colors)
    
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df["Label"], fontsize=10)
    ax1.set_xlabel("Test Accuracy", fontsize=11, fontweight='bold')
    ax1.set_title("(A) L2: Direct 28-class vs Cascade", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 1.05)
    ax1.grid(axis="x", alpha=0.3, linestyle='--')
    
    for i, (bar, acc) in enumerate(zip(bars1, plot_df["Accuracy"])):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='Direct'),
        Patch(facecolor='#DD8452', label='Cascade')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # (B) Per-drug 精度
    ax2 = axes[1]
    
    drug_names = {0: "CIM", 1: "FMD", 2: "GLD", 3: "GSR", 4: "HCT", 
                  5: "IBU", 6: "MHE", 7: "MHL", 8: "MHR"}
    per_drug["Drug_Name"] = per_drug["Drug"].map(drug_names)
    per_drug_sorted = per_drug.sort_values("Accuracy", ascending=True)
    
    bars2 = ax2.barh(range(len(per_drug_sorted)), per_drug_sorted["Accuracy"], 
                     color="#55A868", alpha=0.8)
    
    ax2.set_yticks(range(len(per_drug_sorted)))
    ax2.set_yticklabels(per_drug_sorted["Drug_Name"], fontsize=10)
    ax2.set_xlabel("Test Accuracy", fontsize=11, fontweight='bold')
    ax2.set_title("(B) L2 Cascade per-drug accuracy (RF + Fused)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis="x", alpha=0.3, linestyle='--')
    
    for i, (bar, acc) in enumerate(zip(bars2, per_drug_sorted["Accuracy"])):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    out_path = "figures/Figure_L2_Comparison_Paper.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] 生成: {out_path}")
    plt.close()

def generate_combined_l1_l2_figure():
    """生成L1+L2综合对比图（2x2布局）"""
    df_l1_base = pd.read_csv("results/model_comparison_l1.csv")
    df_l1_fused = pd.read_csv("results/model_comparison_l1_cascade_fused.csv")
    df_l1 = pd.concat([df_l1_base, df_l1_fused], ignore_index=True)

    # L2 direct 数据合并
    df_l2_direct_classic = pd.read_csv("results/model_comparison_l2_direct_classic.csv")
    df_l2_direct_deep = pd.read_csv("results/model_comparison_l2_direct_deep.csv")
    df_l2_cascade = pd.read_csv("results/model_comparison_l2_cascade_summary.csv")
    per_drug = pd.read_csv("results/model_comparison_l2_cascade_per_drug.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) L1 对比
    ax1 = axes[0, 0]
    key_models_l1 = [
        ("PLS-DA", "Raw"),
        ("SVM", "Raw"),
        ("RandomForest", "Raw"),
        ("CNN", "Raw"),
        ("PI-VAE+SVM (Cascade)", "Fused"),
    ]

    plot_data_l1 = []
    for model, feat in key_models_l1:
        row = df_l1[(df_l1["Model"] == model) & (df_l1["Feature"] == feat)]
        if not row.empty:
            r = row.iloc[0]
            label = f"{model.replace('PI-VAE+SVM (Cascade)', 'PI-VAE Cascade')} ({feat})"
            plot_data_l1.append({"Label": label, "Accuracy": r["Accuracy"]})

    plot_df_l1 = pd.DataFrame(plot_data_l1).sort_values("Accuracy", ascending=True)
    colors_l1 = ["#DD8452" if "Cascade" in l else "#4C72B0" for l in plot_df_l1["Label"]]
    bars1 = ax1.barh(range(len(plot_df_l1)), plot_df_l1["Accuracy"], color=colors_l1)
    ax1.set_yticks(range(len(plot_df_l1)))
    ax1.set_yticklabels(plot_df_l1["Label"], fontsize=9)
    ax1.set_xlabel("Accuracy", fontsize=10)
    ax1.set_title("(A) L1 Drug Classification", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1.05)
    ax1.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars1, plot_df_l1["Accuracy"]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.1f}%', ha='left', va='center', fontsize=8)

    # (B) L2 Direct vs Cascade 对比
    ax2 = axes[0, 1]
    # 从 direct_classic 取 SVM/RF(Raw)，cascade 取总体准确率
    plot_data_l2 = []
    for _, row in df_l2_direct_classic.iterrows():
        if row["Feature"] == "Raw" and row["Model"] in ("SVM", "RandomForest"):
            plot_data_l2.append({"Label": f"Direct {row['Model']} (Raw)", "Accuracy": row["Accuracy"]})
    # 级联结果
    if not df_l2_cascade.empty:
        casc_acc = df_l2_cascade.iloc[0]["Overall_Accuracy"]
        plot_data_l2.append({"Label": "Cascade RF (Fused)", "Accuracy": casc_acc})

    plot_df_l2 = pd.DataFrame(plot_data_l2).sort_values("Accuracy", ascending=True)
    colors_l2 = ["#DD8452" if "Cascade" in l else "#4C72B0" for l in plot_df_l2["Label"]]
    bars2 = ax2.barh(range(len(plot_df_l2)), plot_df_l2["Accuracy"], color=colors_l2)
    ax2.set_yticks(range(len(plot_df_l2)))
    ax2.set_yticklabels(plot_df_l2["Label"], fontsize=9)
    ax2.set_xlabel("Accuracy", fontsize=10)
    ax2.set_title("(B) L2 Manufacturer Classification", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars2, plot_df_l2["Accuracy"]):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.1f}%', ha='left', va='center', fontsize=8)

    # (C) L2 Per-drug 条形图
    ax3 = axes[1, 0]
    drug_names = {0: "CIM", 1: "FMD", 2: "GLD", 3: "GSR", 4: "HCT",
                  5: "IBU", 6: "MHE", 7: "MHL", 8: "MHR"}
    per_drug["Drug_Name"] = per_drug["Drug"].map(drug_names)
    per_drug_sorted = per_drug.sort_values("Accuracy", ascending=True)

    bars3 = ax3.barh(range(len(per_drug_sorted)), per_drug_sorted["Accuracy"], color="#55A868")
    ax3.set_yticks(range(len(per_drug_sorted)))
    ax3.set_yticklabels(per_drug_sorted["Drug_Name"], fontsize=9)
    ax3.set_xlabel("Accuracy", fontsize=10)
    ax3.set_title("(C) L2 Cascade per-drug accuracy", fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 1.05)
    ax3.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars3, per_drug_sorted["Accuracy"]):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.1f}%', ha='left', va='center', fontsize=8)

    # (D) 性能总结对比柱状图
    ax4 = axes[1, 1]
    cascade_l1_acc = df_l1[(df_l1["Model"] == "PI-VAE+SVM (Cascade)") &
                           (df_l1["Feature"] == "Fused")]["Accuracy"].iloc[0]
    cascade_l2_acc = df_l2_cascade.iloc[0]["Overall_Accuracy"]
    best_direct_l2 = df_l2_direct_classic[df_l2_direct_classic["Feature"] == "Raw"]["Accuracy"].max()
    best_direct_l1 = df_l1_base[df_l1_base["Feature"] == "Raw"]["Accuracy"].max()

    metrics = ["L1 Drug\nAccuracy", "L2 Manufacturer\nAccuracy"]
    cascade_values = [cascade_l1_acc, cascade_l2_acc]
    direct_values = [best_direct_l1, best_direct_l2]

    x = np.arange(len(metrics))
    width = 0.35

    bars4a = ax4.bar(x - width/2, cascade_values, width, label='PI-VAE Cascade', color='#DD8452')
    bars4b = ax4.bar(x + width/2, direct_values, width, label='Best Direct', color='#4C72B0', alpha=0.7)

    ax4.set_ylabel("Accuracy", fontsize=10)
    ax4.set_title("(D) Performance Summary", fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis="y", alpha=0.3)
    for bar in list(bars4a) + list(bars4b):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height()*100:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = "figures/Figure_Combined_L1_L2_Paper.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] 生成: {out_path}")
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    ensure_dirs()
    
    print("\n" + "="*60)
    print("重新生成所有论文用图片和表格（融合特征级联模型）")
    print("="*60)
    
    # 生成表格
    print("\n【生成表格】")
    generate_table_l1_paper()
    generate_table_l2_paper()
    generate_table_l2_per_drug_paper()
    
    # 生成图片
    print("\n【生成图片】")
    generate_figure_l1_comparison()
    generate_figure_l2_comparison()
    generate_combined_l1_l2_figure()
    
    print("\n" + "="*60)
    print("所有文件生成完成！")
    print("="*60)
    print("\n生成的文件位置：")
    print("  表格: tables_for_paper/")
    print("  图片: figures/Figure_*_Paper.png")
    print("="*60)

if __name__ == "__main__":
    main()
