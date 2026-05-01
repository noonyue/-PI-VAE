"""
生成论文用最终对比表格（LaTeX格式）
"""
import os
import pandas as pd

def format_percentage(val):
    """格式化百分比"""
    return f"{val*100:.2f}%"

def generate_latex_table_l1():
    """生成L1对比表的LaTeX代码"""
    df = pd.read_csv("results/model_comparison_l1.csv")
    
    # 选择关键模型进行展示
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
            rows.append({
                "Model": model,
                "Feature": feat,
                "Accuracy": row.iloc[0]["Accuracy"],
                "Macro_F1": row.iloc[0]["Macro_F1"]
            })
    
    print("\n" + "="*60)
    print("Table 1: L1 Drug Classification Performance")
    print("="*60)
    print("\nLaTeX Table Code:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{L1 Drug Classification Performance Comparison}")
    print("\\label{tab:l1_comparison}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Model & Feature & Accuracy & Macro-F1 \\\\")
    print("\\midrule")
    
    for r in rows:
        model_name = r["Model"].replace("PI-VAE+SVM (Cascade)", "PI-VAE Cascade")
        feat_name = r["Feature"]
        acc = format_percentage(r["Accuracy"])
        f1 = format_percentage(r["Macro_F1"])
        print(f"{model_name} & {feat_name} & {acc} & {f1} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_latex_table_l2():
    """生成L2对比表的LaTeX代码"""
    df = pd.read_csv("results/model_comparison_l2_overview.csv")
    
    # 选择关键方法
    key_methods = [
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
            rows.append({
                "Strategy": strat,
                "Model": model,
                "Feature": feat,
                "Accuracy": row.iloc[0]["Accuracy"],
                "Macro_F1": row.iloc[0]["Macro_F1"]
            })
    
    print("\n" + "="*60)
    print("Table 2: L2 Manufacturer Classification Performance")
    print("="*60)
    print("\nLaTeX Table Code:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{L2 Manufacturer Classification: Direct vs Cascade}")
    print("\\label{tab:l2_comparison}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Strategy & Model & Feature & Accuracy & Macro-F1 \\\\")
    print("\\midrule")
    
    for r in rows:
        strat = r["Strategy"]
        model = r["Model"]
        feat = r["Feature"]
        acc = format_percentage(r["Accuracy"])
        f1 = format_percentage(r["Macro_F1"])
        print(f"{strat} & {model} & {feat} & {acc} & {f1} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    
    generate_latex_table_l1()
    generate_latex_table_l2()
    
    print("\n" + "="*60)
    print("所有表格代码已生成")
    print("="*60)

if __name__ == "__main__":
    main()
