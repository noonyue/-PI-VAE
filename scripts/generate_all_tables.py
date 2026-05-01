#!/usr/bin/env python3
"""
Generate all 8 tables (Table 3.1 - 3.8) as PNG images
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# Configure Chinese font
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def create_table_image(data, headers, title, filename, col_widths=None, highlight_rows=None):
    """Generic table image generator"""
    fig, ax = plt.subplots(figsize=(12, 2 + len(data) * 0.6))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    if col_widths is None:
        col_widths = [1.0 / len(headers)] * len(headers)
    
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', 
                     loc='center', colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Header styling
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # Data row styling
    colors = ['#F5F5F5', '#FFFFFF'] * (len(data) // 2 + 1)
    for i in range(len(data)):
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            
            # Highlight specific rows
            if highlight_rows and i in highlight_rows:
                cell.set_facecolor('#FFF3E0')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor(colors[i])
            
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1.0)
    
    plt.title(title, fontsize=13, fontweight='bold', pad=20, loc='center')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=180, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: figures/{filename}")
    plt.close()

# ========== Table 3.1: Clustering Metrics ==========
df = pd.read_csv('results/2-clustering_metrics.csv')
data_31 = [
    ['Raw（SNV）', '0.434', '1.435', '149.5'],
    ['PCA（64维）', '0.440', '1.427', '150.3'],
    ['PI-VAE（64维）', '0.414', '1.678', '789.1'],
]
headers_31 = ['特征表示', 'Silhouette ↑', 'DBI ↓', 'CHI ↑']
create_table_image(data_31, headers_31, '表 3.1  不同特征表示的聚类质量指标对比',
                   'table3_1_clustering_metrics.png', 
                   col_widths=[0.35, 0.22, 0.22, 0.21],
                   highlight_rows=[2])

# ========== Table 3.2: L1 Performance ==========
data_32 = [
    ['SVM', '100.00%', '95.83%', '100.00%', '94.41%'],
    ['Random Forest', '100.00%', '97.22%', '100.00%', '96.30%'],
    ['PLS-DA', '69.44%', '48.61%', '63.49%', '38.03%'],
]
headers_32 = ['模型', 'Raw Accuracy', 'Latent Accuracy', 'Raw Macro F1', 'Latent Macro F1']
create_table_image(data_32, headers_32, '表 3.2  L1 药物分类性能汇总',
                   'table3_2_l1_performance.png',
                   col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                   highlight_rows=[0, 1])

# ========== Table 3.3: L2 Direct Classification ==========
data_33 = [
    ['SVM', '98.61%', '94.44%', '99.01%', '92.59%'],
    ['Random Forest', '97.22%', '95.83%', '94.61%', '94.10%'],
    ['PLS-DA', '26.39%', '26.39%', '22.02%', '15.93%'],
]
headers_33 = ['模型', 'Raw Accuracy', 'Latent Accuracy', 'Raw Macro F1', 'Latent Macro F1']
create_table_image(data_33, headers_33, '表 3.3  L2 厂家识别性能汇总（直接28类分类）',
                   'table3_3_l2_direct.png',
                   col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                   highlight_rows=[0])

# ========== Table 3.4: Prior RMSE ==========
data_34 = [
    ['Lorentzian（NIR-VAE）', '0.714', '0.739', '0.061'],
    ['Gaussian（消融对照）', '0.697', '0.719', '0.061'],
]
headers_34 = ['解码器', 'RMSE 均值', 'RMSE 中位数', 'RMSE 标准差']
create_table_image(data_34, headers_34, '表 3.4  NIR 解码器消融：Lorentzian vs Gaussian RMSE（测试集）',
                   'table3_4_prior_rmse.png',
                   col_widths=[0.40, 0.20, 0.20, 0.20])

# ========== Table 3.5: Cascade Ablation ==========
data_35 = [
    ['直接 28 类分类（SVM, Raw）', '61.11%', '—'],
    ['标准 AE + L1 预分类', '93.06%', '+31.95 pp'],
    ['Gaussian VAE + L1 预分类', '62.50%', '+1.39 pp'],
    ['PI-VAE 级联系统（本方法）', '97.45%', '+36.34 pp'],
]
headers_35 = ['方法', 'L2 Accuracy', '相对基线提升']
create_table_image(data_35, headers_35, '表 3.5  级联分类消融实验结果',
                   'table3_5_cascade_ablation.png',
                   col_widths=[0.50, 0.25, 0.25],
                   highlight_rows=[3])

# ========== Table 3.6: L2 Per-Drug Results ==========
df_l2 = pd.read_csv('results/4-l2_classification_results.csv')
data_36 = []
for _, row in df_l2.iterrows():
    acc_str = f"{row['Test_Accuracy']*100:.2f}%" if row['Test_Accuracy'] < 1.0 else "100.0%"
    data_36.append([
        row['Drug'], row['Best_Model'], f"{row['CV_Score']:.3f}",
        acc_str, str(row['Train_Samples']), str(row['Test_Samples'])
    ])
headers_36 = ['药品', '最优模型', 'CV 得分', '测试准确率', '训练样本数', '测试样本数']
create_table_image(data_36, headers_36, '表 3.6  L2 逐药级联分类详细结果',
                   'table3_6_l2_per_drug.png',
                   col_widths=[0.15, 0.18, 0.17, 0.17, 0.17, 0.16])

# ========== Table 3.7: SNR Stress Test ==========
data_37 = [
    ['50', '62.50%', '98.61%', '0.63×'],
    ['40', '62.50%', '98.61%', '0.63×'],
    ['30', '62.50%', '100.0%', '0.63×'],
    ['20', '61.11%', '12.50%', '4.89×'],
    ['10', '56.94%', '8.33%', '6.83×'],
]
headers_37 = ['SNR (dB)', 'PI-VAE Accuracy', 'Raw+SVM Accuracy', 'PI-VAE 优势倍数']
create_table_image(data_37, headers_37, '表 3.7  SNR 压力测试：PI-VAE vs Raw+SVM 准确率对比',
                   'table3_7_snr_stress.png',
                   col_widths=[0.20, 0.27, 0.27, 0.26],
                   highlight_rows=[3, 4])

# ========== Table 3.8: OOD Detection ==========
data_38 = [
    ['AUROC', '1.000'],
    ['最优决策阈值', '1.697'],
    ['真正率（TPR）@ 最优阈值', '100.0%'],
    ['假正率（FPR）@ 最优阈值', '0.0%'],
]
headers_38 = ['指标', '数值']
create_table_image(data_38, headers_38, '表 3.8  OOD 检测性能指标',
                   'table3_8_ood_metrics.png',
                   col_widths=[0.60, 0.40],
                   highlight_rows=[0, 2])

print("\n[✓] All 8 tables generated successfully!")
