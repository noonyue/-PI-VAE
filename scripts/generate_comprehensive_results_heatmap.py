"""
生成全部模型结果的综合热图展示
每个子图展示一个模型结果的性能热图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置英文字体和样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 读取所有结果数据
l1_results = pd.read_csv('results/model_comparison_l1.csv')
l1_cascade = pd.read_csv('results/model_comparison_l1_cascade_fused.csv')
l2_direct = pd.read_csv('results/model_comparison_l2_direct_classic.csv')
l2_cascade_summary = pd.read_csv('results/model_comparison_l2_cascade_summary.csv')
l2_cascade_per_drug = pd.read_csv('results/model_comparison_l2_cascade_per_drug.csv')
ablation = pd.read_csv('results/3-ablation_study.csv')
robustness = pd.read_csv('results/8-robustness_stress.csv')
ood = pd.read_csv('results/9-ood_performance_metrics.csv')

# 创建大图：4行2列布局
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

# ========== 子图1: L1药物分类性能热图 ==========
ax1 = fig.add_subplot(gs[0, 0])
l1_pivot = l1_results.pivot_table(index='Model', columns='Feature', values='Accuracy')
sns.heatmap(l1_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Accuracy'}, ax=ax1, linewidths=0.5)
ax1.set_title('(a) L1 Drug Classification - Model×Feature Performance', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Feature Type', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)

# ========== 子图2: L1药物分类F1分数热图 ==========
ax2 = fig.add_subplot(gs[0, 1])
l1_f1_pivot = l1_results.pivot_table(index='Model', columns='Feature', values='Macro_F1')
sns.heatmap(l1_f1_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Macro-F1'}, ax=ax2, linewidths=0.5)
ax2.set_title('(b) L1 Drug Classification - Macro-F1 Heatmap', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Feature Type', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)

# ========== 子图3: L2直接分类性能热图 ==========
ax3 = fig.add_subplot(gs[1, 0])
l2_direct_pivot = l2_direct.pivot_table(index='Model', columns='Feature', values='Accuracy')
sns.heatmap(l2_direct_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Accuracy'}, ax=ax3, linewidths=0.5)
ax3.set_title('(c) L2 Manufacturer - Direct 28-class Classification', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Feature Type', fontsize=12)
ax3.set_ylabel('Model', fontsize=12)

# ========== 子图4: L2逐药物级联分类准确率热图 ==========
ax4 = fig.add_subplot(gs[1, 1])
# 创建9种药物的准确率矩阵（单列热图）
drug_acc = l2_cascade_per_drug[['Drug', 'Accuracy']].copy()
drug_acc['Drug'] = drug_acc['Drug'].astype(str)
drug_matrix = drug_acc.set_index('Drug')['Accuracy'].values.reshape(-1, 1)
sns.heatmap(drug_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            yticklabels=[f'Drug {i}' for i in range(9)],
            xticklabels=['Cascade Accuracy'],
            cbar_kws={'label': 'Accuracy'}, ax=ax4, linewidths=0.5)
ax4.set_title('(d) L2 Manufacturer - Per-Drug Cascade Accuracy', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('', fontsize=12)
ax4.set_ylabel('Drug Type', fontsize=12)

# ========== 子图5: 消融实验对比热图 ==========
ax5 = fig.add_subplot(gs[2, 0])
ablation_data = ablation.set_index('Method')['Accuracy'].values.reshape(-1, 1)
ablation_labels = ablation['Method'].values
sns.heatmap(ablation_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            yticklabels=ablation_labels,
            xticklabels=['Accuracy'],
            cbar_kws={'label': 'Accuracy'}, ax=ax5, linewidths=0.5)
ax5.set_title('(e) Ablation Study - Strategy Comparison', fontsize=14, fontweight='bold', pad=15)
ax5.set_xlabel('', fontsize=12)
ax5.set_ylabel('Method', fontsize=12)

# ========== 子图6: 鲁棒性测试热图（SNR vs 模型） ==========
ax6 = fig.add_subplot(gs[2, 1])
# 转置数据：行为SNR，列为模型
robustness_pivot = robustness.set_index('SNR_dB')
robustness_pivot.columns = ['PI-VAE', 'Raw+SVM']
sns.heatmap(robustness_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Accuracy'}, ax=ax6, linewidths=0.5)
ax6.set_title('(f) Robustness Test - SNR Noise Stress', fontsize=14, fontweight='bold', pad=15)
ax6.set_xlabel('Model', fontsize=12)
ax6.set_ylabel('SNR (dB)', fontsize=12)

# ========== 子图7: OOD检测性能指标 ==========
ax7 = fig.add_subplot(gs[3, 0])
# OOD只有一行数据，展示为单行热图
ood_data = ood[['AUC', 'Best_Threshold', 'TPR_at_best', 'FPR_at_best']].values
sns.heatmap(ood_data, annot=True, fmt='.3f', cmap='Blues',
            yticklabels=['OOD Detection'],
            xticklabels=['AUC', 'Best Threshold', 'TPR', 'FPR'],
            cbar_kws={'label': 'Value'}, ax=ax7, linewidths=0.5)
ax7.set_title('(g) OOD Detection Performance', fontsize=14, fontweight='bold', pad=15)
ax7.set_xlabel('Metric', fontsize=12)
ax7.set_ylabel('', fontsize=12)

# ========== 子图8: 综合性能对比（L1+L2+级联） ==========
ax8 = fig.add_subplot(gs[3, 1])
# 汇总关键结果
summary_data = {
    'L1 (SVM+Raw)': [1.0],
    'L1 (RF+Raw)': [1.0],
    'L1 (PI-VAE+SVM)': [1.0],
    'L2 Direct (SVM+Raw)': [0.986],
    'L2 Cascade (RF+Fused)': [0.972],
}
summary_df = pd.DataFrame(summary_data).T
summary_df.columns = ['Accuracy']
sns.heatmap(summary_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.9, vmax=1.0,
            cbar_kws={'label': 'Accuracy'}, ax=ax8, linewidths=0.5)
ax8.set_title('(h) Overall Performance - Best Model Summary', fontsize=14, fontweight='bold', pad=15)
ax8.set_xlabel('', fontsize=12)
ax8.set_ylabel('Model Configuration', fontsize=12)

# 添加总标题
fig.suptitle('PI-VAE Drug Spectral Analysis System - Comprehensive Results Heatmap',
             fontsize=18, fontweight='bold', y=0.995)

# 保存图片
plt.savefig('figures/comprehensive_results_heatmap.png', dpi=300, bbox_inches='tight')
print("Comprehensive results heatmap saved: figures/comprehensive_results_heatmap.png")
plt.close()
