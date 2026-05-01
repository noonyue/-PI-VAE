"""
Redraw Figure 4: L1 Drug Classification Confusion Matrix
符合学术标准的L1混淆矩阵图（含药物名称标签）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置学术期刊风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 药物名称（9类）
drug_names = ['CIM', 'FMD', 'GLD', 'GSR', 'HCT', 'IBU', 'MHE', 'MHL', 'MHR']

# 真实混淆矩阵数据（从图4读取）
# 行=真实标签，列=预测标签
# 0:CIM 1:FMD 2:GLD 3:GSR 4:HCT 5:IBU 6:MHE 7:MHL 8:MHR
cm = np.array([
    [10,  0,  0,  0,  0,  0,  0,  0,  0],  # CIM: 10/10 correct
    [ 0,  6,  0,  0,  0,  0,  0,  0,  0],  # FMD: 6/6 correct
    [ 0,  0,  4,  0,  0,  0,  0,  0,  0],  # GLD: 4/4 correct
    [ 0,  0,  0,  4,  0,  0,  0,  0,  0],  # GSR: 4/4 correct
    [ 0,  5,  0,  0,  1,  0,  0,  0,  0],  # HCT: 1/6 correct (5 misclassified as FMD)
    [ 0,  0,  0,  0,  0,  8,  0,  0,  0],  # IBU: 8/8 correct
    [ 0,  0,  0,  0,  0,  0,  7,  0,  0],  # MHE: 7/7 correct
    [ 0,  0,  0,  0,  0,  0,  0, 11,  0],  # MHL: 11/11 correct
    [ 0,  0,  0,  0,  0,  0,  0,  0, 16],  # MHR: 16/16 correct
])

# 计算每类准确率
per_class_acc = cm.diagonal() / cm.sum(axis=1)
overall_acc = cm.diagonal().sum() / cm.sum()

# 创建图形
fig, ax = plt.subplots(figsize=(10, 9))

# 归一化混淆矩阵（按行，即按真实类别归一化）
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# 绘制热图
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Proportion', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 坐标轴设置（使用药物名称）
n_classes = len(drug_names)
ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.set_xticklabels(drug_names, fontsize=12, fontweight='bold')
ax.set_yticklabels(drug_names, fontsize=12, fontweight='bold')

# 旋转x轴标签
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

# 在每个格子内写数值（同时显示原始计数和比例）
thresh = 0.5
for i in range(n_classes):
    for j in range(n_classes):
        count = cm[i, j]
        prop = cm_norm[i, j]
        if count > 0:
            color = 'white' if prop > thresh else 'black'
            # 对角线（正确预测）：显示数量和百分比
            if i == j:
                text = f'{count}\n({prop*100:.0f}%)'
                ax.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
            else:
                # 非对角线（误分）：显示数量
                ax.text(j, i, str(count), ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
        else:
            # 零值：不显示或显示小点
            pass

# 添加网格线（区分格子）
ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', size=0)

# 高亮误分格子（红色边框）
for i in range(n_classes):
    for j in range(n_classes):
        if i != j and cm[i, j] > 0:
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='red', linewidth=2.5, zorder=5
            )
            ax.add_patch(rect)

# 坐标轴标签
ax.set_xlabel('Predicted Drug Type', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylabel('True Drug Type', fontsize=13, fontweight='bold', labelpad=12)

# 标题
ax.set_title(
    'Figure 4. L1 Drug Classification Confusion Matrix\n'
    f'(Overall Accuracy = {overall_acc*100:.2f}%, n = {cm.sum()} test samples)',
    fontsize=13, fontweight='bold', pad=15
)

# 添加每类准确率标注（右侧）
for i, (drug, acc) in enumerate(zip(drug_names, per_class_acc)):
    color = '#2E7D32' if acc == 1.0 else '#C62828'
    ax.annotate(
        f'{acc*100:.0f}%',
        xy=(1.01, (n_classes - 1 - i) / (n_classes - 1)),
        xycoords='axes fraction',
        fontsize=9, color=color, fontweight='bold',
        va='center'
    )

# 添加统计信息文本框
stats_text = (
    f'Test samples: {cm.sum()}\n'
    f'Correct: {cm.diagonal().sum()}\n'
    f'Misclassified: {cm.sum() - cm.diagonal().sum()}\n'
    f'Perfect classes: {(per_class_acc == 1.0).sum()}/9'
)
ax.text(
    0.02, 0.02, stats_text,
    transform=ax.transAxes,
    fontsize=9, va='bottom', ha='left',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
              edgecolor='gray', linewidth=1, alpha=0.9)
)

# 调整布局
plt.tight_layout()

# 保存
os.makedirs('figures', exist_ok=True)
out_path = 'figures/4-l1_confusion_matrix.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"[OK] Figure saved: {out_path}")
plt.close()
