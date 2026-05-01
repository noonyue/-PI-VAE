"""
Hierarchical Cascade Classification Strategy Flowchart
绘制层级级联分类策略流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置学术期刊风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 创建图形（14:18比例，竖图）
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 辅助函数
def draw_box(ax, xy, width, height, text, color, fontsize=9, linewidth=2,
             edgecolor='black', text_color='black', bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05",
        edgecolor=edgecolor, facecolor=color,
        linewidth=linewidth, zorder=2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            weight=weight, color=text_color, zorder=3,
            multialignment='center')
    return box

def draw_arrow(ax, start, end, label='', linewidth=2, color='black',
               fontsize=7, style='solid'):
    """绘制箭头"""
    linestyle = '--' if style == 'dashed' else '-'
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=15,
        linewidth=linewidth, color=color,
        linestyle=linestyle, zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x + 1, mid_y, label,
                ha='left', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='none', alpha=0.9), zorder=3)

# ============ 标题 ============
ax.text(50, 97, 'Hierarchical Cascade Classification Strategy',
        ha='center', fontsize=14, weight='bold')
ax.text(50, 94.5, 'From 28-Class to Two-Level Classification',
        ha='center', fontsize=11)

# ============ 1. 输入层（顶部） ============
input_text = "Fused Features\n\nz_fused = [z_UV; z_NIR]\nDimension: 64\n\n357 samples\n(285 train / 72 test)"
draw_box(ax, (35, 85), 30, 8, input_text, '#E1BEE7', fontsize=9,
         linewidth=2.5, bold=True)

# ============ 2. L1分类层 ============
l1_text = "L1: Drug Type Classification\n\nModel: SVM (RBF kernel)\nHyperparameters: C=10, γ=scale\n\nClasses: 9 drugs\nAccuracy: 100% (72/72)"
draw_box(ax, (30, 72), 40, 9, l1_text, '#FFF9C4', fontsize=9,
         linewidth=3, edgecolor='#1976D2', bold=True)

draw_arrow(ax, (50, 85), (50, 81), linewidth=2.5)

# ============ 3. L1输出分支（9个药物） ============
drugs = [
    ('CIM', 50, '#FFCDD2'),
    ('FMD', 30, '#FFE0B2'),
    ('GLD', 19, '#FFF9C4'),
    ('GSR', 20, '#C8E6C9'),
    ('HCT', 30, '#B2EBF2'),
    ('IBU', 40, '#BBDEFB'),
    ('MHE', 34, '#E1BEE7'),
    ('MHL', 54, '#F8BBD0'),
    ('MHR', 80, '#E0E0E0')
]

# 计算分支位置
branch_y = 72
l2_y = 42
x_positions = np.linspace(8, 92, 9)

# 绘制从L1到各药物的箭头
for i, (drug, samples, color) in enumerate(drugs):
    x_pos = x_positions[i]
    # 绘制箭头
    draw_arrow(ax, (50, branch_y), (x_pos, l2_y + 18),
               label=f'{drug}\n({samples})', linewidth=2, color=color, fontsize=6.5)

# ============ 4. L2分类层（9个分类器） ============
l2_data = [
    ('CIM', 'RF', 3, '100%', '10/10', '#FFCDD2', ''),
    ('FMD', 'RF', 3, '100%', '6/6', '#FFE0B2', ''),
    ('GLD', 'RF', 2, '100%', '4/4', '#FFF9C4', ''),
    ('GSR', 'RF', 2, '100%', '4/4', '#C8E6C9', ''),
    ('HCT', 'RF', 3, '83.3%', '5/6', '#B2EBF2', '⚠'),
    ('IBU', 'RF', 4, '100%', '8/8', '#BBDEFB', ''),
    ('MHE', 'SVM', 3, '100%', '7/7', '#E1BEE7', '★'),
    ('MHL', 'RF', 3, '100%', '11/11', '#F8BBD0', ''),
    ('MHR', 'RF', 5, '93.8%', '15/16', '#E0E0E0', '')
]

# 第一行（5个）
for i in range(5):
    drug, model, n_classes, acc, test, color, icon = l2_data[i]
    x_pos = x_positions[i]

    text = f"{drug}\nManufacturer\n\n{model}\n{n_classes} classes\n{acc}\n({test})"
    if icon:
        text = f"{icon} {text}"

    draw_box(ax, (x_pos - 4.5, l2_y + 10), 9, 8, text, color,
             fontsize=6.5, linewidth=2)

# 第二行（4个）
for i in range(4):
    drug, model, n_classes, acc, test, color, icon = l2_data[i + 5]
    x_pos = x_positions[i + 5]

    text = f"{drug}\nManufacturer\n\n{model}\n{n_classes} classes\n{acc}\n({test})"
    if icon:
        text = f"{icon} {text}"

    draw_box(ax, (x_pos - 4.5, l2_y), 9, 8, text, color,
             fontsize=6.5, linewidth=2)

# ============ 5. 最终输出层（底部） ============
output_text = "Cascade Classification Results\n\nOverall L2 Accuracy: 97.22% (70/72)\nPerfect drugs: 7/9 (100%)\nNear-perfect: 2/9 (>80%)\nAverage per-drug accuracy: 97.45%"
draw_box(ax, (25, 28), 50, 8, output_text, '#4CAF50', fontsize=9,
         linewidth=3, text_color='white', bold=True)

# 从L2分类器到输出的箭头
for i, x_pos in enumerate(x_positions):
    if i < 5:
        start_y = l2_y + 10
    else:
        start_y = l2_y
    draw_arrow(ax, (x_pos, start_y), (50, 36), linewidth=1.5, color='black')

# ============ 6. 对比标注框（右侧） ============
# 基线对比
baseline_text = "Baseline:\nDirect 28-Class\n\nModel: SVM\n(same features)\n\nAccuracy: 61.11%\n(44/72)\n\nImprovement:\n+36.34%\n\np-value: <0.001"
draw_box(ax, (78, 55), 18, 20, baseline_text, '#FFCDD2', fontsize=7.5,
         linewidth=2, edgecolor='red')

# 优势标注
advantage_text = "Cascade\nAdvantages\n\n+ Task decomposition:\n  28 to 9+9x(2-5)\n\n+ Per-drug model\n  selection\n\n+ Better small-sample\n  handling\n\n+ Explicit hierarchy"
draw_box(ax, (78, 30), 18, 18, advantage_text, '#E8F5E9', fontsize=7,
         linewidth=2, edgecolor='green')

# ============ 7. 添加图例 ============
legend_elements = [
    mpatches.Patch(facecolor='#FFF9C4', edgecolor='#1976D2', linewidth=2, label='L1: Drug Classification'),
    mpatches.Patch(facecolor='#B2EBF2', edgecolor='black', linewidth=2, label='L2: Manufacturer ID'),
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', linewidth=2, label='Final Results'),
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          frameon=True, fancybox=True, shadow=True)

# ============ 8. 添加说明文字 ============
note_text = "Note: RF = Random Forest, SVM = Support Vector Machine\n★ = SVM selected by LOOCV, ⚠ = Challenging case"
ax.text(50, 22, note_text, ha='center', fontsize=7, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='gray', linewidth=1))

# 保存图形
plt.savefig('figures/cascade_classification_flowchart.png', dpi=300, bbox_inches='tight')
print("[OK] Cascade flowchart saved to: figures/cascade_classification_flowchart.png")
plt.close()
