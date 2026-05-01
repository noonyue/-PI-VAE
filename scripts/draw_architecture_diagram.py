"""
PI-VAE System Architecture Diagram Generator
绘制PI-VAE系统架构总览图
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

# 创建图形
fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 辅助函数：绘制圆角矩形框
def draw_box(ax, xy, width, height, text, color, fontsize=9, linewidth=2,
             edgecolor='black', text_color='black', bold=False):
    """绘制带文字的圆角矩形框"""
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
            weight=weight, color=text_color, zorder=3)
    return box

# 辅助函数：绘制箭头
def draw_arrow(ax, start, end, label='', style='solid', linewidth=2,
               color='black', fontsize=8):
    """绘制箭头并可选添加标签"""
    linestyle = '--' if style == 'dashed' else '-'
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=20,
        linewidth=linewidth, color=color,
        linestyle=linestyle, zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 1.5, label,
                ha='center', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8), zorder=3)

# ============ 1. 数据输入层 ============
input_text = "Raw Spectral Data\n\n357 samples\n9 drugs, 28 manufacturers\nUV-Vis: 2012 wavelengths\nNIR: 400 wavelengths"
draw_box(ax, (2, 70), 12, 18, input_text, '#F5F5F5', fontsize=8, linewidth=2)

# ============ 2. 预处理层 ============
snv_text = "SNV Normalization\n\nx_SNV = (x - μ) / σ\n\nUV drift: ↓99.9997%\nNIR drift: ↓99.9969%"
draw_box(ax, (17, 72), 11, 14, snv_text, '#E3F2FD', fontsize=7.5, linewidth=2)
draw_arrow(ax, (14, 79), (17, 79), 'Preprocessing', linewidth=2)

# ============ 3. PI-VAE特征提取层 ============
# UV-VAE路径（上方）
uv_encoder_text = "UV Encoder\n\n3-layer MLP\n2012→256→128→32×2\n\nz_UV ~ N(μ, σ²)"
draw_box(ax, (31, 80), 10, 12, uv_encoder_text, '#FFE0B2', fontsize=7, linewidth=2)

uv_decoder_text = "Gaussian Peak\nDecoder\n\nI(λ) = ΣAᵢexp[-(λ-λᵢ)²/(2σᵢ²)]\n\nElectronic transitions"
draw_box(ax, (44, 80), 10, 12, uv_decoder_text, '#FFE0B2', fontsize=6.5, linewidth=2.5)

uv_recon_text = "UV\nReconstruction\n\n(B, 2012)"
draw_box(ax, (57, 82), 7, 8, uv_recon_text, '#FFE0B2', fontsize=7, linewidth=2)

# NIR-VAE路径（下方）
nir_encoder_text = "NIR Encoder\n\n3-layer MLP\n400→256→128→32×2\n\nz_NIR ~ N(μ, σ²)"
draw_box(ax, (31, 64), 10, 12, nir_encoder_text, '#C8E6C9', fontsize=7, linewidth=2)

nir_decoder_text = "Lorentzian Peak\nDecoder\n\nI(λ) = ΣAᵢ[Γᵢ²/((λ-λᵢ)²+Γᵢ²)]\n\nVibrational overtones"
draw_box(ax, (44, 64), 10, 12, nir_decoder_text, '#C8E6C9', fontsize=6.5, linewidth=2.5)

nir_recon_text = "NIR\nReconstruction\n\n(B, 400)"
draw_box(ax, (57, 66), 7, 8, nir_recon_text, '#C8E6C9', fontsize=7, linewidth=2)

# 箭头连接
draw_arrow(ax, (28, 86), (31, 86), linewidth=2)
draw_arrow(ax, (28, 70), (31, 70), linewidth=2)
draw_arrow(ax, (41, 86), (44, 86), linewidth=2)
draw_arrow(ax, (41, 70), (44, 70), linewidth=2)
draw_arrow(ax, (54, 86), (57, 86), linewidth=2)
draw_arrow(ax, (54, 70), (57, 70), linewidth=2)

# 特征融合框
fusion_text = "Feature Fusion\n\n[z_UV; z_NIR]\n\n(B, 64)"
draw_box(ax, (43, 50), 9, 8, fusion_text, '#E1BEE7', fontsize=7.5, linewidth=2)

# 从编码器到融合的箭头
draw_arrow(ax, (36, 80), (47, 58), linewidth=2)
draw_arrow(ax, (36, 64), (47, 58), linewidth=2)

# ============ 4. 层级分类层 ============
# L1分类器
l1_text = "L1: Drug Classification\n\nSVM (RBF, C=10)\n9 classes\n\nAccuracy: 100%"
draw_box(ax, (56, 48), 10, 12, l1_text, '#FFF9C4', fontsize=7.5, linewidth=2.5)
draw_arrow(ax, (52, 54), (56, 54), 'Cascade', linewidth=2)

# L2分类器组（简化显示）
l2_text = "L2: Manufacturer ID\n\n28 manufacturers\n9 per-drug models\n\nCIM: RF, 100%\nFMD: RF, 100%\nGLD: RF, 100%\nGSR: RF, 100%\nHCT: RF, 83.3%\nIBU: RF, 100%\nMHE: SVM, 100%\nMHL: RF, 100%\nMHR: RF, 93.8%"
draw_box(ax, (70, 42), 12, 24, l2_text, '#B2EBF2', fontsize=6.5, linewidth=2.5)
draw_arrow(ax, (66, 54), (70, 54), linewidth=2)

# ============ 5. 最终输出层 ============
output_text = "Final Predictions\n\nDrug Type\nManufacturer\nConfidence"
draw_box(ax, (85, 50), 10, 8, output_text, '#E0E0E0', fontsize=7.5, linewidth=2)
draw_arrow(ax, (82, 54), (85, 54), linewidth=2)

# ============ 6. 损失函数标注（底部） ============
loss_recon_text = "L_recon = MSE(x, x̂)"
draw_box(ax, (50, 30), 12, 4, loss_recon_text, '#FFCDD2', fontsize=7, linewidth=1.5)

loss_kl_text = "L_KL = -0.5×Σ(1+log(σ²)-μ²-σ²)"
draw_box(ax, (50, 24), 12, 4, loss_kl_text, '#FFCDD2', fontsize=7, linewidth=1.5)

loss_total_text = "L_total = L_recon + β×L_KL  (β=1.0)"
draw_box(ax, (48, 18), 16, 4, loss_total_text, '#DC143C', fontsize=8,
         linewidth=2.5, text_color='white', bold=True)

# 虚线箭头指向损失
draw_arrow(ax, (60.5, 82), (56, 34), style='dashed', linewidth=1.5, color='#DC143C')
draw_arrow(ax, (60.5, 66), (56, 34), style='dashed', linewidth=1.5, color='#DC143C')
draw_arrow(ax, (36, 80), (56, 28), style='dashed', linewidth=1.5, color='#DC143C')
draw_arrow(ax, (36, 64), (56, 28), style='dashed', linewidth=1.5, color='#DC143C')

# ============ 7. 添加标题 ============
ax.text(50, 96, 'Physics-Informed VAE for Pharmaceutical Spectral Analysis',
        ha='center', fontsize=14, weight='bold')
ax.text(50, 93, 'Hierarchical Cascade Architecture',
        ha='center', fontsize=12)

# ============ 8. 添加图例/说明 ============
legend_text = "Data Flow: Solid arrows\nLoss Feedback: Dashed arrows\nPhysics-Informed: Orange/Green boxes"
ax.text(5, 10, legend_text, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='gray', linewidth=1))

# 保存图形
plt.savefig('figures/pi_vae_architecture_overview.png', dpi=300, bbox_inches='tight')
print("[OK] Architecture diagram saved to: figures/pi_vae_architecture_overview.png")
plt.close()
