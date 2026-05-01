"""
Physics-Informed Decoder Architecture Comparison
绘制物理先验解码器详细架构对比图（UV高斯 vs NIR洛伦兹）
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

# 创建图形（16:12比例）
fig, ax = plt.subplots(figsize=(16, 12))
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

def draw_arrow(ax, start, end, label='', linewidth=2, color='black', fontsize=8):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=20,
        linewidth=linewidth, color=color, zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 1.5, label,
                ha='center', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8), zorder=3)

def draw_gaussian_peaks(ax, xy, width, height):
    """在框内绘制高斯峰示意图"""
    x = np.linspace(xy[0], xy[0] + width, 100)
    y_base = xy[1] + height * 0.2

    # 3个高斯峰
    peaks = [(xy[0] + width*0.25, 0.6), (xy[0] + width*0.5, 0.8), (xy[0] + width*0.75, 0.5)]
    y_total = np.zeros_like(x)

    for peak_x, amp in peaks:
        y = amp * height * 0.6 * np.exp(-((x - peak_x)**2) / (0.5 * width * 0.1)**2)
        y_total += y

    ax.plot(x, y_base + y_total, color='#FF6F00', linewidth=1.5, zorder=4)
    ax.fill_between(x, y_base, y_base + y_total, color='#FFE0B2', alpha=0.5, zorder=4)

def draw_lorentzian_peaks(ax, xy, width, height):
    """在框内绘制洛伦兹峰示意图"""
    x = np.linspace(xy[0], xy[0] + width, 100)
    y_base = xy[1] + height * 0.2

    # 3个洛伦兹峰
    peaks = [(xy[0] + width*0.3, 0.6, 0.8), (xy[0] + width*0.55, 0.75, 1.0), (xy[0] + width*0.8, 0.5, 0.7)]
    y_total = np.zeros_like(x)

    for peak_x, amp, gamma in peaks:
        gamma_scaled = gamma * width * 0.05
        y = amp * height * 0.6 * (gamma_scaled**2) / ((x - peak_x)**2 + gamma_scaled**2)
        y_total += y

    ax.plot(x, y_base + y_total, color='#2E7D32', linewidth=1.5, zorder=4)
    ax.fill_between(x, y_base, y_base + y_total, color='#C8E6C9', alpha=0.5, zorder=4)

# ============ 标题 ============
ax.text(50, 96, 'Physics-Informed Decoder Design', ha='center', fontsize=14, weight='bold')
ax.text(50, 93, 'Gaussian vs Lorentzian Peak Models', ha='center', fontsize=12)

# ============ 上半部分：UV-VAE高斯解码器 ============
uv_y_base = 52

# 输入层
uv_input_text = "Latent Code\nz_UV\n\n(B, 32)\n\nLearned\nrepresentation"
draw_box(ax, (5, uv_y_base + 10), 10, 14, uv_input_text, '#FFE0B2', fontsize=8, linewidth=2)

# 参数生成MLP
uv_mlp_text = "Parameter\nGenerator MLP\n\nLinear(32→128)\n+ ReLU\nLinear(128→256)\n+ ReLU\nLinear(256→N×3)"
draw_box(ax, (18, uv_y_base + 10), 12, 14, uv_mlp_text, '#FFE0B2', fontsize=7, linewidth=2)

# 参数分解（3个分支）
param_y_positions = [uv_y_base + 20, uv_y_base + 14, uv_y_base + 8]
param_colors = ['#BBDEFB', '#C8E6C9', '#FFF9C4']
param_texts = [
    "λᵢ (Position)\n[200, 800] nm\nSigmoid×600+200",
    "Aᵢ (Amplitude)\n[0, ∞)\nSoftplus",
    "σᵢ (Width)\n[5, 50] nm\nSigmoid×45+5"
]

for i, (y_pos, color, text) in enumerate(zip(param_y_positions, param_colors, param_texts)):
    draw_box(ax, (33, y_pos), 9, 4.5, text, color, fontsize=6.5, linewidth=1.5)
    draw_arrow(ax, (30, uv_y_base + 17), (33, y_pos + 2.25), linewidth=1.5)

# 高斯峰生成器
uv_gaussian_text = "Gaussian Peak Generator\n\nI(λ) = Σᵢ₌₁ᴺ Aᵢ × exp[-(λ-λᵢ)²/(2σᵢ²)]\n\nElectronic transition line shape"
gaussian_box = draw_box(ax, (45, uv_y_base + 10), 16, 14, uv_gaussian_text, '#FFE0B2',
                        fontsize=8, linewidth=3, bold=True)
# 在框内绘制高斯峰
draw_gaussian_peaks(ax, (46, uv_y_base + 11), 14, 6)

# 箭头连接
for y_pos in param_y_positions:
    draw_arrow(ax, (42, y_pos + 2.25), (45, uv_y_base + 17), linewidth=1.5)

# 重建输出
uv_output_text = "Reconstructed\nUV Spectrum\n\n(B, 2012)\n200-800 nm"
draw_box(ax, (64, uv_y_base + 12), 11, 10, uv_output_text, '#FFE0B2', fontsize=8, linewidth=2)
draw_arrow(ax, (61, uv_y_base + 17), (64, uv_y_base + 17), linewidth=2)

# 物理意义标注
uv_physics_text = "✓ Gaussian: Natural line shape\n   for electronic transitions\n✓ Peak position → Chromophore\n   energy levels\n✓ Peak width → Broadening\n✓ Peak height → Oscillator strength"
draw_box(ax, (78, uv_y_base + 10), 18, 14, uv_physics_text, '#F5F5F5',
         fontsize=6.5, linewidth=1.5, edgecolor='gray')

# ============ 中间分隔线 ============
ax.plot([5, 95], [50, 50], 'k-', linewidth=2, zorder=1)
ax.text(50, 50.5, 'UV: Electronic Transitions  ↔  NIR: Vibrational Overtones',
        ha='center', va='bottom', fontsize=11, weight='bold', color='#424242')

# ============ 下半部分：NIR-VAE洛伦兹解码器 ============
nir_y_base = 8

# 输入层
nir_input_text = "Latent Code\nz_NIR\n\n(B, 32)\n\nLearned\nrepresentation"
draw_box(ax, (5, nir_y_base + 10), 10, 14, nir_input_text, '#C8E6C9', fontsize=8, linewidth=2)

# 参数生成MLP
nir_mlp_text = "Parameter\nGenerator MLP\n\nLinear(32→128)\n+ ReLU\nLinear(128→256)\n+ ReLU\nLinear(256→N×3)"
draw_box(ax, (18, nir_y_base + 10), 12, 14, nir_mlp_text, '#C8E6C9', fontsize=7, linewidth=2)

# 参数分解（3个分支）
nir_param_y_positions = [nir_y_base + 20, nir_y_base + 14, nir_y_base + 8]
nir_param_texts = [
    "λᵢ (Position)\n[780, 2500] nm\nSigmoid×1720+780",
    "Aᵢ (Amplitude)\n[0, ∞)\nSoftplus",
    "Γᵢ (FWHM)\n[10, 100] nm\nSigmoid×90+10"
]

for i, (y_pos, color, text) in enumerate(zip(nir_param_y_positions, param_colors, nir_param_texts)):
    draw_box(ax, (33, y_pos), 9, 4.5, text, color, fontsize=6.5, linewidth=1.5)
    draw_arrow(ax, (30, nir_y_base + 17), (33, y_pos + 2.25), linewidth=1.5)

# 洛伦兹峰生成器
nir_lorentzian_text = "Lorentzian Peak Generator\n\nI(λ) = Σᵢ₌₁ᴺ Aᵢ × [Γᵢ²/((λ-λᵢ)²+Γᵢ²)]\n\nVibrational overtone line shape"
lorentzian_box = draw_box(ax, (45, nir_y_base + 10), 16, 14, nir_lorentzian_text, '#C8E6C9',
                          fontsize=8, linewidth=3, bold=True)
# 在框内绘制洛伦兹峰
draw_lorentzian_peaks(ax, (46, nir_y_base + 11), 14, 6)

# 箭头连接
for y_pos in nir_param_y_positions:
    draw_arrow(ax, (42, y_pos + 2.25), (45, nir_y_base + 17), linewidth=1.5)

# 重建输出
nir_output_text = "Reconstructed\nNIR Spectrum\n\n(B, 400)\n780-2500 nm"
draw_box(ax, (64, nir_y_base + 12), 11, 10, nir_output_text, '#C8E6C9', fontsize=8, linewidth=2)
draw_arrow(ax, (61, nir_y_base + 17), (64, nir_y_base + 17), linewidth=2)

# 物理意义标注
nir_physics_text = "✓ Lorentzian: Natural line shape\n   for vibrational modes\n✓ Peak position → Functional\n   group vibrations\n✓ Peak width → Lifetime broadening\n✓ Peak height → Absorption coefficient"
draw_box(ax, (78, nir_y_base + 10), 18, 14, nir_physics_text, '#F5F5F5',
         fontsize=6.5, linewidth=1.5, edgecolor='gray')

# 箭头连接各层
draw_arrow(ax, (15, uv_y_base + 17), (18, uv_y_base + 17), linewidth=2)
draw_arrow(ax, (15, nir_y_base + 17), (18, nir_y_base + 17), linewidth=2)

# 保存图形
plt.savefig('figures/pi_vae_decoder_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Decoder comparison diagram saved to: figures/pi_vae_decoder_comparison.png")
plt.close()
