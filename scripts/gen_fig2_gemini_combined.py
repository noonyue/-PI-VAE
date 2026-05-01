#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: PI-VAE Architecture
策略：
  1. Gemini Imagen 生成高质量概念插图（上半部分）
  2. matplotlib 生成精确结构图（下半部分）
  3. PIL 拼合为最终 Figure 2
"""
import os, sys, base64, json, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from plotting_style import setup_style
setup_style()

# ── 路径 & API
OUT_DIR  = Path('d:/work/class/GEN_MODEL/figures/paper')
OUT_DIR.mkdir(parents=True, exist_ok=True)
API_KEY  = 'AIzaSyBcTEBoO1httzE2V4gOuXFRqimbzcAEVuc'
PROXY    = 'http://127.0.0.1:33210'
IMAGEN_URL = (
    'https://generativelanguage.googleapis.com/v1beta/'
    'models/imagen-4.0-generate-001:predict'
    f'?key={API_KEY}'
)

# ── 色彩
CUV  = '#2166AC'
CNIR = '#C0392B'
CLAT = '#27AE60'
CFUS = '#D4860B'
CL1  = '#6C3483'
CL2  = '#1A5276'
CLOS = '#7F8C8D'
CRAW = '#34495E'

# ── Gemini Imagen 辅助函数
def fetch_gemini_image(prompt: str, out_path: Path) -> bool:
    """调用 Gemini Imagen API 生成图像，保存到 out_path，返回是否成功。"""
    session = requests.Session()
    session.proxies = {'http': PROXY, 'https': PROXY}
    session.verify = False
    payload = {
        'instances': [{'prompt': prompt}],
        'parameters': {
            'sampleCount': 1,
            'aspectRatio': '16:9',
        }
    }
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = session.post(IMAGEN_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        b64 = data['predictions'][0]['bytesBase64Encoded']
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        print(f'[Gemini OK] -> {out_path}')
        return True
    except Exception as e:
        print(f'[Gemini ERR] {type(e).__name__}: {e}')
        return False


GEMINI_PROMPT = (
    "Scientific academic architecture diagram, white background, clean vector style, "
    "no decorative elements, no 3D effects, flat design. "
    "Title at top: 'PI-VAE System Architecture'. "
    "Three clearly labeled horizontal sections separated by dashed vertical lines. "
    ""
    "LEFT SECTION (label 'A. Input'): "
    "Two stacked spectral plots. Top: blue Gaussian-shaped curve labeled 'UV-Vis (200-700nm)' "
    "with annotation 'Gaussian peaks'. Bottom: red Lorentzian-shaped curve labeled 'NIR (700-2500nm)' "
    "with annotation 'Lorentzian peaks'. "
    "A gray rounded rectangle below both curves labeled 'SNV Normalization'. "
    ""
    "MIDDLE SECTION (label 'B. PI-VAE Core'): "
    "Two parallel horizontal lanes. "
    "Top lane (light blue background): 'UV-Vis Path'. "
    "Shows MLP encoder with 4 layers (circles as nodes: 5->4->3->2), arrow to reparameterization "
    "box 'z~N(mu,sigma^2)', arrow to green box 'z_UV 32-dim', arrow to blue 'Gaussian Peak Decoder', "
    "arrow to reconstructed UV curve. Below: gray loss box 'beta-VAE Loss: L_rec + beta*D_KL'. "
    "Bottom lane (light red background): 'NIR Path'. "
    "Mirror structure: MLP encoder, reparameterization, green 'z_NIR 32-dim', "
    "red 'Lorentzian Peak Decoder', reconstructed NIR curve. "
    "Physics formula boxes: UV shows 'f(lambda)=Sum Ak*exp[-(lambda-mu_k)^2/2sigma_k^2]', "
    "NIR shows 'f(lambda)=Sum Ak/[1+((lambda-lambda_k)/Gamma_k)^2]'. "
    ""
    "RIGHT SECTION (label 'C. Cascade Classifier'): "
    "Orange fusion box at top 'Feature Fusion: [z_UV; z_NIR; x_raw] 64-dim + raw'. "
    "Arrow down to purple box 'L1: SVM (RBF Kernel)' with badge '100% Accuracy'. "
    "9 drug-type nodes below (CIM FMD GLD GSR HCT IBU MHE MHL MHR) in purple. "
    "Each drug node has arrow down to a dark-blue L2 box (SVM/RF/PLS label). "
    "Accuracy badges per drug below L2 boxes. "
    "Bottom row: small blue manufacturer squares labeled M, indicating 28 total manufacturers. "
    "Overall L2 accuracy badge: '97.22%'. "
    ""
    "Color coding: blue=#2166AC for UV, red=#C0392B for NIR, green=#27AE60 for latent space, "
    "orange=#D4860B for fusion, purple=#6C3483 for L1, dark blue=#1A5276 for L2. "
    "All text in sans-serif font. Arrows show information flow left to right."
)

# ── matplotlib 精确结构图（复用 gen_combined_fig2_arch_flow.py 核心逻辑）

def rbox(ax, x, y, w, h, txt, fc, ec='#2C3E50', tc='white',
         fs=7.5, fw='bold', alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle='round,pad=0.09', facecolor=fc,
                                edgecolor=ec, lw=1.1, alpha=alpha, zorder=z))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, zorder=z+1, linespacing=1.35)


def rarr(ax, x0, y0, x1, y1, c='#555', lw=1.3):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))


def nn_layer(ax, cx, cy, n, r=0.06, fc='white', ec='#444'):
    gap = r * 2.7
    xs = np.linspace(cx - gap*(n-1)/2, cx + gap*(n-1)/2, n)
    for xi in xs:
        ax.add_patch(plt.Circle((xi, cy), r, color=fc, ec=ec, lw=0.9, zorder=5))
    return xs


def draw_mlp(ax, cx, y_top, clr, n_list=(5, 4, 3, 2), gap=0.5):
    prev_xs, ys = None, [y_top - i*gap for i in range(len(n_list))]
    for i, (n, y) in enumerate(zip(n_list, ys)):
        fc = CLAT if i == len(n_list)-1 else clr
        xs = nn_layer(ax, cx, y, n, fc=fc, ec='#333')
        if prev_xs is not None:
            for px in prev_xs:
                for nx in xs:
                    ax.plot([px, nx], [ys[i-1], y], color='#ccc', lw=0.35, zorder=2)
        prev_xs = xs
    return ys

def draw_section_a(ax):
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 4.9, 9.6,
                                boxstyle='round,pad=0.1', fc='#FAFAFA',
                                ec='#BBBBBB', lw=1.2, zorder=0))
    lam = np.linspace(0, 1, 300)
    yuv = (0.55*np.exp(-((lam-0.30)/0.08)**2) +
           0.90*np.exp(-((lam-0.58)/0.10)**2) +
           0.40*np.exp(-((lam-0.78)/0.07)**2))
    px = np.linspace(0.3, 4.8, 300)
    ax.plot(px, 5.8+yuv*2.8, color=CUV, lw=2.0, zorder=3)
    ax.fill_between(px, 5.8, 5.8+yuv*2.8, alpha=0.22, color=CUV)
    ax.text(2.55, 9.2, 'UV-Vis Spectrum', ha='center', fontsize=8,
            color=CUV, fontweight='bold')
    ax.text(2.55, 8.85, '200-700 nm  |  Gaussian peaks', ha='center',
            fontsize=6.5, color='#555')
    rarr(ax, 4.42, 7.2, 4.95, 7.2, CUV, lw=1.8)
    ynir = (0.65/(1+((lam-0.25)/0.07)**2) +
            0.90/(1+((lam-0.55)/0.09)**2) +
            0.45/(1+((lam-0.82)/0.06)**2))
    ax.plot(px, 2.3+ynir*1.8, color=CNIR, lw=2.0, zorder=3)
    ax.fill_between(px, 2.3, 2.3+ynir*1.8, alpha=0.22, color=CNIR)
    ax.text(2.55, 1.95, 'NIR Spectrum', ha='center', fontsize=8,
            color=CNIR, fontweight='bold')
    ax.text(2.55, 1.58, '700-2500 nm  |  Lorentzian peaks', ha='center',
            fontsize=6.5, color='#555')
    rarr(ax, 4.42, 3.7, 4.95, 3.7, CNIR, lw=1.8)
    rbox(ax, 2.55, 4.6, 2.8, 0.55, 'Preprocessing: SNV Normalisation',
         '#546E7A', tc='white', fs=7, fw='normal')

def draw_section_b(ax):
    ax.add_patch(FancyBboxPatch((5.1, 0.2), 10.8, 9.6,
                                boxstyle='round,pad=0.1', fc='#F8FDF3',
                                ec='#99BB88', lw=1.5, zorder=0))
    ax.text(10.5, 9.65, 'Physics-Informed Variational Autoencoder (PI-VAE)',
            ha='center', fontsize=9.5, fontweight='bold', color='#1a1a1a')
    # UV lane
    ax.add_patch(FancyBboxPatch((5.25, 5.4), 10.5, 4.1,
                                boxstyle='round,pad=0.05', fc='#E8F4FD',
                                ec='none', alpha=0.6, zorder=1))
    ax.text(5.5, 9.3, 'UV-Vis Path', fontsize=7.5, color=CUV, fontweight='bold')
    ys_uv = draw_mlp(ax, cx=7.0, y_top=8.8, clr=CUV)
    ax.text(7.0, 8.8+0.22, 'UV Encoder', ha='center', fontsize=7, color=CUV, fontweight='bold')
    ax.text(7.0, ys_uv[0]-0.18, 'Input (400-d)', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_uv[1]-0.18, 'FC-256', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_uv[2]-0.18, 'FC-128', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_uv[3]-0.18, u'\u03bc,\u03c3\u00b2 (32-d)', ha='center', fontsize=6, color=CLAT, fontweight='bold')
    rbox(ax, 9.2, 7.3, 1.6, 0.5, u'z ~ N(\u03bc,\u03c3\u00b2)\nReparam.', CLAT, fs=7)
    rarr(ax, 7.35, ys_uv[3], 8.4, 7.3, CLAT)
    rbox(ax, 11.0, 7.3, 1.3, 0.5, 'z_UV\n(32-dim)', CLAT, fs=7)
    rarr(ax, 10.0, 7.3, 10.35, 7.3, CLAT)
    rbox(ax, 12.9, 7.3, 1.7, 0.5, 'Gaussian\nPeak Decoder', CUV, fs=7)
    rarr(ax, 11.65, 7.3, 12.05, 7.3, CUV)
    lx = np.linspace(14.1, 15.7, 100)
    ry = 7.0 + 0.75*np.exp(-((lx-14.9)/0.28)**2) + 0.45*np.exp(-((lx-15.4)/0.22)**2)
    ax.plot(lx, ry, color=CUV, lw=1.8)
    ax.fill_between(lx, 7.0, ry, alpha=0.2, color=CUV)
    ax.text(14.9, 6.72, u'x\u0302_UV', ha='center', fontsize=7.5, color=CUV, fontweight='bold')
    rarr(ax, 13.75, 7.3, 14.1, 7.5, CUV)
    ax.text(14.9, 6.3, u'f(\u03bb) = \u03a3 A\u2096 exp[-(\u03bb-\u03bc\u2096)\u00b2 / 2\u03c3\u2096\u00b2]',
            ha='center', fontsize=6.5, color='#222',
            bbox=dict(fc='white', ec=CUV, lw=0.8, boxstyle='round,pad=0.25'))
    rbox(ax, 11.0, 6.1, 2.0, 0.48, u'\u03b2-VAE Loss: L_rec + \u03b2 D_KL', CLOS, fs=6.5)
    rarr(ax, 11.0, 7.05, 11.0, 6.34, CLOS, lw=0.9)
    rarr(ax, 12.9, 7.05, 12.9, 6.34, CLOS, lw=0.9)
    # NIR lane
    ax.add_patch(FancyBboxPatch((5.25, 0.5), 10.5, 4.6,
                                boxstyle='round,pad=0.05', fc='#FDEDEC',
                                ec='none', alpha=0.6, zorder=1))
    ax.text(5.5, 4.85, 'NIR Path', fontsize=7.5, color=CNIR, fontweight='bold')
    ys_nir = draw_mlp(ax, cx=7.0, y_top=4.3, clr=CNIR)
    ax.text(7.0, 4.3+0.22, 'NIR Encoder', ha='center', fontsize=7, color=CNIR, fontweight='bold')
    ax.text(7.0, ys_nir[0]-0.18, 'Input (200-d)', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_nir[1]-0.18, 'FC-256', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_nir[2]-0.18, 'FC-128', ha='center', fontsize=6, color='#555')
    ax.text(7.0, ys_nir[3]-0.18, u'\u03bc,\u03c3\u00b2 (32-d)', ha='center', fontsize=6, color=CLAT, fontweight='bold')
    rbox(ax, 9.2, 2.85, 1.6, 0.5, u'z ~ N(\u03bc,\u03c3\u00b2)\nReparam.', CLAT, fs=7)
    rarr(ax, 7.35, ys_nir[3], 8.4, 2.85, CLAT)
    rbox(ax, 11.0, 2.85, 1.3, 0.5, 'z_NIR\n(32-dim)', CLAT, fs=7)
    rarr(ax, 10.0, 2.85, 10.35, 2.85, CLAT)
    rbox(ax, 12.9, 2.85, 1.7, 0.5, 'Lorentzian\nPeak Decoder', CNIR, fs=7)
    rarr(ax, 11.65, 2.85, 12.05, 2.85, CNIR)
    lx2 = np.linspace(14.1, 15.7, 100)
    ry2 = 2.6 + 0.65/(1+((lx2-14.6)/0.22)**2) + 0.45/(1+((lx2-15.2)/0.18)**2)
    ax.plot(lx2, ry2, color=CNIR, lw=1.8)
    ax.fill_between(lx2, 2.6, ry2, alpha=0.2, color=CNIR)
    ax.text(14.9, 2.32, u'x\u0302_NIR', ha='center', fontsize=7.5, color=CNIR, fontweight='bold')
    rarr(ax, 13.75, 2.85, 14.1, 3.0, CNIR)
    ax.text(14.9, 1.9, u'f(\u03bb) = \u03a3 A\u2096 / [1 + ((\u03bb-\u03bb\u2096)/\u0393\u2096)\u00b2]',
            ha='center', fontsize=6.5, color='#222',
            bbox=dict(fc='white', ec=CNIR, lw=0.8, boxstyle='round,pad=0.25'))
    rbox(ax, 11.0, 1.65, 2.0, 0.48, u'\u03b2-VAE Loss: L_rec + \u03b2 D_KL', CLOS, fs=6.5)
    rarr(ax, 11.0, 2.60, 11.0, 1.89, CLOS, lw=0.9)
    rarr(ax, 12.9, 2.60, 12.9, 1.89, CLOS, lw=0.9)
    # Feature fusion arrows into section C
    rarr(ax, 11.65, 7.3, 15.6, 8.6, CLAT, lw=1.2)
    rarr(ax, 11.65, 2.85, 15.6, 8.3, CLAT, lw=1.2)

def draw_section_c(ax):
    ax.add_patch(FancyBboxPatch((16.05, 0.2), 7.9, 9.6,
                                boxstyle='round,pad=0.1', fc='#F4F6F7',
                                ec='#AABBCC', lw=1.5, zorder=0))
    # Feature fusion box
    rbox(ax, 19.95, 8.6, 5.2, 0.65,
         u'Feature Fusion\n[z_UV \u2295 z_NIR \u2295 x_raw]  (64-dim + raw)', CFUS, fs=7.5)
    rbox(ax, 17.5, 7.75, 1.5, 0.5, 'Raw\nFeatures', CRAW, fs=6.5)
    rarr(ax, 17.2, 7.75, 17.8, 8.34, CRAW, lw=0.9)
    # L1 SVM
    rbox(ax, 19.95, 7.6, 5.2, 0.65,
         'L1: SVM (RBF Kernel)  -  Drug-type Classification', CL1, fs=8)
    rarr(ax, 19.95, 8.27, 19.95, 7.93, CL1, lw=1.6)
    ax.text(19.95, 7.15, u'\u2713 L1 Accuracy: 100%', ha='center',
            fontsize=8, color=CL1, fontweight='bold')
    # 9 drug nodes
    drugs = ['CIM','FMD','GLD','GSR','HCT','IBU','MHE','MHL','MHR']
    xs9 = np.linspace(16.5, 23.4, 9)
    y_d = 6.45
    for xi, d in zip(xs9, drugs):
        rbox(ax, xi, y_d, 0.82, 0.46, d, CL1, fs=6.5, alpha=0.8)
        rarr(ax, 19.95, 6.83, xi, y_d+0.23, CL1, lw=0.75)
    # L2 per-drug models
    l2m  = ['SVM','RF', 'PLS','RF', 'SVM','PLS','SVM','RF', 'SVM']
    accs = ['100%','100%','100%','100%','100%','83%','100%','100%','92%']
    y_l2 = 5.35
    for xi, m, a in zip(xs9, l2m, accs):
        rbox(ax, xi, y_l2, 0.82, 0.46, f'L2\n{m}', CL2, fs=6, alpha=0.85)
        rarr(ax, xi, y_d-0.23, xi, y_l2+0.23, CL2, lw=0.85)
        col = CL2 if a == '100%' else CNIR
        ax.text(xi, y_l2-0.38, a, ha='center', fontsize=6, color=col, fontweight='bold')
    # 28 manufacturer dots
    y_mfr = 4.2
    n_mfrs = [3,3,2,2,3,4,3,3,5]
    for xi, nm in zip(xs9, n_mfrs):
        show = min(nm, 3)
        mxs = np.linspace(xi-0.35, xi+0.35, show)
        for mx in mxs:
            rbox(ax, mx, y_mfr, 0.26, 0.32, 'M', '#2471A3',
                 tc='white', fs=5.5, alpha=0.75)
            rarr(ax, xi, y_l2-0.23, mx, y_mfr+0.16, CL2, lw=0.5)
        if nm > 3:
            ax.text(xi+0.48, y_mfr, f'+{nm-3}', fontsize=5, color=CL2)
    ax.text(19.95, 3.6, u'28 Manufacturers Total\n\u2713 L2 Accuracy: 97.22%',
            ha='center', fontsize=8, color=CL2, fontweight='bold')

def build_matplotlib_fig() -> Path:
    """生成 matplotlib 精确结构图，返回保存路径。"""
    fig, ax = plt.subplots(figsize=(24, 10), facecolor='white')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')

    draw_section_a(ax)
    draw_section_b(ax)
    draw_section_c(ax)

    # 分区虚线
    ax.axvline(5.05, color='#CCC', lw=1.0, ls='--', ymin=0.03, ymax=0.97)
    ax.axvline(15.95, color='#CCC', lw=1.0, ls='--', ymin=0.03, ymax=0.97)

    # 底部区块标签
    for xc, lbl in [(2.55, 'A. Input'),
                    (10.5, 'B. PI-VAE Core'),
                    (19.95, 'C. Cascade Classifier')]:
        ax.text(xc, 0.08, lbl, ha='center', fontsize=8, color='#555', style='italic')

    fig.suptitle('Figure 2.  PI-VAE System Architecture & Physics-Informed Decoder Design',
                 fontsize=12, fontweight='bold', y=1.01)

    out = OUT_DIR / 'fig2_matplotlib_precise.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[matplotlib OK] -> {out}')
    return out


def combine_images(gemini_path: Path, mpl_path: Path, out_path: Path):
    """将 Gemini 概念图（上）与 matplotlib 精确图（下）垂直拼合。"""
    try:
        from PIL import Image
        img_g = Image.open(gemini_path).convert('RGB')
        img_m = Image.open(mpl_path).convert('RGB')
        # 统一宽度为 matplotlib 图宽度
        target_w = img_m.width
        ratio = target_w / img_g.width
        new_h = int(img_g.height * ratio)
        img_g = img_g.resize((target_w, new_h), Image.LANCZOS)
        # 添加分隔条
        sep_h = 20
        total_h = new_h + sep_h + img_m.height
        combined = Image.new('RGB', (target_w, total_h), color=(240, 240, 240))
        combined.paste(img_g, (0, 0))
        combined.paste(img_m, (0, new_h + sep_h))
        combined.save(out_path, dpi=(200, 200))
        print(f'[Combine OK] -> {out_path}')
    except ImportError:
        print('[WARN] PIL not available, copying matplotlib figure as final output')
        import shutil
        shutil.copy(mpl_path, out_path)


def main():
    # 1. Gemini 概念图
    gemini_out = OUT_DIR / 'fig2_gemini_concept.png'
    gemini_ok  = fetch_gemini_image(GEMINI_PROMPT, gemini_out)

    # 2. matplotlib 精确图
    mpl_out = build_matplotlib_fig()

    # 3. 拼合
    final_out = OUT_DIR / 'combined_fig2_arch_flow.png'
    if gemini_ok and gemini_out.exists():
        combine_images(gemini_out, mpl_out, final_out)
    else:
        print('[INFO] Gemini unavailable, using matplotlib-only output.')
        import shutil
        shutil.copy(mpl_out, final_out)
        print(f'[Final] -> {final_out}')


if __name__ == '__main__':
    main()







