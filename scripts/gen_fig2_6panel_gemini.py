#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: PI-VAE Architecture — 6-panel Gemini Imagen version.
6个子图分别对应论文2.2-2.4节的核心模块，每图独立调用Gemini Imagen生成，
最终用matplotlib拼合为2x3学术风格版面。

子图分工（与论文对应）：
  (a) 双模态光谱输入：UV高斯峰 vs NIR洛伦兹峰
  (b) SNV预处理：原始光谱 vs 归一化后
  (c) PI-VAE编码器：MLP + 重参数化采样
  (d) 物理先验解码器：高斯(UV) + 洛伦兹(NIR)重构
  (e) 多模态特征融合：z_UV ⊕ z_NIR ⊕ x_raw → 64维
  (f) 级联分类器：L1(SVM,100%) → 9药物 → L2 → 28厂商(97.22%)
"""
import os, sys, base64, requests, time
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import setup_style, add_panel_label
setup_style()

OUT_DIR  = Path('d:/work/class/GEN_MODEL/figures/paper')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR  = OUT_DIR / 'fig2_panels'
SUB_DIR.mkdir(parents=True, exist_ok=True)

API_KEY  = 'AIzaSyBcTEBoO1httzE2V4gOuXFRqimbzcAEVuc'
PROXY    = 'http://127.0.0.1:33210'
IMAGEN_URL = (
    'https://generativelanguage.googleapis.com/v1beta/'
    'models/imagen-4.0-generate-001:predict'
    f'?key={API_KEY}'
)
# ── 6个子图 Gemini Prompts（严格对应论文内容）
PANELS = [
    {
        'key': 'a',
        'file': 'fig2a_dual_input.png',
        'title': '(a) Dual-Modal Spectral Input',
        'prompt': (
            "Academic scientific diagram, white background, clean minimal flat style, no 3D, no shadows. "
            "Two vertically stacked spectral line plots, separated by a thin horizontal divider. "
            "TOP PLOT: smooth continuous blue line curve (color #2166AC) with exactly 3 overlapping Gaussian-shaped peaks, "
            "x-axis label 'Wavelength (nm)  200-700', y-axis label 'Absorbance', "
            "plot title 'UV-Vis Spectrum' in blue bold, "
            "annotation box on largest peak: 'Gaussian peaks: f(lambda)=Sum Ak*exp(-(lambda-mu)^2/2sigma^2)' in small text. "
            "BOTTOM PLOT: smooth continuous red line curve (color #C0392B) with exactly 3 broader Lorentzian-shaped peaks, "
            "x-axis label 'Wavelength (nm)  700-2500', y-axis label 'Absorbance', "
            "plot title 'NIR Spectrum' in red bold, "
            "annotation box on largest peak: 'Lorentzian peaks: f(lambda)=Sum Ak*gamma^2/((lambda-mu)^2+gamma^2)' in small text. "
            "Both plots: light gray grid, thin axis spines, sans-serif font, no legend. "
            "Right side of both plots: a small vertical bracket labeled 'Dual-modal input to PI-VAE'."
        )
    },
    {
        'key': 'b',
        'file': 'fig2b_snv_preprocess.png',
        'title': '(b) SNV Preprocessing',
        'prompt': (
            "Academic scientific diagram, white background, clean flat style. "
            "Two side-by-side spectral plots connected by a bold right-pointing arrow in the center. "
            "LEFT PLOT title 'Before SNV' in dark gray bold: "
            "5-7 overlapping colored spectral lines (use tab10 colors: blue, orange, green, red, purple) "
            "with clear vertical offset/baseline drift between lines, y-axis range 0 to 2.5, "
            "subtitle below title: 'Baseline drift + multiplicative scatter' in small italic red text. "
            "CENTER: thick black arrow pointing right, label above: 'SNV Normalization', "
            "formula below arrow: 'x_norm = (x - mean(x)) / std(x)' in monospace font inside a light gray box. "
            "RIGHT PLOT title 'After SNV' in dark gray bold: "
            "same 5-7 colored lines now tightly clustered near zero baseline, y-axis range -3 to 3, "
            "subtitle: 'Drift removed, unit variance' in small italic green text. "
            "Both plots: x-axis label 'Wavelength (a.u.)', thin gray grid, sans-serif font."
        )
    },
    {
        'key': 'c',
        'file': 'fig2c_encoder.png',
        'title': '(c) PI-VAE Encoder',
        'prompt': (
            "Academic neural network architecture diagram, white background, clean flat style, left-to-right flow. "
            "Title at top: 'PI-VAE Encoder (UV-Vis & NIR)'. "
            "TWO parallel vertical encoder lanes side by side, each showing a fully-connected neural network: "
            "LEFT LANE (blue #2166AC background tint): "
            "label 'UV-Vis Encoder' at top in blue bold. "
            "4 rows of circles representing MLP layers from top to bottom: "
            "Row 1: 6 large circles labeled 'Input  200-d' (blue outline), "
            "Row 2: 4 medium circles labeled 'FC-256', "
            "Row 3: 3 circles labeled 'FC-128', "
            "Row 4: 2 green circles labeled 'mu, sigma^2  32-d'. "
            "All circles connected by thin gray lines between adjacent rows. "
            "RIGHT LANE (red #C0392B background tint): "
            "mirror structure labeled 'NIR Encoder', same 4 rows, input labeled 'Input  700-d'. "
            "BOTTOM CENTER: green rounded rectangle 'Reparameterization: z = mu + epsilon * sigma, epsilon~N(0,I)'. "
            "Arrows from both encoder outputs converge into the reparameterization box. "
            "Two output arrows labeled 'z_UV (32-dim)' and 'z_NIR (32-dim)' in green. "
            "Clean sans-serif font, no 3D effects."
        )
    },
    {
        'key': 'd',
        'file': 'fig2d_decoder.png',
        'title': '(d) Physics-Informed Decoder',
        'prompt': (
            "Academic scientific diagram, white background, clean flat style, left-to-right flow. "
            "Title: 'Physics-Informed Decoder'. "
            "TWO parallel horizontal decoder pipelines, one above the other. "
            "TOP PIPELINE (UV-Vis, blue theme #2166AC): "
            "Green box 'z_UV (32-dim)' -> arrow -> blue box 'Gaussian Peak Decoder' -> arrow -> "
            "small inset plot showing blue Gaussian curve labeled 'x_hat_UV  Reconstructed'. "
            "Below the decoder box: formula in white-background bordered box: "
            "'x_recon(lambda) = Sum_i A_i * exp(-(lambda-mu_i)^2 / 2*sigma_i^2) + b' "
            "with annotations: A_i=amplitude, mu_i=peak center, sigma_i=half-width. "
            "Gray box below pipeline: 'Loss = ||x - x_hat||^2 + beta * KL(q(z|x)||p(z))'. "
            "BOTTOM PIPELINE (NIR, red theme #C0392B): "
            "Green box 'z_NIR (32-dim)' -> arrow -> red box 'Lorentzian Peak Decoder' -> arrow -> "
            "small inset plot showing red Lorentzian curve labeled 'x_hat_NIR  Reconstructed'. "
            "Below decoder: formula box: "
            "'x_recon(lambda) = Sum_i A_i * gamma_i^2 / ((lambda-mu_i)^2 + gamma_i^2) + b' "
            "with annotation: gamma_i=HWHM (half-width at half-maximum). "
            "Gray loss box below. Clean arrows, sans-serif font."
        )
    },
    {
        'key': 'e',
        'file': 'fig2e_fusion.png',
        'title': '(e) Multimodal Feature Fusion',
        'prompt': (
            "Academic diagram, white background, clean flat style. "
            "Title: 'Multimodal Feature Fusion'. "
            "Three input boxes on the left, converging via arrows into a central fusion box, then one output box on right. "
            "INPUT BOX 1 (top-left): green rounded rect 'z_UV  32-dim  (UV latent code)'. "
            "INPUT BOX 2 (middle-left): green rounded rect 'z_NIR  32-dim  (NIR latent code)'. "
            "INPUT BOX 3 (bottom-left): dark gray rounded rect 'x_raw  UV+NIR concatenated  (SNV-normalized)'. "
            "Three arrows converge into CENTER FUSION BOX: large orange rounded rectangle labeled "
            "'Feature Fusion  [z_UV ; z_NIR ; x_raw]' with concatenation brackets symbol, "
            "subtitle inside box: 'Dimension: 32+32+raw = 64-dim fused vector'. "
            "OUTPUT (right): dark blue box 'Fused Feature Vector  64-dim' with bold arrow from fusion box. "
            "Below output box: label 'Input to Cascade Classifier'. "
            "Color coding legend at bottom: green=latent, orange=fusion, gray=raw. "
            "Clean connective arrows, sans-serif font, no 3D."
        )
    },
    {
        'key': 'f',
        'file': 'fig2f_cascade.png',
        'title': '(f) Cascade Classifier  L1→L2',
        'prompt': (
            "Academic classification hierarchy diagram, white background, clean flat style, top-to-bottom flow. "
            "Title: 'Cascade Classifier: L1 Drug-type -> L2 Manufacturer'. "
            "TOP: dark blue input box 'Fused Feature Vector  64-dim'. "
            "Arrow down to PURPLE box 'L1 Classifier: SVM (RBF kernel, C=10, gamma=0.01)' "
            "with green accuracy badge on right: 'L1 Accuracy: 100%'. "
            "Below L1 box: 9 small purple rounded rectangles in a single row, "
            "labeled left to right: CIM, FMD, GLD, GSR, HCT, IBU, MHE, MHL, MHR "
            "(representing 9 drug types), each connected from L1 box by downward arrows. "
            "Below each drug node: a dark blue box labeled 'L2' with small model label "
            "(SVM or RF or PLS) and accuracy value "
            "(CIM:100%, FMD:100%, GLD:100%, GSR:100%, HCT:100%, IBU:83%, MHE:100%, MHL:100%, MHR:92%). "
            "100% values in dark blue, non-100% values in red. "
            "Bottom row: small blue squares labeled 'M' (manufacturer nodes), "
            "count per drug: CIM=3, FMD=3, GLD=2, GSR=2, HCT=3, IBU=4, MHE=3, MHL=3, MHR=5, total=28. "
            "Bottom banner: bold dark blue text 'Total: 28 Manufacturers  |  L2 Overall Accuracy: 97.22%'. "
            "Colors: purple #6C3483 for L1, dark blue #1A5276 for L2, blue #2471A3 for M nodes. "
            "Clean hierarchy, thin arrows, sans-serif font."
        )
    },
]

def fetch_panel(prompt: str, out_path: Path) -> bool:
    """调用 Gemini Imagen 生成单张子图。"""
    session = requests.Session()
    session.proxies = {'http': PROXY, 'https': PROXY}
    session.verify = False
    payload = {
        'instances': [{'prompt': prompt}],
        'parameters': {'sampleCount': 1, 'aspectRatio': '4:3'}
    }
    try:
        resp = session.post(IMAGEN_URL, json=payload, timeout=120)
        resp.raise_for_status()
        b64 = resp.json()['predictions'][0]['bytesBase64Encoded']
        out_path.write_bytes(base64.b64decode(b64))
        print(f'  [OK] {out_path.name}')
        return True
    except Exception as e:
        print(f'  [ERR] {out_path.name}: {e}')
        return False


def assemble_6panel(panel_paths: list, titles: list, out_path: Path):
    """将6张子图拼合为2x3学术版面，无边框，紧凑布局。"""
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             hspace=0.12, wspace=0.03,
                             left=0.005, right=0.995,
                             top=0.91, bottom=0.005)
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    subtitle_map = {
        'a': 'Dual-Modal Spectral Input',
        'b': 'SNV Preprocessing',
        'c': 'PI-VAE Encoder',
        'd': 'Physics-Informed Decoder',
        'e': 'Multimodal Feature Fusion',
        'f': 'Cascade Classifier  L1\u2192L2',
    }
    for idx, (path, title) in enumerate(zip(panel_paths, titles)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        # 完全去掉边框和刻度
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if path.exists():
            img = imread(str(path))
            ax.imshow(img, aspect='auto', interpolation='lanczos')
        else:
            ax.set_facecolor('#F5F5F5')
            ax.text(0.5, 0.5, 'Generating...',
                    ha='center', va='center', fontsize=10,
                    color='#aaa', transform=ax.transAxes)
        lbl = panel_labels[idx]
        # 子图字母标签：置于图像外左上角（axes 坐标系外，不遮挡内容）
        ax.text(-0.01, 1.04, f'({lbl})',
                transform=ax.transAxes,
                fontsize=15, fontweight='bold', color='#111',
                va='bottom', ha='left')
        # 子图标题：置于字母标签右侧同一行
        ax.set_title(subtitle_map[lbl], fontsize=10,
                     fontweight='bold', pad=6, color='#333', loc='center')

    fig.suptitle(
        'Figure 2.  PI-VAE Architecture: Dual-Modal Input → Physics-Informed VAE → Cascade Classifier',
        fontsize=11.5, fontweight='bold', y=0.972, color='#111'
    )
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    plt.close()
    print(f'[Assembled] -> {out_path}')


def main():
    print('=== Figure 2: PI-VAE Architecture (6-panel Gemini) ===')
    panel_paths = []
    titles      = []
    for i, p in enumerate(PANELS):
        out = SUB_DIR / p['file']
        print(f'[{i+1}/6] Generating panel ({p["key"]}): {p["title"]} ...')
        fetch_panel(p['prompt'], out)
        panel_paths.append(out)
        titles.append(p['title'])
        if i < len(PANELS) - 1:
            time.sleep(1)  # 避免 API 限速

    final = OUT_DIR / 'combined_fig2_arch_flow.png'
    assemble_6panel(panel_paths, titles, final)
    print(f'\n[DONE] Final figure -> {final}')


if __name__ == '__main__':
    main()


