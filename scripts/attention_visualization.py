"""
Transformer注意力权重可视化
分析PI-VAE中UV光谱Transformer编码器的注意力模式
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.upgraded_pi_vae import UpgradedPIVAE
from models.transformer_encoder import SpectralTransformerEncoder


def load_data(excel_path="Sampedata0.xlsx"):
    df_uv = pd.read_excel(excel_path, sheet_name="VIS_0", header=0)
    df_nir = pd.read_excel(excel_path, sheet_name="NIR_0", header=0)

    raw_labels = df_uv.iloc[:, 0].values
    unique_labels = sorted(set(raw_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    drug_labels = np.array([label_map[l] for l in raw_labels], dtype=int)
    drug_name_map = {i: str(l) for l, i in label_map.items()}
    uv_spectra = df_uv.iloc[:, 2:].values.astype(np.float32)
    nir_spectra = df_nir.iloc[:, 2:].values.astype(np.float32)

    # SNV归一化
    def snv(x):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    return snv(uv_spectra), snv(nir_spectra), drug_labels, drug_name_map


def load_model(ckpt_path="checkpoints_upgraded/best_model.pth"):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    t_cfg = cfg['model']['transformer']

    # 读取数据以获取维度
    uv_spectra, nir_spectra, _, _name_map = load_data(cfg['data']['excel_path'])
    uv_dim = uv_spectra.shape[1]
    nir_dim = nir_spectra.shape[1]

    model = UpgradedPIVAE(
        uv_dim=uv_dim,
        nir_dim=nir_dim,
        latent_dim=cfg['model']['latent_dim'],
        n_peaks=cfg['model']['n_peaks'],
        d_model=t_cfg['d_model'],
        n_heads=t_cfg['n_heads'],
        n_layers=t_cfg['n_layers'],
        d_ff=t_cfg['d_ff'],
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 重新加载数据返回完整4值
    uv_spectra, nir_spectra, drug_labels, drug_name_map = load_data(cfg['data']['excel_path'])
    return model, uv_spectra, nir_spectra, drug_labels, drug_name_map


def get_attention_weights(model, uv_tensor, max_samples=50, stride=8):
    """提取UV编码器所有层的注意力权重（降采样序列以节省内存）"""
    # 降采样序列长度：2012 -> ~252（stride=8）
    uv_sub = uv_tensor[:max_samples, ::stride]   # (N, L//stride)

    with torch.no_grad():
        x = uv_sub.unsqueeze(-1)                      # (N, L, 1)
        x = model.uv_encoder.input_projection(x)      # (N, L, d_model)
        # 位置编码需要重新创建（序列长度变了）
        from models.transformer_encoder import PositionalEncoding
        d_model = x.shape[-1]
        pe = PositionalEncoding(d_model, max_len=x.shape[1] + 1, dropout=0.0)
        x = pe(x)

        all_attn = []
        for layer in model.uv_encoder.encoder_layers:
            x, attn = layer(x, return_attention=True)
            all_attn.append(attn.cpu().numpy())       # (N, heads, L, L)
    return all_attn, uv_sub.shape[1]


def plot_attention_heatmaps(all_attn, drug_labels, uv_spectra, save_dir="figures", drug_name_map=None):
    """按药物分组绘制注意力热图"""
    os.makedirs(save_dir, exist_ok=True)

    drug_ids = sorted(set(drug_labels))
    drug_names = drug_name_map if drug_name_map else {
        0: "CIM", 1: "FMD", 2: "GLD", 3: "GSR", 4: "HCT",
        5: "IBU", 6: "MHE", 7: "MHL", 8: "MHR"}
    n_layers = len(all_attn)
    L = all_attn[0].shape[-1]  # seq_len

    # ---- 图1：每层平均注意力热图（取全样本平均）----
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    # 采样波段，显示前200个点（UV前200波长）
    sample_len = min(200, L)
    for li, attn in enumerate(all_attn):
        # attn: (B, H, L, L) -> 平均 heads 和 batch
        mean_attn = attn[:, :, :sample_len, :sample_len].mean(axis=(0, 1))
        im = axes[li].imshow(mean_attn, cmap='viridis', aspect='auto')
        axes[li].set_title(f"Layer {li+1}", fontsize=12, fontweight='bold')
        axes[li].set_xlabel("Key (wavelength index)", fontsize=9)
        if li == 0:
            axes[li].set_ylabel("Query (wavelength index)", fontsize=9)
        plt.colorbar(im, ax=axes[li], fraction=0.046, pad=0.04)

    fig.suptitle("UV Transformer Attention Weights (All Samples, Mean over Heads)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(save_dir, "attention_heatmap_layers.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")

    # ---- 图2：各药物平均注意力（最后一层，所有head平均）----
    last_attn = all_attn[-1]  # (B, H, L, L)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for di, drug_id in enumerate(drug_ids):
        idx = np.where(drug_labels == drug_id)[0]
        drug_attn = last_attn[idx]  # (n_samples, H, L, L)
        mean_drug = drug_attn[:, :, :sample_len, :sample_len].mean(axis=(0, 1))

        im = axes[di].imshow(mean_drug, cmap='hot', aspect='auto')
        axes[di].set_title(f"{drug_names.get(drug_id, f'Drug{drug_id}')} (n={len(idx)})",
                           fontsize=11, fontweight='bold')
        axes[di].set_xlabel("Key", fontsize=8)
        axes[di].set_ylabel("Query", fontsize=8)
        plt.colorbar(im, ax=axes[di], fraction=0.046, pad=0.04)

    fig.suptitle("Last Layer Attention per Drug Class (UV Encoder)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(save_dir, "attention_heatmap_per_drug.png")
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out2}")

    # ---- 图3：注意力权重在波长上的分布（CLS-style：每个token对其他token的总关注度）----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(drug_ids)))
    wavelengths = np.arange(sample_len)

    for li in [0, -1]:
        ax = axes[0] if li == 0 else axes[1]
        layer_name = "Layer 1" if li == 0 else f"Layer {n_layers}"
        attn_layer = all_attn[li]  # (B, H, L, L)

        for di, drug_id in enumerate(drug_ids):
            idx = np.where(drug_labels == drug_id)[0]
            # 列方向求和：每个wavelength被关注的总量
            attn_sum = attn_layer[idx][:, :, :sample_len, :sample_len]
            col_attn = attn_sum.mean(axis=(0, 1)).sum(axis=0)   # (sample_len,)
            col_attn /= col_attn.sum() + 1e-8
            ax.plot(wavelengths, col_attn, label=drug_names.get(drug_id, f"Drug{drug_id}"),
                    color=colors[di], alpha=0.7, linewidth=1.2)

        ax.set_title(f"{layer_name} - Attention Distribution over Wavelengths (per Drug)",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Wavelength Index (UV)", fontsize=10)
        ax.set_ylabel("Normalized Attention", fontsize=10)
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(save_dir, "attention_wavelength_distribution.png")
    plt.savefig(out3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out3}")

    return [out, out2, out3]


def main():
    print("Loading model and data...")
    model, uv_spectra, nir_spectra, drug_labels, drug_name_map = load_model()
    print(f"Model loaded. UV: {uv_spectra.shape}, Drugs: {set(drug_labels)}")

    uv_tensor = torch.FloatTensor(uv_spectra)

    print("Extracting attention weights (stride=8 downsampling)...")
    all_attn, seq_len = get_attention_weights(model, uv_tensor, max_samples=len(uv_spectra), stride=8)
    print(f"Extracted {len(all_attn)} layers, shape: {all_attn[0].shape}, seq_len: {seq_len}")

    print("Generating attention visualizations...")
    outputs = plot_attention_heatmaps(all_attn, drug_labels[:all_attn[0].shape[0]],
                                      uv_spectra, drug_name_map=drug_name_map)

    print("\nDone! Generated files:")
    for f in outputs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
