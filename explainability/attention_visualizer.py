"""
Attention Visualizer for Transformer-based PI-VAE
注意力权重可视化工具

This module extracts and visualizes attention weights from the Transformer encoder
to understand which spectral regions the model focuses on.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class AttentionVisualizer:
    """
    注意力权重可视化器

    功能:
    1. 提取多头注意力权重
    2. 生成注意力热图
    3. 标注关键波长
    4. 分析注意力模式
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化注意力可视化器

        Args:
            model: 训练好的PI-VAE模型（包含Transformer编码器）
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # 存储注意力权重
        self.attention_weights = {
            'uv': [],
            'nir': []
        }

    def extract_attention_weights(self, uv_spectra: torch.Tensor,
                                 nir_spectra: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        提取注意力权重

        Args:
            uv_spectra: UV光谱数据 (batch_size, uv_dim)
            nir_spectra: NIR光谱数据 (batch_size, nir_dim)

        Returns:
            attention_weights: 注意力权重字典
        """
        uv_spectra = uv_spectra.to(self.device)
        nir_spectra = nir_spectra.to(self.device)

        with torch.no_grad():
            # 前向传播并获取注意力权重
            outputs = self.model(uv_spectra, nir_spectra, return_attention=True)

            # 提取注意力权重
            if 'attention_weights_uv' in outputs:
                self.attention_weights['uv'] = outputs['attention_weights_uv']
            if 'attention_weights_nir' in outputs:
                self.attention_weights['nir'] = outputs['attention_weights_nir']

        return self.attention_weights

    def plot_attention_heatmap(self, attention_weights: torch.Tensor,
                              wavelengths: np.ndarray,
                              sample_idx: int = 0,
                              layer_idx: int = -1,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None):
        """
        绘制注意力热图

        Args:
            attention_weights: 注意力权重 (n_layers, batch_size, n_heads, seq_len, seq_len)
            wavelengths: 波长数组
            sample_idx: 样本索引
            layer_idx: 层索引（-1表示最后一层）
            figsize: 图像大小
            save_path: 保存路径
        """
        # 选择特定层和样本
        if isinstance(attention_weights, list):
            attn = attention_weights[layer_idx][sample_idx]  # (n_heads, seq_len, seq_len)
        else:
            attn = attention_weights[layer_idx, sample_idx]

        attn = attn.cpu().numpy()
        n_heads = attn.shape[0]

        # 创建子图
        n_cols = 4
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_heads > 1 else [axes]

        # 绘制每个注意力头
        for head_idx in range(n_heads):
            ax = axes[head_idx]

            # 绘制热图
            im = ax.imshow(attn[head_idx], cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx + 1}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Key Position', fontsize=9)
            ax.set_ylabel('Query Position', fontsize=9)

            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 隐藏多余的子图
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Multi-Head Attention Weights (Layer {layer_idx + 1}, Sample {sample_idx})',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")

        plt.show()

    def plot_average_attention(self, attention_weights: torch.Tensor,
                              wavelengths: np.ndarray,
                              layer_idx: int = -1,
                              figsize: Tuple[int, int] = (14, 6),
                              save_path: Optional[str] = None):
        """
        绘制平均注意力权重

        Args:
            attention_weights: 注意力权重
            wavelengths: 波长数组
            layer_idx: 层索引
            figsize: 图像大小
            save_path: 保存路径
        """
        # 选择特定层
        if isinstance(attention_weights, list):
            attn = attention_weights[layer_idx]  # (batch_size, n_heads, seq_len, seq_len)
        else:
            attn = attention_weights[layer_idx]

        attn = attn.cpu().numpy()

        # 平均所有样本和头
        avg_attn = np.mean(attn, axis=(0, 1))  # (seq_len, seq_len)

        # 计算每个位置的平均注意力（作为query）
        avg_attn_per_position = np.mean(avg_attn, axis=1)  # (seq_len,)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：平均注意力热图
        im1 = ax1.imshow(avg_attn, cmap='viridis', aspect='auto')
        ax1.set_title('Average Attention Weights', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Key Position', fontsize=10)
        ax1.set_ylabel('Query Position', fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 右图：每个位置的平均注意力
        ax2.plot(wavelengths, avg_attn_per_position, linewidth=2, color='steelblue')
        ax2.fill_between(wavelengths, avg_attn_per_position, alpha=0.3, color='steelblue')
        ax2.set_title('Average Attention per Wavelength', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        ax2.set_ylabel('Average Attention', fontsize=10)
        ax2.grid(alpha=0.3)

        # 标注Top-5关键波长
        top_indices = np.argsort(avg_attn_per_position)[-5:][::-1]
        for idx in top_indices:
            ax2.axvline(wavelengths[idx], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(wavelengths[idx], avg_attn_per_position[idx],
                    f'{wavelengths[idx]:.0f}nm',
                    fontsize=8, ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Average attention plot saved to {save_path}")

        plt.show()

    def plot_attention_rollout(self, attention_weights: List[torch.Tensor],
                              wavelengths: np.ndarray,
                              sample_idx: int = 0,
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None):
        """
        绘制注意力rollout（跨层累积注意力）

        Args:
            attention_weights: 所有层的注意力权重列表
            wavelengths: 波长数组
            sample_idx: 样本索引
            figsize: 图像大小
            save_path: 保存路径
        """
        # 计算attention rollout
        rollout = self._compute_attention_rollout(attention_weights, sample_idx)
        rollout = rollout.cpu().numpy()

        # 计算每个位置的累积注意力
        rollout_per_position = np.mean(rollout, axis=0)  # (seq_len,)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：Rollout热图
        im1 = ax1.imshow(rollout, cmap='hot', aspect='auto')
        ax1.set_title('Attention Rollout', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Key Position', fontsize=10)
        ax1.set_ylabel('Query Position', fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 右图：每个位置的累积注意力
        ax2.plot(wavelengths, rollout_per_position, linewidth=2, color='darkred')
        ax2.fill_between(wavelengths, rollout_per_position, alpha=0.3, color='darkred')
        ax2.set_title('Cumulative Attention per Wavelength', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        ax2.set_ylabel('Cumulative Attention', fontsize=10)
        ax2.grid(alpha=0.3)

        # 标注Top-10关键波长
        top_indices = np.argsort(rollout_per_position)[-10:][::-1]
        for idx in top_indices:
            ax2.axvline(wavelengths[idx], color='blue', linestyle='--', alpha=0.4, linewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention rollout plot saved to {save_path}")

        plt.show()

    def _compute_attention_rollout(self, attention_weights: List[torch.Tensor],
                                  sample_idx: int) -> torch.Tensor:
        """
        计算attention rollout

        Args:
            attention_weights: 所有层的注意力权重
            sample_idx: 样本索引

        Returns:
            rollout: 累积注意力矩阵
        """
        # 初始化为单位矩阵
        result = torch.eye(attention_weights[0].shape[-1]).to(self.device)

        # 逐层累积
        for layer_attn in attention_weights:
            # 平均所有头
            attn = layer_attn[sample_idx].mean(dim=0)  # (seq_len, seq_len)

            # 添加残差连接
            attn = attn + torch.eye(attn.shape[0]).to(self.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # 累积
            result = torch.matmul(attn, result)

        return result

    def get_top_k_attended_wavelengths(self, attention_weights: torch.Tensor,
                                      wavelengths: np.ndarray,
                                      k: int = 20,
                                      layer_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取Top-K被关注的波长

        Args:
            attention_weights: 注意力权重
            wavelengths: 波长数组
            k: 返回前k个
            layer_idx: 层索引

        Returns:
            top_wavelengths: Top-K波长
            top_attention: 对应的注意力值
        """
        # 选择特定层
        if isinstance(attention_weights, list):
            attn = attention_weights[layer_idx]
        else:
            attn = attention_weights[layer_idx]

        attn = attn.cpu().numpy()

        # 平均所有样本、头和query位置
        avg_attn = np.mean(attn, axis=(0, 1, 2))  # (seq_len,)

        # 获取Top-K
        top_indices = np.argsort(avg_attn)[-k:][::-1]
        top_wavelengths = wavelengths[top_indices]
        top_attention = avg_attn[top_indices]

        return top_wavelengths, top_attention

    def compare_attention_patterns(self, uv_attention: torch.Tensor,
                                  nir_attention: torch.Tensor,
                                  uv_wavelengths: np.ndarray,
                                  nir_wavelengths: np.ndarray,
                                  layer_idx: int = -1,
                                  figsize: Tuple[int, int] = (14, 6),
                                  save_path: Optional[str] = None):
        """
        比较UV和NIR的注意力模式

        Args:
            uv_attention: UV注意力权重
            nir_attention: NIR注意力权重
            uv_wavelengths: UV波长数组
            nir_wavelengths: NIR波长数组
            layer_idx: 层索引
            figsize: 图像大小
            save_path: 保存路径
        """
        # 提取平均注意力
        if isinstance(uv_attention, list):
            uv_attn = uv_attention[layer_idx].cpu().numpy()
            nir_attn = nir_attention[layer_idx].cpu().numpy()
        else:
            uv_attn = uv_attention[layer_idx].cpu().numpy()
            nir_attn = nir_attention[layer_idx].cpu().numpy()

        uv_avg = np.mean(uv_attn, axis=(0, 1, 2))
        nir_avg = np.mean(nir_attn, axis=(0, 1, 2))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # UV注意力
        ax1.plot(uv_wavelengths, uv_avg, linewidth=2, color='purple', label='UV-Vis')
        ax1.fill_between(uv_wavelengths, uv_avg, alpha=0.3, color='purple')
        ax1.set_title('UV-Vis Attention Pattern', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Wavelength (nm)', fontsize=10)
        ax1.set_ylabel('Average Attention', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # NIR注意力
        ax2.plot(nir_wavelengths, nir_avg, linewidth=2, color='darkgreen', label='NIR')
        ax2.fill_between(nir_wavelengths, nir_avg, alpha=0.3, color='darkgreen')
        ax2.set_title('NIR Attention Pattern', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        ax2.set_ylabel('Average Attention', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention comparison plot saved to {save_path}")

        plt.show()

    def generate_attention_report(self, uv_spectra: torch.Tensor,
                                 nir_spectra: torch.Tensor,
                                 uv_wavelengths: np.ndarray,
                                 nir_wavelengths: np.ndarray,
                                 output_dir: str = 'attention_reports'):
        """
        生成完整的注意力分析报告

        Args:
            uv_spectra: UV光谱数据
            nir_spectra: NIR光谱数据
            uv_wavelengths: UV波长数组
            nir_wavelengths: NIR波长数组
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        print("Generating attention analysis report...")

        # 提取注意力权重
        attention_weights = self.extract_attention_weights(uv_spectra, nir_spectra)

        # 1. UV注意力热图
        if attention_weights['uv']:
            self.plot_attention_heatmap(
                attention_weights['uv'],
                uv_wavelengths,
                sample_idx=0,
                save_path=os.path.join(output_dir, 'uv_attention_heatmap.png')
            )

        # 2. NIR注意力热图
        if attention_weights['nir']:
            self.plot_attention_heatmap(
                attention_weights['nir'],
                nir_wavelengths,
                sample_idx=0,
                save_path=os.path.join(output_dir, 'nir_attention_heatmap.png')
            )

        # 3. 平均注意力
        if attention_weights['uv']:
            self.plot_average_attention(
                attention_weights['uv'],
                uv_wavelengths,
                save_path=os.path.join(output_dir, 'uv_average_attention.png')
            )

        if attention_weights['nir']:
            self.plot_average_attention(
                attention_weights['nir'],
                nir_wavelengths,
                save_path=os.path.join(output_dir, 'nir_average_attention.png')
            )

        # 4. 注意力模式比较
        if attention_weights['uv'] and attention_weights['nir']:
            self.compare_attention_patterns(
                attention_weights['uv'],
                attention_weights['nir'],
                uv_wavelengths,
                nir_wavelengths,
                save_path=os.path.join(output_dir, 'attention_comparison.png')
            )

        # 5. Top-K关键波长
        if attention_weights['uv']:
            top_uv_wl, top_uv_attn = self.get_top_k_attended_wavelengths(
                attention_weights['uv'], uv_wavelengths, k=20
            )

            import pandas as pd
            df_uv = pd.DataFrame({
                'Wavelength (nm)': top_uv_wl,
                'Attention': top_uv_attn
            })
            df_uv.to_csv(os.path.join(output_dir, 'top_uv_wavelengths.csv'), index=False)

        if attention_weights['nir']:
            top_nir_wl, top_nir_attn = self.get_top_k_attended_wavelengths(
                attention_weights['nir'], nir_wavelengths, k=20
            )

            df_nir = pd.DataFrame({
                'Wavelength (nm)': top_nir_wl,
                'Attention': top_nir_attn
            })
            df_nir.to_csv(os.path.join(output_dir, 'top_nir_wavelengths.csv'), index=False)

        print(f"Attention analysis report generated in {output_dir}/")
