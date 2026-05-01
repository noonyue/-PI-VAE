"""
Peak Parameter Interpreter for PI-VAE
峰参数解释器

This module interprets the physical meaning of peak parameters learned by the PI-VAE decoder
and generates human-readable explanations.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
import os


class PeakInterpreter:
    """
    峰参数解释器

    功能:
    1. 峰参数解耦分析
    2. 物理意义解释
    3. 生成解释报告（PDF/HTML）
    4. 峰参数统计分析
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化峰参数解释器

        Args:
            model: 训练好的PI-VAE模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # 峰参数存储
        self.peak_params = {
            'uv': None,
            'nir': None
        }

    def extract_peak_parameters(self, uv_spectra: torch.Tensor,
                               nir_spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取峰参数

        Args:
            uv_spectra: UV光谱数据
            nir_spectra: NIR光谱数据

        Returns:
            peak_params: 峰参数字典
        """
        uv_spectra = uv_spectra.to(self.device)
        nir_spectra = nir_spectra.to(self.device)

        with torch.no_grad():
            outputs = self.model(uv_spectra, nir_spectra)

            # 提取峰参数
            if 'uv_peaks' in outputs:
                self.peak_params['uv'] = outputs['uv_peaks']  # (batch, n_peaks, 3)
            if 'nir_peaks' in outputs:
                self.peak_params['nir'] = outputs['nir_peaks']  # (batch, n_peaks, 3)

        return self.peak_params

    def analyze_peak_statistics(self, peak_params: torch.Tensor,
                               peak_type: str = 'gaussian') -> Dict[str, np.ndarray]:
        """
        分析峰参数统计特性

        Args:
            peak_params: 峰参数 (batch, n_peaks, 3) - [position, height, width]
            peak_type: 峰类型 ('gaussian' or 'lorentzian')

        Returns:
            statistics: 统计信息字典
        """
        peak_params = peak_params.cpu().numpy()

        positions = peak_params[:, :, 0]  # (batch, n_peaks)
        heights = peak_params[:, :, 1]
        widths = peak_params[:, :, 2]

        statistics = {
            'position_mean': np.mean(positions, axis=0),
            'position_std': np.std(positions, axis=0),
            'position_min': np.min(positions, axis=0),
            'position_max': np.max(positions, axis=0),
            'height_mean': np.mean(heights, axis=0),
            'height_std': np.std(heights, axis=0),
            'height_min': np.min(heights, axis=0),
            'height_max': np.max(heights, axis=0),
            'width_mean': np.mean(widths, axis=0),
            'width_std': np.std(widths, axis=0),
            'width_min': np.min(widths, axis=0),
            'width_max': np.max(widths, axis=0),
        }

        return statistics

    def plot_peak_distribution(self, peak_params: torch.Tensor,
                              wavelengths: np.ndarray,
                              peak_type: str = 'gaussian',
                              figsize: Tuple[int, int] = (15, 5),
                              save_path: Optional[str] = None):
        """
        绘制峰参数分布图

        Args:
            peak_params: 峰参数
            wavelengths: 波长数组
            peak_type: 峰类型
            figsize: 图像大小
            save_path: 保存路径
        """
        peak_params = peak_params.cpu().numpy()
        positions = peak_params[:, :, 0]
        heights = peak_params[:, :, 1]
        widths = peak_params[:, :, 2]

        # 转换position到实际波长
        positions_wl = positions * (wavelengths[-1] - wavelengths[0]) + wavelengths[0]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 峰位置分布
        axes[0].violinplot([positions_wl[:, i] for i in range(positions_wl.shape[1])],
                          positions=range(positions_wl.shape[1]),
                          showmeans=True, showmedians=True)
        axes[0].set_xlabel('Peak Index', fontsize=11)
        axes[0].set_ylabel('Position (nm)', fontsize=11)
        axes[0].set_title('Peak Position Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # 峰高度分布
        axes[1].violinplot([heights[:, i] for i in range(heights.shape[1])],
                          positions=range(heights.shape[1]),
                          showmeans=True, showmedians=True)
        axes[1].set_xlabel('Peak Index', fontsize=11)
        axes[1].set_ylabel('Height', fontsize=11)
        axes[1].set_title('Peak Height Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # 峰宽度分布
        axes[2].violinplot([widths[:, i] for i in range(widths.shape[1])],
                          positions=range(widths.shape[1]),
                          showmeans=True, showmedians=True)
        axes[2].set_xlabel('Peak Index', fontsize=11)
        axes[2].set_ylabel('Width', fontsize=11)
        axes[2].set_title('Peak Width Distribution', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3)

        plt.suptitle(f'{peak_type.capitalize()} Peak Parameter Distribution',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Peak distribution plot saved to {save_path}")

        plt.show()

    def plot_latent_to_peak_mapping(self, latent_vectors: torch.Tensor,
                                   peak_params: torch.Tensor,
                                   param_idx: int = 0,
                                   figsize: Tuple[int, int] = (12, 8),
                                   save_path: Optional[str] = None):
        """
        绘制潜变量到峰参数的映射关系

        Args:
            latent_vectors: 潜变量 (batch, latent_dim)
            peak_params: 峰参数 (batch, n_peaks, 3)
            param_idx: 参数索引 (0=position, 1=height, 2=width)
            figsize: 图像大小
            save_path: 保存路径
        """
        latent_vectors = latent_vectors.cpu().numpy()
        peak_params = peak_params.cpu().numpy()

        param_names = ['Position', 'Height', 'Width']
        param_name = param_names[param_idx]

        # 选择前几个潜变量维度
        n_latent_dims = min(4, latent_vectors.shape[1])
        n_peaks = peak_params.shape[1]

        fig, axes = plt.subplots(n_latent_dims, n_peaks, figsize=figsize)
        if n_latent_dims == 1:
            axes = axes.reshape(1, -1)
        if n_peaks == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_latent_dims):
            for j in range(n_peaks):
                ax = axes[i, j]

                # 散点图
                ax.scatter(latent_vectors[:, i], peak_params[:, j, param_idx],
                          alpha=0.5, s=10, color='steelblue')

                # 拟合线
                z = np.polyfit(latent_vectors[:, i], peak_params[:, j, param_idx], 1)
                p = np.poly1d(z)
                x_line = np.linspace(latent_vectors[:, i].min(), latent_vectors[:, i].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, alpha=0.7)

                # 计算相关系数
                corr = np.corrcoef(latent_vectors[:, i], peak_params[:, j, param_idx])[0, 1]

                ax.set_xlabel(f'z[{i}]', fontsize=9)
                ax.set_ylabel(f'Peak {j+1} {param_name}', fontsize=9)
                ax.set_title(f'r={corr:.3f}', fontsize=9)
                ax.grid(alpha=0.3)

        plt.suptitle(f'Latent-to-Peak {param_name} Mapping',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent-to-peak mapping plot saved to {save_path}")

        plt.show()

    def interpret_peak_parameters(self, peak_params: torch.Tensor,
                                 wavelengths: np.ndarray,
                                 peak_type: str = 'gaussian') -> List[Dict]:
        """
        解释峰参数的物理意义

        Args:
            peak_params: 峰参数
            wavelengths: 波长数组
            peak_type: 峰类型

        Returns:
            interpretations: 解释列表
        """
        stats = self.analyze_peak_statistics(peak_params, peak_type)

        interpretations = []

        for i in range(len(stats['position_mean'])):
            # 转换到实际波长
            pos_wl = stats['position_mean'][i] * (wavelengths[-1] - wavelengths[0]) + wavelengths[0]

            interpretation = {
                'peak_index': i + 1,
                'position_nm': pos_wl,
                'position_std': stats['position_std'][i] * (wavelengths[-1] - wavelengths[0]),
                'height_mean': stats['height_mean'][i],
                'height_std': stats['height_std'][i],
                'width_mean': stats['width_mean'][i],
                'width_std': stats['width_std'][i],
                'peak_type': peak_type,
            }

            # 物理意义解释
            if peak_type == 'gaussian':
                interpretation['physical_meaning'] = self._interpret_gaussian_peak(pos_wl)
            else:
                interpretation['physical_meaning'] = self._interpret_lorentzian_peak(pos_wl)

            # 稳定性评估
            if stats['position_std'][i] < 0.05:
                interpretation['stability'] = 'High (consistent across samples)'
            elif stats['position_std'][i] < 0.15:
                interpretation['stability'] = 'Medium (moderate variation)'
            else:
                interpretation['stability'] = 'Low (high variation)'

            interpretations.append(interpretation)

        return interpretations

    def _interpret_gaussian_peak(self, wavelength: float) -> str:
        """解释高斯峰的物理意义（UV-Vis）"""
        if 200 <= wavelength < 300:
            return "UV region: Likely π→π* transitions in aromatic compounds"
        elif 300 <= wavelength < 400:
            return "Near-UV region: n→π* transitions, conjugated systems"
        elif 400 <= wavelength < 500:
            return "Visible blue region: Chromophores, colored compounds"
        elif 500 <= wavelength < 600:
            return "Visible green-yellow region: Extended conjugation"
        elif 600 <= wavelength < 700:
            return "Visible red region: Long conjugated systems"
        else:
            return "Near-IR region: Charge transfer transitions"

    def _interpret_lorentzian_peak(self, wavelength: float) -> str:
        """解释洛伦兹峰的物理意义（NIR）"""
        if 700 <= wavelength < 1100:
            return "NIR region: Aromatic C-H overtones and combinations"
        elif 1100 <= wavelength < 1400:
            return "NIR region: Aliphatic C-H second overtones"
        elif 1400 <= wavelength < 1600:
            return "NIR region: O-H and N-H first overtones"
        elif 1600 <= wavelength < 1800:
            return "NIR region: C=O first overtones"
        elif 1800 <= wavelength < 2200:
            return "NIR region: C-H, O-H, N-H combination bands"
        else:
            return "NIR region: Fundamental vibrations and combinations"

    def plot_peak_reconstruction(self, spectra: torch.Tensor,
                                reconstructed: torch.Tensor,
                                peak_params: torch.Tensor,
                                wavelengths: np.ndarray,
                                sample_idx: int = 0,
                                peak_type: str = 'gaussian',
                                figsize: Tuple[int, int] = (14, 6),
                                save_path: Optional[str] = None):
        """
        绘制峰重构分解图

        Args:
            spectra: 原始光谱
            reconstructed: 重构光谱
            peak_params: 峰参数
            wavelengths: 波长数组
            sample_idx: 样本索引
            peak_type: 峰类型
            figsize: 图像大小
            save_path: 保存路径
        """
        spectra = spectra[sample_idx].cpu().numpy()
        reconstructed = reconstructed[sample_idx].cpu().numpy()
        peak_params = peak_params[sample_idx].cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：原始 vs 重构
        ax1.plot(wavelengths, spectra, 'k-', linewidth=2, label='Original', alpha=0.7)
        ax1.plot(wavelengths, reconstructed, 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
        ax1.fill_between(wavelengths, spectra, reconstructed, alpha=0.2, color='gray')
        ax1.set_xlabel('Wavelength (nm)', fontsize=11)
        ax1.set_ylabel('Intensity', fontsize=11)
        ax1.set_title('Original vs Reconstructed Spectrum', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 右图：峰分解
        ax2.plot(wavelengths, reconstructed, 'k-', linewidth=2, label='Total', alpha=0.7)

        # 绘制每个峰
        colors = plt.cm.tab10(np.linspace(0, 1, len(peak_params)))
        for i, (pos, height, width) in enumerate(peak_params):
            # 生成单个峰
            pos_wl = pos * (wavelengths[-1] - wavelengths[0]) + wavelengths[0]

            if peak_type == 'gaussian':
                peak = height * np.exp(-0.5 * ((wavelengths - pos_wl) / (width * 100)) ** 2)
            else:  # lorentzian
                peak = height / (1 + ((wavelengths - pos_wl) / (width * 100)) ** 2)

            ax2.plot(wavelengths, peak, '--', color=colors[i], linewidth=1.5,
                    label=f'Peak {i+1} ({pos_wl:.0f}nm)', alpha=0.7)

        ax2.set_xlabel('Wavelength (nm)', fontsize=11)
        ax2.set_ylabel('Intensity', fontsize=11)
        ax2.set_title(f'{peak_type.capitalize()} Peak Decomposition', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Peak reconstruction plot saved to {save_path}")

        plt.show()

    def generate_interpretation_report(self, uv_spectra: torch.Tensor,
                                      nir_spectra: torch.Tensor,
                                      uv_wavelengths: np.ndarray,
                                      nir_wavelengths: np.ndarray,
                                      output_dir: str = 'interpretation_reports'):
        """
        生成完整的峰参数解释报告

        Args:
            uv_spectra: UV光谱数据
            nir_spectra: NIR光谱数据
            uv_wavelengths: UV波长数组
            nir_wavelengths: NIR波长数组
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        print("Generating peak parameter interpretation report...")

        # 提取峰参数
        peak_params = self.extract_peak_parameters(uv_spectra, nir_spectra)

        # 获取重构光谱
        with torch.no_grad():
            outputs = self.model(uv_spectra.to(self.device), nir_spectra.to(self.device))

        # 1. UV峰参数分布
        if peak_params['uv'] is not None:
            self.plot_peak_distribution(
                peak_params['uv'],
                uv_wavelengths,
                peak_type='gaussian',
                save_path=os.path.join(output_dir, 'uv_peak_distribution.png')
            )

        # 2. NIR峰参数分布
        if peak_params['nir'] is not None:
            self.plot_peak_distribution(
                peak_params['nir'],
                nir_wavelengths,
                peak_type='lorentzian',
                save_path=os.path.join(output_dir, 'nir_peak_distribution.png')
            )

        # 3. UV峰重构分解
        if peak_params['uv'] is not None and 'uv_recon' in outputs:
            self.plot_peak_reconstruction(
                uv_spectra,
                outputs['uv_recon'],
                peak_params['uv'],
                uv_wavelengths,
                sample_idx=0,
                peak_type='gaussian',
                save_path=os.path.join(output_dir, 'uv_peak_reconstruction.png')
            )

        # 4. NIR峰重构分解
        if peak_params['nir'] is not None and 'nir_recon' in outputs:
            self.plot_peak_reconstruction(
                nir_spectra,
                outputs['nir_recon'],
                peak_params['nir'],
                nir_wavelengths,
                sample_idx=0,
                peak_type='lorentzian',
                save_path=os.path.join(output_dir, 'nir_peak_reconstruction.png')
            )

        # 5. 生成文本解释
        if peak_params['uv'] is not None:
            uv_interpretations = self.interpret_peak_parameters(
                peak_params['uv'], uv_wavelengths, peak_type='gaussian'
            )
            df_uv = pd.DataFrame(uv_interpretations)
            df_uv.to_csv(os.path.join(output_dir, 'uv_peak_interpretations.csv'), index=False)

        if peak_params['nir'] is not None:
            nir_interpretations = self.interpret_peak_parameters(
                peak_params['nir'], nir_wavelengths, peak_type='lorentzian'
            )
            df_nir = pd.DataFrame(nir_interpretations)
            df_nir.to_csv(os.path.join(output_dir, 'nir_peak_interpretations.csv'), index=False)

        # 6. 生成HTML报告
        self._generate_html_report(uv_interpretations, nir_interpretations, output_dir)

        print(f"Peak parameter interpretation report generated in {output_dir}/")

    def _generate_html_report(self, uv_interpretations: List[Dict],
                             nir_interpretations: List[Dict],
                             output_dir: str):
        """生成HTML格式的解释报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Peak Parameter Interpretation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin-top: 15px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .peak-info { margin: 10px 0; padding: 10px; background-color: #ecf0f1; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Peak Parameter Interpretation Report</h1>
            <p>Generated by PI-VAE Explainability Module</p>
        """

        # UV峰解释
        html_content += "<h2>UV-Vis Gaussian Peaks</h2>"
        for interp in uv_interpretations:
            html_content += f"""
            <div class="peak-info">
                <h3>Peak {interp['peak_index']}</h3>
                <p><strong>Position:</strong> {interp['position_nm']:.1f} ± {interp['position_std']:.1f} nm</p>
                <p><strong>Height:</strong> {interp['height_mean']:.3f} ± {interp['height_std']:.3f}</p>
                <p><strong>Width:</strong> {interp['width_mean']:.3f} ± {interp['width_std']:.3f}</p>
                <p><strong>Physical Meaning:</strong> {interp['physical_meaning']}</p>
                <p><strong>Stability:</strong> {interp['stability']}</p>
            </div>
            """

        # NIR峰解释
        html_content += "<h2>NIR Lorentzian Peaks</h2>"
        for interp in nir_interpretations:
            html_content += f"""
            <div class="peak-info">
                <h3>Peak {interp['peak_index']}</h3>
                <p><strong>Position:</strong> {interp['position_nm']:.1f} ± {interp['position_std']:.1f} nm</p>
                <p><strong>Height:</strong> {interp['height_mean']:.3f} ± {interp['height_std']:.3f}</p>
                <p><strong>Width:</strong> {interp['width_mean']:.3f} ± {interp['width_std']:.3f}</p>
                <p><strong>Physical Meaning:</strong> {interp['physical_meaning']}</p>
                <p><strong>Stability:</strong> {interp['stability']}</p>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # 保存HTML
        with open(os.path.join(output_dir, 'interpretation_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report saved to {os.path.join(output_dir, 'interpretation_report.html')}")
