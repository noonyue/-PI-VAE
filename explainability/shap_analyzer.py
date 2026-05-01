"""
SHAP Analyzer for PI-VAE Model
SHAP分析工具

This module uses SHAP (SHapley Additive exPlanations) to identify the most important
wavelengths and features for drug classification.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import shap


class SHAPAnalyzer:
    """
    SHAP分析器，用于识别关键波长和特征

    功能:
    1. 计算特征重要性
    2. 提取Top-K关键波长
    3. 生成SHAP可视化图
    4. 映射到化学基团
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化SHAP分析器

        Args:
            model: 训练好的PI-VAE模型
            device: 计算设备 ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # SHAP explainer
        self.explainer = None
        self.shap_values = None
        self.base_values = None

    def create_explainer(self, background_data: torch.Tensor,
                        explainer_type: str = 'deep'):
        """
        创建SHAP解释器

        Args:
            background_data: 背景数据集，用于计算SHAP基线
            explainer_type: 解释器类型 ('deep', 'gradient', 'kernel')
        """
        background_data = background_data.to(self.device)

        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, background_data)
        elif explainer_type == 'kernel':
            # Kernel SHAP需要包装模型
            def model_predict(x):
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                return outputs['z_fused'].cpu().numpy()

            self.explainer = shap.KernelExplainer(
                model_predict,
                background_data.cpu().numpy()
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

        print(f"SHAP {explainer_type} explainer created successfully")

    def compute_shap_values(self, test_data: torch.Tensor,
                           max_samples: Optional[int] = None) -> np.ndarray:
        """
        计算SHAP值

        Args:
            test_data: 测试数据
            max_samples: 最大样本数（用于加速计算）

        Returns:
            shap_values: SHAP值数组
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")

        test_data = test_data.to(self.device)

        # 限制样本数
        if max_samples is not None and len(test_data) > max_samples:
            indices = np.random.choice(len(test_data), max_samples, replace=False)
            test_data = test_data[indices]

        print(f"Computing SHAP values for {len(test_data)} samples...")

        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(test_data)

        # 如果是多输出，取平均
        if isinstance(self.shap_values, list):
            self.shap_values = np.mean(self.shap_values, axis=0)

        print(f"SHAP values computed. Shape: {self.shap_values.shape}")
        return self.shap_values

    def get_feature_importance(self, aggregation: str = 'mean_abs') -> np.ndarray:
        """
        计算特征重要性

        Args:
            aggregation: 聚合方法 ('mean_abs', 'mean', 'max', 'std')

        Returns:
            importance: 特征重要性数组
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        if aggregation == 'mean_abs':
            importance = np.mean(np.abs(self.shap_values), axis=0)
        elif aggregation == 'mean':
            importance = np.mean(self.shap_values, axis=0)
        elif aggregation == 'max':
            importance = np.max(np.abs(self.shap_values), axis=0)
        elif aggregation == 'std':
            importance = np.std(self.shap_values, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return importance

    def get_top_k_wavelengths(self, wavelengths: np.ndarray, k: int = 20,
                             aggregation: str = 'mean_abs') -> Tuple[np.ndarray, np.ndarray]:
        """
        提取Top-K关键波长

        Args:
            wavelengths: 波长数组
            k: 返回前k个重要波长
            aggregation: 聚合方法

        Returns:
            top_wavelengths: Top-K波长
            top_importance: 对应的重要性值
        """
        importance = self.get_feature_importance(aggregation)

        # 获取Top-K索引
        top_indices = np.argsort(importance)[-k:][::-1]
        top_wavelengths = wavelengths[top_indices]
        top_importance = importance[top_indices]

        return top_wavelengths, top_importance

    def plot_feature_importance(self, wavelengths: np.ndarray,
                               top_k: int = 30,
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None):
        """
        绘制特征重要性图

        Args:
            wavelengths: 波长数组
            top_k: 显示前k个重要特征
            figsize: 图像大小
            save_path: 保存路径
        """
        importance = self.get_feature_importance('mean_abs')

        # 获取Top-K
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_wavelengths = wavelengths[top_indices]
        top_importance = importance[top_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(range(top_k), top_importance, color='steelblue', alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"{w:.1f} nm" for w in top_wavelengths])
        ax.set_xlabel('SHAP Importance (Mean |SHAP|)', fontsize=12)
        ax.set_ylabel('Wavelength', fontsize=12)
        ax.set_title(f'Top {top_k} Most Important Wavelengths', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_shap_summary(self, test_data: torch.Tensor,
                         wavelengths: np.ndarray,
                         max_display: int = 20,
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None):
        """
        绘制SHAP摘要图

        Args:
            test_data: 测试数据
            wavelengths: 波长数组
            max_display: 最多显示的特征数
            figsize: 图像大小
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")

        # 创建特征名称
        feature_names = [f"{w:.1f}nm" for w in wavelengths]

        # 绘制SHAP摘要图
        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.shap_values,
            test_data.cpu().numpy(),
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")

        plt.show()

    def plot_shap_waterfall(self, sample_idx: int,
                           test_data: torch.Tensor,
                           wavelengths: np.ndarray,
                           max_display: int = 15,
                           save_path: Optional[str] = None):
        """
        绘制单个样本的SHAP瀑布图

        Args:
            sample_idx: 样本索引
            test_data: 测试数据
            wavelengths: 波长数组
            max_display: 最多显示的特征数
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")

        # 创建特征名称
        feature_names = [f"{w:.1f}nm" for w in wavelengths]

        # 创建Explanation对象
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=np.mean(self.shap_values),
            data=test_data[sample_idx].cpu().numpy(),
            feature_names=feature_names
        )

        # 绘制瀑布图
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")

        plt.show()

    def map_to_chemical_groups(self, wavelengths: np.ndarray,
                               top_k: int = 20) -> Dict[str, List[float]]:
        """
        将关键波长映射到化学基团

        Args:
            wavelengths: 波长数组
            top_k: Top-K关键波长

        Returns:
            mapping: 化学基团映射字典
        """
        top_wavelengths, _ = self.get_top_k_wavelengths(wavelengths, k=top_k)

        # 化学基团波长范围 (示例，需根据实际情况调整)
        chemical_groups = {
            'Aromatic C-H': (700, 900),      # 芳香族C-H
            'Aliphatic C-H': (1100, 1250),   # 脂肪族C-H
            'C=O': (1650, 1750),             # 羰基
            'N-H': (1500, 1600),             # 氨基
            'O-H': (1400, 1450),             # 羟基
            'C-O': (1000, 1100),             # 醚键
            'Aromatic Ring': (1580, 1620),   # 芳香环
        }

        # 映射
        mapping = {group: [] for group in chemical_groups}

        for wl in top_wavelengths:
            for group, (min_wl, max_wl) in chemical_groups.items():
                if min_wl <= wl <= max_wl:
                    mapping[group].append(float(wl))

        # 移除空组
        mapping = {k: v for k, v in mapping.items() if v}

        return mapping

    def generate_report(self, wavelengths: np.ndarray,
                       test_data: torch.Tensor,
                       output_dir: str = 'explainability_reports'):
        """
        生成完整的SHAP分析报告

        Args:
            wavelengths: 波长数组
            test_data: 测试数据
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating SHAP analysis report...")

        # 1. 特征重要性图
        self.plot_feature_importance(
            wavelengths,
            top_k=30,
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )

        # 2. SHAP摘要图
        self.plot_shap_summary(
            test_data,
            wavelengths,
            max_display=20,
            save_path=os.path.join(output_dir, 'shap_summary.png')
        )

        # 3. Top-K关键波长
        top_wavelengths, top_importance = self.get_top_k_wavelengths(wavelengths, k=20)

        # 保存到CSV
        import pandas as pd
        df = pd.DataFrame({
            'Wavelength (nm)': top_wavelengths,
            'SHAP Importance': top_importance
        })
        df.to_csv(os.path.join(output_dir, 'top_wavelengths.csv'), index=False)

        # 4. 化学基团映射
        mapping = self.map_to_chemical_groups(wavelengths, top_k=20)

        # 保存映射
        with open(os.path.join(output_dir, 'chemical_group_mapping.txt'), 'w') as f:
            f.write("Chemical Group Mapping\n")
            f.write("=" * 50 + "\n\n")
            for group, wls in mapping.items():
                f.write(f"{group}:\n")
                for wl in wls:
                    f.write(f"  - {wl:.1f} nm\n")
                f.write("\n")

        print(f"SHAP analysis report generated in {output_dir}/")
