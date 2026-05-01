"""
Physics-Informed Constraint Loss
物理约束损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失函数

    确保峰参数符合物理规律：
    1. 峰位在合理范围内 [0, 1]（归一化后的波长）
    2. 峰高非负
    3. 峰宽在合理范围内 [min_width, max_width]
    4. 峰位单调性（可选）

    应用：
    - 防止峰参数退化（如负峰宽、越界峰位）
    - 提升重构光谱的物理合理性
    - 增强模型可解释性
    """
    def __init__(self, min_width=0.01, max_width=0.5,
                 enforce_monotonicity=False,
                 weight_position=1.0,
                 weight_height=1.0,
                 weight_width=1.0,
                 weight_monotonicity=0.5):
        """
        Args:
            min_width: 最小峰宽
            max_width: 最大峰宽
            enforce_monotonicity: 是否强制峰位单调性
            weight_position: 峰位约束权重
            weight_height: 峰高约束权重
            weight_width: 峰宽约束权重
            weight_monotonicity: 单调性约束权重
        """
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.enforce_monotonicity = enforce_monotonicity
        self.weight_position = weight_position
        self.weight_height = weight_height
        self.weight_width = weight_width
        self.weight_monotonicity = weight_monotonicity

    def forward(self, peak_params):
        """
        计算物理约束损失

        Args:
            peak_params: (batch, n_peaks, 3) 峰参数 [position, height, width]

        Returns:
            loss: 物理约束损失标量
            loss_dict: 各项损失的字典（用于监控）
        """
        positions = peak_params[:, :, 0]  # (batch, n_peaks)
        heights = peak_params[:, :, 1]    # (batch, n_peaks)
        widths = peak_params[:, :, 2]     # (batch, n_peaks)

        loss_dict = {}

        # 约束1: 峰位在 [0, 1] 范围内
        loss_pos_lower = F.relu(-positions).mean()  # 惩罚负峰位
        loss_pos_upper = F.relu(positions - 1.0).mean()  # 惩罚超过1的峰位
        loss_position = (loss_pos_lower + loss_pos_upper) * self.weight_position
        loss_dict['position'] = loss_position.item()

        # 约束2: 峰高非负
        loss_height = F.relu(-heights).mean() * self.weight_height
        loss_dict['height'] = loss_height.item()

        # 约束3: 峰宽在 [min_width, max_width] 范围内
        loss_width_lower = F.relu(self.min_width - widths).mean()
        loss_width_upper = F.relu(widths - self.max_width).mean()
        loss_width = (loss_width_lower + loss_width_upper) * self.weight_width
        loss_dict['width'] = loss_width.item()

        # 总损失
        total_loss = loss_position + loss_height + loss_width

        # 约束4: 峰位单调性（可选）
        if self.enforce_monotonicity:
            # 惩罚非单调的峰位序列
            position_diffs = positions[:, 1:] - positions[:, :-1]  # (batch, n_peaks-1)
            loss_monotonicity = F.relu(-position_diffs).mean() * self.weight_monotonicity
            loss_dict['monotonicity'] = loss_monotonicity.item()
            total_loss = total_loss + loss_monotonicity
        else:
            loss_dict['monotonicity'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class SpectralReconstructionLoss(nn.Module):
    """
    光谱重构损失（结合物理约束）

    除了MSE重构损失，还考虑：
    1. 峰形质量（峰的平滑度）
    2. 基线合理性
    3. 光谱连续性
    """
    def __init__(self, mse_weight=1.0, smoothness_weight=0.1, baseline_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.baseline_weight = baseline_weight

    def forward(self, x_recon, x_target):
        """
        Args:
            x_recon: 重构光谱 (batch, spectrum_dim)
            x_target: 目标光谱 (batch, spectrum_dim)

        Returns:
            loss: 重构损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. MSE重构损失
        loss_mse = F.mse_loss(x_recon, x_target) * self.mse_weight
        loss_dict['mse'] = loss_mse.item()

        # 2. 平滑度损失（惩罚剧烈波动）
        # 计算一阶差分
        diff_recon = x_recon[:, 1:] - x_recon[:, :-1]
        diff_target = x_target[:, 1:] - x_target[:, :-1]
        loss_smoothness = F.mse_loss(diff_recon, diff_target) * self.smoothness_weight
        loss_dict['smoothness'] = loss_smoothness.item()

        # 3. 基线合理性（惩罚过大的负值）
        loss_baseline = F.relu(-x_recon).mean() * self.baseline_weight
        loss_dict['baseline'] = loss_baseline.item()

        # 总损失
        total_loss = loss_mse + loss_smoothness + loss_baseline
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class PeakShapeRegularization(nn.Module):
    """
    峰形正则化

    确保生成的峰形符合物理特性：
    - 高斯峰：对称、钟形
    - 洛伦兹峰：长尾、中心尖锐
    """
    def __init__(self, peak_type='gaussian', weight=0.1):
        """
        Args:
            peak_type: 'gaussian' 或 'lorentzian'
            weight: 正则化权重
        """
        super().__init__()
        self.peak_type = peak_type
        self.weight = weight

    def forward(self, reconstructed_spectrum, peak_params):
        """
        Args:
            reconstructed_spectrum: (batch, spectrum_dim)
            peak_params: (batch, n_peaks, 3)

        Returns:
            loss: 峰形正则化损失
        """
        # 检查峰的对称性（针对高斯峰）
        if self.peak_type == 'gaussian':
            # 计算光谱的二阶导数（曲率）
            second_derivative = reconstructed_spectrum[:, 2:] - 2 * reconstructed_spectrum[:, 1:-1] + reconstructed_spectrum[:, :-2]

            # 高斯峰的二阶导数应该平滑
            loss = torch.abs(second_derivative).mean() * self.weight

        elif self.peak_type == 'lorentzian':
            # 洛伦兹峰的尾部应该缓慢衰减
            # 检查尾部的斜率
            tail_slope = torch.abs(reconstructed_spectrum[:, 1:] - reconstructed_spectrum[:, :-1])
            loss = F.relu(tail_slope - 0.1).mean() * self.weight

        else:
            loss = torch.tensor(0.0, device=reconstructed_spectrum.device)

        return loss


class CombinedPhysicsLoss(nn.Module):
    """
    组合物理损失

    整合所有物理约束和正则化项
    """
    def __init__(self,
                 constraint_weight=1.0,
                 reconstruction_weight=1.0,
                 shape_regularization_weight=0.1,
                 peak_type='gaussian'):
        super().__init__()
        self.constraint_loss = PhysicsConstraintLoss()
        self.reconstruction_loss = SpectralReconstructionLoss()
        self.shape_regularization = PeakShapeRegularization(peak_type=peak_type)

        self.constraint_weight = constraint_weight
        self.reconstruction_weight = reconstruction_weight
        self.shape_regularization_weight = shape_regularization_weight

    def forward(self, x_recon, x_target, peak_params):
        """
        Args:
            x_recon: 重构光谱 (batch, spectrum_dim)
            x_target: 目标光谱 (batch, spectrum_dim)
            peak_params: 峰参数 (batch, n_peaks, 3)

        Returns:
            total_loss: 总物理损失
            loss_dict: 详细损失字典
        """
        # 1. 峰参数约束损失
        constraint_loss, constraint_dict = self.constraint_loss(peak_params)
        constraint_loss = constraint_loss * self.constraint_weight

        # 2. 重构损失
        recon_loss, recon_dict = self.reconstruction_loss(x_recon, x_target)
        recon_loss = recon_loss * self.reconstruction_weight

        # 3. 峰形正则化
        shape_loss = self.shape_regularization(x_recon, peak_params)
        shape_loss = shape_loss * self.shape_regularization_weight

        # 总损失
        total_loss = constraint_loss + recon_loss + shape_loss

        # 汇总损失字典
        loss_dict = {
            'constraint': constraint_dict,
            'reconstruction': recon_dict,
            'shape_regularization': shape_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


# 测试代码
if __name__ == "__main__":
    print("测试物理约束损失函数...")

    batch_size = 16
    n_peaks = 10
    spectrum_dim = 200

    # 创建测试数据
    peak_params = torch.randn(batch_size, n_peaks, 3)
    # 模拟一些违反约束的情况
    peak_params[:, 0, 0] = -0.5  # 负峰位
    peak_params[:, 1, 1] = -1.0  # 负峰高
    peak_params[:, 2, 2] = 0.8   # 过大峰宽

    x_recon = torch.randn(batch_size, spectrum_dim)
    x_target = torch.randn(batch_size, spectrum_dim)

    # 测试物理约束损失
    print("\n1. 测试物理约束损失")
    physics_loss = PhysicsConstraintLoss(enforce_monotonicity=True)
    loss, loss_dict = physics_loss(peak_params)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Loss Dict: {loss_dict}")

    # 测试光谱重构损失
    print("\n2. 测试光谱重构损失")
    recon_loss = SpectralReconstructionLoss()
    loss, loss_dict = recon_loss(x_recon, x_target)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Loss Dict: {loss_dict}")

    # 测试峰形正则化
    print("\n3. 测试峰形正则化")
    shape_reg = PeakShapeRegularization(peak_type='gaussian')
    loss = shape_reg(x_recon, peak_params)
    print(f"   Shape Regularization Loss: {loss.item():.4f}")

    # 测试组合物理损失
    print("\n4. 测试组合物理损失")
    combined_loss = CombinedPhysicsLoss(peak_type='gaussian')
    loss, loss_dict = combined_loss(x_recon, x_target, peak_params)
    print(f"   Total Combined Loss: {loss.item():.4f}")
    print(f"   Detailed Loss Dict:")
    print(f"     - Constraint: {loss_dict['constraint']}")
    print(f"     - Reconstruction: {loss_dict['reconstruction']}")
    print(f"     - Shape Regularization: {loss_dict['shape_regularization']:.4f}")

    print("\n✓ 物理约束损失函数测试通过！")
