"""
Data Augmentation for Spectral Data
光谱数据增强模块（用于对比学习）
"""

import torch
import numpy as np


class SpectralAugmentation:
    """
    光谱数据增强工具类

    提供多种增强策略：
    1. 高斯噪声注入
    2. 光谱平移
    3. 基线漂移
    4. Mixup
    5. 强度缩放
    6. 波段遮蔽

    用途：
    - 对比学习的正样本对生成
    - 提升模型鲁棒性
    - 模拟真实世界的光谱变异
    """

    @staticmethod
    def add_gaussian_noise(spectra, noise_level=0.01):
        """
        添加高斯噪声

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            noise_level: 噪声标准差

        Returns:
            augmented_spectra: 添加噪声后的光谱
        """
        noise = torch.randn_like(spectra) * noise_level
        return spectra + noise

    @staticmethod
    def spectral_shift(spectra, max_shift=5):
        """
        光谱平移（模拟波长校准误差）

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            max_shift: 最大平移像素数

        Returns:
            shifted_spectra: 平移后的光谱
        """
        if len(spectra.shape) == 1:
            spectra = spectra.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, spectrum_dim = spectra.shape
        shifted_spectra = torch.zeros_like(spectra)

        for i in range(batch_size):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                shifted_spectra[i, shift:] = spectra[i, :-shift]
                shifted_spectra[i, :shift] = spectra[i, 0]  # 用边界值填充
            elif shift < 0:
                shifted_spectra[i, :shift] = spectra[i, -shift:]
                shifted_spectra[i, shift:] = spectra[i, -1]
            else:
                shifted_spectra[i] = spectra[i]

        if squeeze_output:
            shifted_spectra = shifted_spectra.squeeze(0)

        return shifted_spectra

    @staticmethod
    def baseline_drift(spectra, drift_type='linear', drift_strength=0.1):
        """
        基线漂移（模拟仪器漂移）

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            drift_type: 'linear', 'quadratic', 'sine'
            drift_strength: 漂移强度

        Returns:
            drifted_spectra: 添加基线漂移后的光谱
        """
        if len(spectra.shape) == 1:
            spectra = spectra.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, spectrum_dim = spectra.shape
        x = torch.linspace(0, 1, spectrum_dim, device=spectra.device)

        drifted_spectra = spectra.clone()

        for i in range(batch_size):
            if drift_type == 'linear':
                # 线性漂移
                slope = (torch.rand(1, device=spectra.device) - 0.5) * 2 * drift_strength
                drift = slope * x
            elif drift_type == 'quadratic':
                # 二次漂移
                a = (torch.rand(1, device=spectra.device) - 0.5) * 2 * drift_strength
                drift = a * (x - 0.5) ** 2
            elif drift_type == 'sine':
                # 正弦漂移
                freq = torch.rand(1, device=spectra.device) * 3 + 1
                drift = torch.sin(2 * np.pi * freq * x) * drift_strength
            else:
                drift = torch.zeros_like(x)

            drifted_spectra[i] = spectra[i] + drift

        if squeeze_output:
            drifted_spectra = drifted_spectra.squeeze(0)

        return drifted_spectra

    @staticmethod
    def mixup(spectra1, spectra2, alpha=0.2):
        """
        Mixup数据增强

        Args:
            spectra1: (batch, spectrum_dim)
            spectra2: (batch, spectrum_dim)
            alpha: Beta分布参数

        Returns:
            mixed_spectra: 混合后的光谱
            lam: 混合系数
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        mixed_spectra = lam * spectra1 + (1 - lam) * spectra2
        return mixed_spectra, lam

    @staticmethod
    def intensity_scaling(spectra, scale_range=(0.8, 1.2)):
        """
        强度缩放（模拟光源强度变化）

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            scale_range: 缩放范围 (min_scale, max_scale)

        Returns:
            scaled_spectra: 缩放后的光谱
        """
        min_scale, max_scale = scale_range
        scale = torch.rand(spectra.shape[0] if len(spectra.shape) > 1 else 1,
                          device=spectra.device) * (max_scale - min_scale) + min_scale

        if len(spectra.shape) == 1:
            return spectra * scale
        else:
            return spectra * scale.unsqueeze(1)

    @staticmethod
    def spectral_masking(spectra, mask_ratio=0.1, mask_value=0.0):
        """
        波段遮蔽（随机遮蔽部分波段）

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            mask_ratio: 遮蔽比例
            mask_value: 遮蔽值

        Returns:
            masked_spectra: 遮蔽后的光谱
        """
        masked_spectra = spectra.clone()

        if len(spectra.shape) == 1:
            spectrum_dim = spectra.shape[0]
            n_mask = int(spectrum_dim * mask_ratio)
            mask_indices = torch.randperm(spectrum_dim)[:n_mask]
            masked_spectra[mask_indices] = mask_value
        else:
            batch_size, spectrum_dim = spectra.shape
            n_mask = int(spectrum_dim * mask_ratio)
            for i in range(batch_size):
                mask_indices = torch.randperm(spectrum_dim)[:n_mask]
                masked_spectra[i, mask_indices] = mask_value

        return masked_spectra

    @staticmethod
    def random_augment(spectra, augmentation_prob=0.8):
        """
        随机应用多种增强策略

        Args:
            spectra: (batch, spectrum_dim) 或 (spectrum_dim,)
            augmentation_prob: 每种增强的应用概率

        Returns:
            augmented_spectra: 增强后的光谱
        """
        augmented = spectra.clone()

        # 1. 高斯噪声
        if torch.rand(1).item() < augmentation_prob:
            noise_level = torch.rand(1).item() * 0.02 + 0.005  # 0.005-0.025
            augmented = SpectralAugmentation.add_gaussian_noise(augmented, noise_level)

        # 2. 光谱平移
        if torch.rand(1).item() < augmentation_prob:
            max_shift = np.random.randint(3, 8)
            augmented = SpectralAugmentation.spectral_shift(augmented, max_shift)

        # 3. 基线漂移
        if torch.rand(1).item() < augmentation_prob:
            drift_type = np.random.choice(['linear', 'quadratic', 'sine'])
            drift_strength = torch.rand(1).item() * 0.15 + 0.05  # 0.05-0.2
            augmented = SpectralAugmentation.baseline_drift(augmented, drift_type, drift_strength)

        # 4. 强度缩放
        if torch.rand(1).item() < augmentation_prob:
            scale_range = (0.85, 1.15)
            augmented = SpectralAugmentation.intensity_scaling(augmented, scale_range)

        # 5. 波段遮蔽
        if torch.rand(1).item() < augmentation_prob * 0.5:  # 较低概率
            mask_ratio = torch.rand(1).item() * 0.1 + 0.05  # 0.05-0.15
            augmented = SpectralAugmentation.spectral_masking(augmented, mask_ratio)

        return augmented


class ContrastiveAugmentationPair:
    """
    对比学习的增强样本对生成器

    为每个样本生成两个不同的增强视图
    """
    def __init__(self, augmentation_strength='moderate'):
        """
        Args:
            augmentation_strength: 'weak', 'moderate', 'strong'
        """
        self.augmentation_strength = augmentation_strength

        if augmentation_strength == 'weak':
            self.aug_prob = 0.5
        elif augmentation_strength == 'moderate':
            self.aug_prob = 0.8
        elif augmentation_strength == 'strong':
            self.aug_prob = 1.0
        else:
            self.aug_prob = 0.8

    def __call__(self, spectra):
        """
        生成增强样本对

        Args:
            spectra: (batch, spectrum_dim)

        Returns:
            view1: 第一个增强视图
            view2: 第二个增强视图
        """
        view1 = SpectralAugmentation.random_augment(spectra, self.aug_prob)
        view2 = SpectralAugmentation.random_augment(spectra, self.aug_prob)

        return view1, view2


# 测试代码
if __name__ == "__main__":
    print("测试光谱数据增强模块...")

    # 创建测试数据
    batch_size = 8
    spectrum_dim = 200
    spectra = torch.randn(batch_size, spectrum_dim)

    # 测试各种增强方法
    print("\n1. 测试高斯噪声")
    noisy = SpectralAugmentation.add_gaussian_noise(spectra, noise_level=0.01)
    print(f"   Original shape: {spectra.shape}, Noisy shape: {noisy.shape}")
    print(f"   Noise std: {(noisy - spectra).std().item():.4f}")

    print("\n2. 测试光谱平移")
    shifted = SpectralAugmentation.spectral_shift(spectra, max_shift=5)
    print(f"   Shifted shape: {shifted.shape}")

    print("\n3. 测试基线漂移")
    for drift_type in ['linear', 'quadratic', 'sine']:
        drifted = SpectralAugmentation.baseline_drift(spectra, drift_type=drift_type)
        print(f"   {drift_type} drift shape: {drifted.shape}")

    print("\n4. 测试Mixup")
    spectra2 = torch.randn(batch_size, spectrum_dim)
    mixed, lam = SpectralAugmentation.mixup(spectra, spectra2, alpha=0.2)
    print(f"   Mixed shape: {mixed.shape}, Lambda: {lam:.4f}")

    print("\n5. 测试强度缩放")
    scaled = SpectralAugmentation.intensity_scaling(spectra, scale_range=(0.8, 1.2))
    print(f"   Scaled shape: {scaled.shape}")

    print("\n6. 测试波段遮蔽")
    masked = SpectralAugmentation.spectral_masking(spectra, mask_ratio=0.1)
    print(f"   Masked shape: {masked.shape}")
    print(f"   Masked values: {(masked == 0).sum().item()} / {masked.numel()}")

    print("\n7. 测试随机增强")
    augmented = SpectralAugmentation.random_augment(spectra, augmentation_prob=0.8)
    print(f"   Augmented shape: {augmented.shape}")

    print("\n8. 测试对比学习增强样本对")
    pair_generator = ContrastiveAugmentationPair(augmentation_strength='moderate')
    view1, view2 = pair_generator(spectra)
    print(f"   View1 shape: {view1.shape}")
    print(f"   View2 shape: {view2.shape}")
    print(f"   View1 vs View2 difference: {(view1 - view2).abs().mean().item():.4f}")

    print("\n✓ 光谱数据增强模块测试通过！")
