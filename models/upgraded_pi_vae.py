"""
Upgraded PI-VAE Model with Transformer Encoder
升级版PI-VAE模型（集成Transformer编码器、对比学习、物理约束）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_encoder import SpectralTransformerEncoder
from models.contrastive_loss import InfoNCELoss
from models.physics_loss import CombinedPhysicsLoss


class GaussianPeakDecoder(nn.Module):
    """
    高斯峰解码器（用于UV-Vis光谱）
    与原始PI-VAE保持一致
    """
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        self.spectrum_dim = spectrum_dim

        # 从潜变量映射到峰参数：位置、高度、宽度
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_peaks * 3)
        )

    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            reconstructed: (batch, spectrum_dim)
            peak_params: (batch, n_peaks, 3)
        """
        batch_size = z.size(0)

        # 生成峰参数
        peak_params = self.fc_peaks(z).view(batch_size, self.n_peaks, 3)

        # 应用激活函数确保参数合理
        positions = torch.sigmoid(peak_params[:, :, 0])  # [0, 1]
        heights = F.softplus(peak_params[:, :, 1])       # > 0
        widths = torch.sigmoid(peak_params[:, :, 2]) * 0.3 + 0.01  # [0.01, 0.31]

        peak_params = torch.stack([positions, heights, widths], dim=2)

        # 生成光谱
        x = torch.linspace(0, 1, self.spectrum_dim, device=z.device)
        x = x.view(1, 1, -1).expand(batch_size, self.n_peaks, -1)

        positions = positions.unsqueeze(2)
        heights = heights.unsqueeze(2)
        widths = widths.unsqueeze(2)

        # 高斯峰公式
        gaussian_peaks = heights * torch.exp(-((x - positions) ** 2) / (2 * widths ** 2))
        reconstructed = gaussian_peaks.sum(dim=1)

        return reconstructed, peak_params


class LorentzianPeakDecoder(nn.Module):
    """
    洛伦兹峰解码器（用于NIR光谱）
    与原始PI-VAE保持一致
    """
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        self.spectrum_dim = spectrum_dim

        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_peaks * 3)
        )

    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            reconstructed: (batch, spectrum_dim)
            peak_params: (batch, n_peaks, 3)
        """
        batch_size = z.size(0)

        peak_params = self.fc_peaks(z).view(batch_size, self.n_peaks, 3)

        positions = torch.sigmoid(peak_params[:, :, 0])
        heights = F.softplus(peak_params[:, :, 1])
        widths = torch.sigmoid(peak_params[:, :, 2]) * 0.3 + 0.01

        peak_params = torch.stack([positions, heights, widths], dim=2)

        x = torch.linspace(0, 1, self.spectrum_dim, device=z.device)
        x = x.view(1, 1, -1).expand(batch_size, self.n_peaks, -1)

        positions = positions.unsqueeze(2)
        heights = heights.unsqueeze(2)
        widths = widths.unsqueeze(2)

        # 洛伦兹峰公式
        lorentzian_peaks = heights * (widths ** 2) / ((x - positions) ** 2 + widths ** 2)
        reconstructed = lorentzian_peaks.sum(dim=1)

        return reconstructed, peak_params


class UpgradedPIVAE(nn.Module):
    """
    升级版PI-VAE模型

    主要改进：
    1. Transformer编码器替代MLP
    2. 集成对比学习损失
    3. 物理约束损失
    4. 多任务学习框架
    """
    def __init__(self,
                 uv_dim,
                 nir_dim,
                 latent_dim=32,
                 n_peaks=10,
                 # Transformer参数
                 d_model=256,
                 n_heads=8,
                 n_layers=4,
                 d_ff=512,
                 dropout=0.1,
                 # 损失权重
                 beta=1.0,
                 contrastive_weight=0.5,
                 physics_weight=0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.contrastive_weight = contrastive_weight
        self.physics_weight = physics_weight

        # Transformer编码器
        self.uv_encoder = SpectralTransformerEncoder(
            spectrum_dim=uv_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        self.nir_encoder = SpectralTransformerEncoder(
            spectrum_dim=nir_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # 物理先验解码器
        self.uv_decoder = GaussianPeakDecoder(latent_dim, n_peaks, uv_dim)
        self.nir_decoder = LorentzianPeakDecoder(latent_dim, n_peaks, nir_dim)

        # 损失函数
        self.contrastive_loss_fn = InfoNCELoss(temperature=0.07)
        self.physics_loss_uv = CombinedPhysicsLoss(peak_type='gaussian')
        self.physics_loss_nir = CombinedPhysicsLoss(peak_type='lorentzian')

    def encode(self, uv_spectra, nir_spectra, return_attention=False):
        """
        编码光谱数据

        Args:
            uv_spectra: (batch, uv_dim)
            nir_spectra: (batch, nir_dim)
            return_attention: 是否返回注意力权重

        Returns:
            z_uv, z_nir: 潜变量
            mu_uv, logvar_uv, mu_nir, logvar_nir: 分布参数
            attention_weights: 注意力权重（可选）
        """
        if return_attention:
            mu_uv, logvar_uv, attn_uv = self.uv_encoder(uv_spectra, return_attention=True)
            mu_nir, logvar_nir, attn_nir = self.nir_encoder(nir_spectra, return_attention=True)
        else:
            mu_uv, logvar_uv = self.uv_encoder(uv_spectra)
            mu_nir, logvar_nir = self.nir_encoder(nir_spectra)

        # 重参数化
        z_uv = self.uv_encoder.reparameterize(mu_uv, logvar_uv)
        z_nir = self.nir_encoder.reparameterize(mu_nir, logvar_nir)

        if return_attention:
            return z_uv, z_nir, mu_uv, logvar_uv, mu_nir, logvar_nir, attn_uv, attn_nir
        return z_uv, z_nir, mu_uv, logvar_uv, mu_nir, logvar_nir

    def decode(self, z_uv, z_nir):
        """
        解码潜变量

        Args:
            z_uv: (batch, latent_dim)
            z_nir: (batch, latent_dim)

        Returns:
            uv_recon, nir_recon: 重构光谱
            uv_peaks, nir_peaks: 峰参数
        """
        uv_recon, uv_peaks = self.uv_decoder(z_uv)
        nir_recon, nir_peaks = self.nir_decoder(z_nir)
        return uv_recon, nir_recon, uv_peaks, nir_peaks

    def forward(self, uv_spectra, nir_spectra, return_attention=False):
        """
        前向传播

        Args:
            uv_spectra: (batch, uv_dim)
            nir_spectra: (batch, nir_dim)
            return_attention: 是否返回注意力权重

        Returns:
            outputs: 包含重构光谱、潜变量、峰参数等的字典
        """
        # 编码
        if return_attention:
            z_uv, z_nir, mu_uv, logvar_uv, mu_nir, logvar_nir, attn_uv, attn_nir = \
                self.encode(uv_spectra, nir_spectra, return_attention=True)
        else:
            z_uv, z_nir, mu_uv, logvar_uv, mu_nir, logvar_nir = \
                self.encode(uv_spectra, nir_spectra)

        # 解码
        uv_recon, nir_recon, uv_peaks, nir_peaks = self.decode(z_uv, z_nir)

        outputs = {
            'uv_recon': uv_recon,
            'nir_recon': nir_recon,
            'z_uv': z_uv,
            'z_nir': z_nir,
            'mu_uv': mu_uv,
            'logvar_uv': logvar_uv,
            'mu_nir': mu_nir,
            'logvar_nir': logvar_nir,
            'uv_peaks': uv_peaks,
            'nir_peaks': nir_peaks
        }

        if return_attention:
            outputs['attention_uv'] = attn_uv
            outputs['attention_nir'] = attn_nir

        return outputs

    def compute_loss(self, uv_spectra, nir_spectra, labels,
                     uv_spectra_aug=None, nir_spectra_aug=None):
        """
        计算多任务损失

        Args:
            uv_spectra: 原始UV光谱 (batch, uv_dim)
            nir_spectra: 原始NIR光谱 (batch, nir_dim)
            labels: 样本标签 (batch,)
            uv_spectra_aug: 增强UV光谱（用于对比学习）
            nir_spectra_aug: 增强NIR光谱（用于对比学习）

        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        # 前向传播
        outputs = self.forward(uv_spectra, nir_spectra)

        # 1. 重构损失
        recon_loss_uv = F.mse_loss(outputs['uv_recon'], uv_spectra)
        recon_loss_nir = F.mse_loss(outputs['nir_recon'], nir_spectra)
        recon_loss = recon_loss_uv + recon_loss_nir

        # 2. KL散度损失
        kl_loss_uv = -0.5 * torch.sum(1 + outputs['logvar_uv'] - outputs['mu_uv'].pow(2) - outputs['logvar_uv'].exp())
        kl_loss_nir = -0.5 * torch.sum(1 + outputs['logvar_nir'] - outputs['mu_nir'].pow(2) - outputs['logvar_nir'].exp())
        kl_loss = (kl_loss_uv + kl_loss_nir) / uv_spectra.size(0)

        # 3. 对比学习损失（如果提供增强样本）
        contrastive_loss = torch.tensor(0.0, device=uv_spectra.device)
        if uv_spectra_aug is not None and nir_spectra_aug is not None:
            # 编码增强样本
            z_uv_aug, z_nir_aug, _, _, _, _ = self.encode(uv_spectra_aug, nir_spectra_aug)

            # 计算对比损失
            contrastive_loss_uv = self.contrastive_loss_fn(outputs['z_uv'], z_uv_aug, labels)
            contrastive_loss_nir = self.contrastive_loss_fn(outputs['z_nir'], z_nir_aug, labels)
            contrastive_loss = (contrastive_loss_uv + contrastive_loss_nir) / 2

        # 4. 物理约束损失
        physics_loss_uv, physics_dict_uv = self.physics_loss_uv(
            outputs['uv_recon'], uv_spectra, outputs['uv_peaks']
        )
        physics_loss_nir, physics_dict_nir = self.physics_loss_nir(
            outputs['nir_recon'], nir_spectra, outputs['nir_peaks']
        )
        physics_loss = (physics_loss_uv + physics_loss_nir) / 2

        # 总损失
        total_loss = (recon_loss +
                     self.beta * kl_loss +
                     self.contrastive_weight * contrastive_loss +
                     self.physics_weight * physics_loss)

        # 损失字典
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl_divergence': kl_loss.item(),
            'contrastive': contrastive_loss.item(),
            'physics': physics_loss.item(),
            'recon_uv': recon_loss_uv.item(),
            'recon_nir': recon_loss_nir.item(),
            'kl_uv': kl_loss_uv.item() / uv_spectra.size(0),
            'kl_nir': kl_loss_nir.item() / uv_spectra.size(0)
        }

        return total_loss, loss_dict


# 测试代码
if __name__ == "__main__":
    print("测试升级版PI-VAE模型...")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建模型
    uv_dim = 200
    nir_dim = 300
    batch_size = 16

    model = UpgradedPIVAE(
        uv_dim=uv_dim,
        nir_dim=nir_dim,
        latent_dim=32,
        n_peaks=10,
        d_model=256,
        n_heads=8,
        n_layers=4
    ).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建测试数据
    uv_spectra = torch.randn(batch_size, uv_dim).to(device)
    nir_spectra = torch.randn(batch_size, nir_dim).to(device)
    labels = torch.randint(0, 9, (batch_size,)).to(device)

    # 测试前向传播
    print("\n1. 测试前向传播")
    outputs = model(uv_spectra, nir_spectra)
    print(f"   UV重构形状: {outputs['uv_recon'].shape}")
    print(f"   NIR重构形状: {outputs['nir_recon'].shape}")
    print(f"   UV潜变量形状: {outputs['z_uv'].shape}")
    print(f"   NIR潜变量形状: {outputs['z_nir'].shape}")

    # 测试注意力权重提取
    print("\n2. 测试注意力权重提取")
    outputs = model(uv_spectra, nir_spectra, return_attention=True)
    print(f"   UV注意力层数: {len(outputs['attention_uv'])}")
    print(f"   NIR注意力层数: {len(outputs['attention_nir'])}")
    print(f"   注意力权重形状: {outputs['attention_uv'][0].shape}")

    # 测试损失计算（无对比学习）
    print("\n3. 测试损失计算（无对比学习）")
    total_loss, loss_dict = model.compute_loss(uv_spectra, nir_spectra, labels)
    print(f"   总损失: {total_loss.item():.4f}")
    print(f"   重构损失: {loss_dict['reconstruction']:.4f}")
    print(f"   KL损失: {loss_dict['kl_divergence']:.4f}")
    print(f"   物理损失: {loss_dict['physics']:.4f}")

    # 测试损失计算（有对比学习）
    print("\n4. 测试损失计算（有对比学习）")
    uv_spectra_aug = uv_spectra + torch.randn_like(uv_spectra) * 0.01
    nir_spectra_aug = nir_spectra + torch.randn_like(nir_spectra) * 0.01
    total_loss, loss_dict = model.compute_loss(
        uv_spectra, nir_spectra, labels,
        uv_spectra_aug, nir_spectra_aug
    )
    print(f"   总损失: {total_loss.item():.4f}")
    print(f"   对比学习损失: {loss_dict['contrastive']:.4f}")

    print("\n✓ 升级版PI-VAE模型测试通过！")
