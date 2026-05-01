"""
Training Script for Upgraded PI-VAE System
升级版PI-VAE系统训练脚本
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.upgraded_pi_vae import UpgradedPIVAE
from utils.augmentation import ContrastiveAugmentationPair


class SpectralDataset(Dataset):
    """光谱数据集"""
    def __init__(self, uv_spectra, nir_spectra, drug_labels, manufacturer_labels,
                 augmentation=None):
        self.uv_spectra = torch.FloatTensor(uv_spectra)
        self.nir_spectra = torch.FloatTensor(nir_spectra)
        self.drug_labels = torch.LongTensor(drug_labels)
        self.manufacturer_labels = torch.LongTensor(manufacturer_labels)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.uv_spectra)

    def __getitem__(self, idx):
        uv = self.uv_spectra[idx]
        nir = self.nir_spectra[idx]
        drug_label = self.drug_labels[idx]
        mfr_label = self.manufacturer_labels[idx]

        # 如果启用数据增强，生成增强样本对
        if self.augmentation is not None:
            uv_aug1, uv_aug2 = self.augmentation(uv.unsqueeze(0))
            nir_aug1, nir_aug2 = self.augmentation(nir.unsqueeze(0))
            return {
                'uv': uv,
                'nir': nir,
                'uv_aug': uv_aug1.squeeze(0),
                'nir_aug': nir_aug1.squeeze(0),
                'drug_label': drug_label,
                'mfr_label': mfr_label
            }

        return {
            'uv': uv,
            'nir': nir,
            'drug_label': drug_label,
            'mfr_label': mfr_label
        }


def load_data(config):
    """加载数据"""
    print("Loading data...")
    excel_path = config['data']['excel_path']

    # 读取UV和NIR数据
    df_uv = pd.read_excel(excel_path, sheet_name=config['data']['uv_sheet'])
    df_nir = pd.read_excel(excel_path, sheet_name=config['data']['nir_sheet'])

    # 提取标签
    drug_labels = df_uv.iloc[:, 0].values
    manufacturer_labels = df_uv.iloc[:, 1].values

    # 提取光谱数据
    uv_spectra = df_uv.iloc[:, 2:].values.astype(np.float32)
    nir_spectra = df_nir.iloc[:, 2:].values.astype(np.float32)

    # SNV预处理
    uv_spectra = preprocess_snv(uv_spectra)
    nir_spectra = preprocess_snv(nir_spectra)

    # 编码标签
    drug_encoder = LabelEncoder()
    mfr_encoder = LabelEncoder()
    drug_labels_encoded = drug_encoder.fit_transform(drug_labels)
    mfr_labels_encoded = mfr_encoder.fit_transform(manufacturer_labels)

    print(f"UV spectra shape: {uv_spectra.shape}")
    print(f"NIR spectra shape: {nir_spectra.shape}")
    print(f"Number of drugs: {len(np.unique(drug_labels_encoded))}")
    print(f"Number of manufacturers: {len(np.unique(mfr_labels_encoded))}")

    return (uv_spectra, nir_spectra, drug_labels_encoded, mfr_labels_encoded,
            drug_encoder, mfr_encoder)


def preprocess_snv(spectra):
    """SNV标准化"""
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    return (spectra - mean) / (std + 1e-8)


def create_dataloaders(uv_spectra, nir_spectra, drug_labels, mfr_labels, config):
    """创建数据加载器"""
    # 划分训练集和测试集
    indices = np.arange(len(uv_spectra))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - config['data']['train_test_split'],
        random_state=config['data']['random_seed'],
        stratify=drug_labels
    )

    # 创建增强器
    augmentation = None
    if config['augmentation']['enabled']:
        augmentation = ContrastiveAugmentationPair(
            augmentation_strength=config['augmentation']['strength']
        )

    # 创建数据集
    train_dataset = SpectralDataset(
        uv_spectra[train_idx],
        nir_spectra[train_idx],
        drug_labels[train_idx],
        mfr_labels[train_idx],
        augmentation=augmentation
    )

    test_dataset = SpectralDataset(
        uv_spectra[test_idx],
        nir_spectra[test_idx],
        drug_labels[test_idx],
        mfr_labels[test_idx],
        augmentation=None
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_idx, test_idx


def train_epoch(model, train_loader, optimizer, device, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_components = {
        'reconstruction': 0,
        'kl_divergence': 0,
        'contrastive': 0,
        'physics': 0
    }

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        uv = batch['uv'].to(device)
        nir = batch['nir'].to(device)
        drug_labels = batch['drug_label'].to(device)

        # 获取增强样本（如果有）
        uv_aug = batch.get('uv_aug', None)
        nir_aug = batch.get('nir_aug', None)
        if uv_aug is not None:
            uv_aug = uv_aug.to(device)
            nir_aug = nir_aug.to(device)

        # 前向传播和损失计算
        optimizer.zero_grad()
        loss, loss_dict = model.compute_loss(
            uv, nir, drug_labels, uv_aug, nir_aug
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()

        # 累积损失
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]

        # 更新进度条
        if batch_idx % config['logging']['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'kl': f"{loss_dict['kl_divergence']:.4f}"
            })

    # 计算平均损失
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches

    return avg_loss, loss_components


def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            uv = batch['uv'].to(device)
            nir = batch['nir'].to(device)
            drug_labels = batch['drug_label'].to(device)

            loss, _ = model.compute_loss(uv, nir, drug_labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, config, filename):
    """保存检查点"""
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def plot_training_curves(train_losses, val_losses, config):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    figure_dir = config['logging']['figure_dir']
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主训练函数"""
    # 加载配置
    with open('configs/upgraded_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    seed = config['data']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 设置设备
    device = 'cuda' if (config['device']['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    print(f"Using device: {device}")

    # 加载数据
    (uv_spectra, nir_spectra, drug_labels, mfr_labels,
     drug_encoder, mfr_encoder) = load_data(config)

    # 创建数据加载器
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        uv_spectra, nir_spectra, drug_labels, mfr_labels, config
    )

    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # 创建模型
    model = UpgradedPIVAE(
        uv_dim=uv_spectra.shape[1],
        nir_dim=nir_spectra.shape[1],
        latent_dim=config['model']['latent_dim'],
        n_peaks=config['model']['n_peaks'],
        d_model=config['model']['transformer']['d_model'],
        n_heads=config['model']['transformer']['n_heads'],
        n_layers=config['model']['transformer']['n_layers'],
        d_ff=config['model']['transformer']['d_ff'],
        dropout=config['model']['transformer']['dropout'],
        beta=config['model']['loss_weights']['beta'],
        contrastive_weight=config['model']['loss_weights']['contrastive_weight'],
        physics_weight=config['model']['loss_weights']['physics_weight']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
        min_lr=config['training']['scheduler']['min_lr']
    )

    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, config, epoch
        )
        train_losses.append(train_loss)

        # 评估
        val_loss = evaluate(model, test_loader, device)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印信息
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  - Reconstruction: {loss_components['reconstruction']:.4f}")
        print(f"  - KL Divergence: {loss_components['kl_divergence']:.4f}")
        print(f"  - Contrastive: {loss_components['contrastive']:.4f}")
        print(f"  - Physics: {loss_components['physics']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config, 'best_model.pth')
        else:
            patience_counter += 1

        # 定期保存检查点
        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config, f'checkpoint_epoch_{epoch}.pth')

        # 早停
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # 训练完成
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 保存最终模型
    save_checkpoint(model, optimizer, epoch, val_loss, config, 'final_model.pth')

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, config)

    print("\nTraining finished!")


if __name__ == "__main__":
    main()
