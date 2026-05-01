#!/usr/bin/env python3
"""
生成真实的OOD重建误差数据，替换figure8中的模拟数据。
从真实数据集中：
- In-distribution: 训练集上训练的VAE对测试集的重建误差
- OOD: 用合成扰动/不同分布样本的重建误差
将结果保存到 results/9-ood_real_errors.csv
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── 复用 pipeline 中的模型定义 ──────────────────────────────────────────────

class GaussianPeakDecoder(nn.Module):
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.n_peaks = n_peaks
        self.spectrum_dim = spectrum_dim
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, n_peaks * 3))
        self.register_buffer('wavelengths', torch.linspace(0, 1, spectrum_dim))

    def forward(self, z):
        B = z.size(0)
        p = self.fc_peaks(z).view(B, self.n_peaks, 3)
        pos = torch.sigmoid(p[:, :, 0])
        hgt = torch.abs(p[:, :, 1]) + 0.1
        wid = torch.abs(p[:, :, 2]) + 0.01
        spec = torch.zeros(B, self.spectrum_dim, device=z.device)
        for i in range(self.n_peaks):
            diff = self.wavelengths.unsqueeze(0) - pos[:, i:i+1]
            spec += hgt[:, i:i+1] * torch.exp(-0.5 * (diff / wid[:, i:i+1])**2)
        return spec

class LorentzianPeakDecoder(nn.Module):
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.n_peaks = n_peaks
        self.spectrum_dim = spectrum_dim
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, n_peaks * 3))
        self.register_buffer('wavelengths', torch.linspace(0, 1, spectrum_dim))

    def forward(self, z):
        B = z.size(0)
        p = self.fc_peaks(z).view(B, self.n_peaks, 3)
        pos = torch.sigmoid(p[:, :, 0])
        hgt = torch.abs(p[:, :, 1]) + 0.1
        wid = torch.abs(p[:, :, 2]) + 0.01
        spec = torch.zeros(B, self.spectrum_dim, device=z.device)
        for i in range(self.n_peaks):
            diff = self.wavelengths.unsqueeze(0) - pos[:, i:i+1]
            spec += hgt[:, i:i+1] / (1 + (diff / wid[:, i:i+1])**2)
        return spec

class UV_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, n_peaks=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2))
        self.decoder = GaussianPeakDecoder(latent_dim, n_peaks, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class NIR_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, n_peaks=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2))
        self.decoder = LorentzianPeakDecoder(latent_dim, n_peaks, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def snv(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std  = spectra.std(axis=1,  keepdims=True)
    return (spectra - mean) / (std + 1e-8)

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def train_vae(model, loader, epochs=150, lr=1e-3, device='cpu', name='VAE'):
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    best_loss, patience, wait = float('inf'), 20, 0
    print(f'  Training {name}...')
    for ep in range(epochs):
        model.train()
        total = 0
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader.dataset)
        if avg < best_loss:
            best_loss, wait = avg, 0
        else:
            wait += 1
            if wait >= patience:
                print(f'    Early stop at epoch {ep+1}')
                break
    return model

def per_sample_mse(model, X_np, device='cpu'):
    """计算每个样本的重建MSE"""
    model.eval()
    X = torch.FloatTensor(X_np).to(device)
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = X[i:i+64]
            recon, mu, _ = model(xb)
            mse = ((recon - xb)**2).mean(dim=1)  # per-sample MSE
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # 1. 加载数据
    print('Loading data...')
    df_uv  = pd.read_excel('Sampedata0.xlsx', sheet_name='VIS_0')
    df_nir = pd.read_excel('Sampedata0.xlsx', sheet_name='NIR_0')
    drug_labels = df_uv.iloc[:, 0].values
    uv_raw  = df_uv.iloc[:, 2:].values.astype(np.float32)
    nir_raw = df_nir.iloc[:, 2:].values.astype(np.float32)

    # 2. SNV 预处理
    uv  = snv(uv_raw)
    nir = snv(nir_raw)

    le = LabelEncoder()
    drug_enc = le.fit_transform(drug_labels)

    # 3. 训练/测试划分（与 pipeline 相同）
    idx = np.arange(len(uv))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                       stratify=drug_enc)

    uv_tr,  uv_te  = uv[tr_idx],  uv[te_idx]
    nir_tr, nir_te = nir[tr_idx], nir[te_idx]

    # 4. 构建 DataLoader
    def make_loader(X, shuffle=True):
        ds = TensorDataset(torch.FloatTensor(X))
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    uv_tr_ld  = make_loader(uv_tr,  shuffle=True)
    nir_tr_ld = make_loader(nir_tr, shuffle=True)

    # 5. 训练 UV-VAE + NIR-VAE
    uv_vae  = train_vae(UV_VAE(uv_tr.shape[1],  32, 10), uv_tr_ld,
                         epochs=150, device=device, name='UV-VAE')
    nir_vae = train_vae(NIR_VAE(nir_tr.shape[1], 32, 10), nir_tr_ld,
                         epochs=150, device=device, name='NIR-VAE')

    # 6. 计算 In-distribution 重建误差（测试集）
    print('Computing in-distribution reconstruction errors (test set)...')
    uv_mse_in  = per_sample_mse(uv_vae,  uv_te,  device)
    nir_mse_in = per_sample_mse(nir_vae, nir_te, device)
    err_in = (uv_mse_in + nir_mse_in) / 2   # 融合双模态均值误差
    n_in   = len(err_in)
    print(f'  In-dist samples: {n_in}, mean MSE: {err_in.mean():.4f}')

    # 7. 构造 OOD 样本（三种策略，模拟真实OOD情形）
    print('Generating OOD samples...')
    rng = np.random.default_rng(42)

    # 策略A: 强高斯噪声（SNR≈10dB）
    noise_scale = uv_te.std() * 3.0
    uv_ood_a  = uv_te  + rng.normal(0, noise_scale, uv_te.shape).astype(np.float32)
    nir_ood_a = nir_te + rng.normal(0, noise_scale, nir_te.shape).astype(np.float32)

    # 策略B: 频谱反转（完全异分布）
    uv_ood_b  = -uv_te
    nir_ood_b = -nir_te

    # 策略C: 随机打乱波长顺序（UV和NIR分别使用各自维度的permutation）
    perm_uv  = rng.permutation(uv_te.shape[1])
    perm_nir = rng.permutation(nir_te.shape[1])
    uv_ood_c  = uv_te[:, perm_uv]
    nir_ood_c = nir_te[:, perm_nir]

    # 合并 OOD
    uv_ood_all  = np.vstack([uv_ood_a,  uv_ood_b,  uv_ood_c])
    nir_ood_all = np.vstack([nir_ood_a, nir_ood_b, nir_ood_c])

    uv_mse_ood  = per_sample_mse(uv_vae,  uv_ood_all,  device)
    nir_mse_ood = per_sample_mse(nir_vae, nir_ood_all, device)
    err_ood = (uv_mse_ood + nir_mse_ood) / 2
    n_ood   = len(err_ood)
    print(f'  OOD samples: {n_ood}, mean MSE: {err_ood.mean():.4f}')

    # 8. 计算最优阈值与 AUC
    y_true  = np.array([0]*n_in + [1]*n_ood)
    scores  = np.concatenate([err_in, err_ood])
    auc     = roc_auc_score(y_true, scores)
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
    # 最大化 TPR-FPR 差值
    best_idx  = np.argmax(tpr_arr - fpr_arr)
    best_thr  = thresholds[best_idx]
    best_tpr  = tpr_arr[best_idx]
    best_fpr  = fpr_arr[best_idx]
    print(f'  AUC={auc:.4f}  Threshold={best_thr:.4f}  TPR={best_tpr:.4f}  FPR={best_fpr:.4f}')

    # 9. 保存误差分布数据（供 figure8 使用）
    os.makedirs('results', exist_ok=True)

    # 保存汇总指标（覆盖原9-ood_performance_metrics.csv）
    pd.DataFrame([{
        'AUC': round(float(auc), 4),
        'Best_Threshold': round(float(best_thr), 7),
        'TPR_at_best': round(float(best_tpr), 4),
        'FPR_at_best': round(float(best_fpr), 4)
    }]).to_csv('results/9-ood_performance_metrics.csv', index=False)
    print('Saved: results/9-ood_performance_metrics.csv')

    # 新增：保存逐样本误差分布（供直方图使用）
    df_in  = pd.DataFrame({'error': err_in,  'label': 'in-distribution'})
    df_out = pd.DataFrame({'error': err_ood, 'label': 'OOD'})
    df_all = pd.concat([df_in, df_out], ignore_index=True)
    df_all.to_csv('results/9-ood_error_distribution.csv', index=False)
    print(f'Saved: results/9-ood_error_distribution.csv  (n_in={n_in}, n_ood={n_ood})')

    print('\nDone.')


if __name__ == '__main__':
    main()
