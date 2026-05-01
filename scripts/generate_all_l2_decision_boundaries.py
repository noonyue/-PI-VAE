"""
生成全部L2分类模型的决策边界大图
包含：PLS-DA, SVM, RandomForest (Direct策略) 和 RandomForest (Cascade策略)
使用t-SNE降维到2D空间展示决策边界
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import torch

# 添加父目录以导入pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi_vae_pipeline import load_data, preprocess_spectra, SpectralDataset, UV_VAE, NIR_VAE, train_vae, extract_latent_features

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# PLS-DA包装器
class PLSDAWrapper:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_numeric = np.array([np.where(self.classes_ == c)[0][0] for c in y])
        # 限制n_components不超过特征数和类别数
        max_comp = min(X.shape[1], len(self.classes_) - 1, self.n_components)
        n_comp = max(1, max_comp)
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(X, y_numeric)
        return self

    def predict(self, X):
        y_cont = self.model.predict(X).ravel()
        y_idx = np.clip(np.round(y_cont).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[y_idx]

def plot_decision_boundary(ax, clf, X_train, y_train, X_test, y_test, title, accuracy):
    """绘制决策边界"""
    # 创建网格
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测网格点
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab20')

    # 绘制训练点
    scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                               cmap='tab20', edgecolors='k', s=50, alpha=0.6,
                               marker='o', label='Train')

    # 绘制测试点
    scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                              cmap='tab20', edgecolors='k', s=80, alpha=0.8,
                              marker='s', label='Test')

    ax.set_title(f'{title}\n(Accuracy: {accuracy:.2%})', fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=9)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def main():
    print("Loading data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    uv_raw, nir_raw, drug_labels, manuf_labels = load_data('Sampedata0.xlsx')
    uv = preprocess_spectra(uv_raw, method='snv')
    nir = preprocess_spectra(nir_raw, method='snv')

    # 选择样本最多的药物进行可视化
    unique_drugs = np.unique(drug_labels)
    drug_counts = [(d, np.sum(drug_labels == d)) for d in unique_drugs]
    target_drug = max(drug_counts, key=lambda x: x[1])[0]

    print(f"Visualizing drug: {target_drug}")

    # 筛选目标药物
    mask = drug_labels == target_drug
    uv_drug = uv[mask]
    nir_drug = nir[mask]
    manuf_drug = manuf_labels[mask]

    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(manuf_drug)

    # 划分数据集
    X_raw = np.hstack([uv_drug, nir_drug])
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # 训练VAE获取潜在特征（用于Cascade策略）
    print("Training VAE for cascade features...")
    uv_loader = torch.utils.data.DataLoader(SpectralDataset(uv_drug[train_idx]), batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(SpectralDataset(nir_drug[train_idx]), batch_size=32, shuffle=True)

    uv_vae = UV_VAE(input_dim=uv_drug.shape[1], latent_dim=16, n_peaks=8)
    nir_vae = NIR_VAE(input_dim=nir_drug.shape[1], latent_dim=16, n_peaks=8)

    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=50, device=device, model_name="UV-VAE")
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=50, device=device, model_name="NIR-VAE")

    # 提取潜在特征
    uv_tr_loader = torch.utils.data.DataLoader(SpectralDataset(uv_drug[train_idx]), batch_size=64, shuffle=False)
    uv_te_loader = torch.utils.data.DataLoader(SpectralDataset(uv_drug[test_idx]), batch_size=64, shuffle=False)
    nir_tr_loader = torch.utils.data.DataLoader(SpectralDataset(nir_drug[train_idx]), batch_size=64, shuffle=False)
    nir_te_loader = torch.utils.data.DataLoader(SpectralDataset(nir_drug[test_idx]), batch_size=64, shuffle=False)

    z_uv_tr = extract_latent_features(uv_vae, uv_tr_loader, device)
    z_uv_te = extract_latent_features(uv_vae, uv_te_loader, device)
    z_nir_tr = extract_latent_features(nir_vae, nir_tr_loader, device)
    z_nir_te = extract_latent_features(nir_vae, nir_te_loader, device)

    X_train_latent = np.hstack([z_nir_tr, z_uv_tr])
    X_test_latent = np.hstack([z_nir_te, z_uv_te])

    # 创建大图：2行2列
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    models_config = [
        ('PLS-DA (Direct)', PLSDAWrapper(n_components=5), X_train_raw, X_test_raw, 'a'),
        ('SVM (Direct)', SVC(kernel='rbf', C=10, gamma='scale'), X_train_raw, X_test_raw, 'b'),
        ('RandomForest (Direct)', RandomForestClassifier(n_estimators=100, random_state=42), X_train_raw, X_test_raw, 'c'),
        ('RandomForest (Cascade)', RandomForestClassifier(n_estimators=100, random_state=42), X_train_latent, X_test_latent, 'd'),
    ]

    for idx, (model_name, clf, X_tr, X_te, panel_label) in enumerate(models_config):
        print(f"Processing {model_name}...")

        # t-SNE降维
        X_all = np.vstack([X_tr, X_te])
        y_all = np.concatenate([y_train, y_test])

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1))
        X_2d = tsne.fit_transform(X_all)

        X_train_2d = X_2d[:len(X_tr)]
        X_test_2d = X_2d[len(X_tr):]

        # 训练模型
        clf.fit(X_train_2d, y_train)
        y_pred = clf.predict(X_test_2d)
        accuracy = accuracy_score(y_test, y_pred)

        # 绘制决策边界
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        plot_decision_boundary(ax, clf, X_train_2d, y_train, X_test_2d, y_test,
                               f'({panel_label}) {model_name}', accuracy)

    # 添加总标题
    fig.suptitle(f'L2 Manufacturer Classification - Decision Boundaries (Drug: {target_drug})',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图片
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/all_l2_decision_boundaries.png', dpi=300, bbox_inches='tight')
    print("All L2 decision boundaries saved: figures/all_l2_decision_boundaries.png")
    plt.close()

if __name__ == '__main__':
    main()
