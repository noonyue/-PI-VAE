"""
PI-VAE Pipeline for Drug and Manufacturer Identification
基于物理先验的变分自编码器光谱分析系统
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ==================== Data Loading and Preprocessing ====================

class SpectralDataset(Dataset):
    """光谱数据集类"""
    def __init__(self, spectra, labels=None):
        self.spectra = torch.FloatTensor(spectra)
        self.labels = labels
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.spectra[idx], self.labels[idx]
        return self.spectra[idx]

def load_data(excel_path='Sampedata0.xlsx'):
    """加载UV和NIR光谱数据"""
    print("Loading data from Excel...")
    
    # 读取UV和NIR数据
    df_uv = pd.read_excel(excel_path, sheet_name='VIS_0')
    df_nir = pd.read_excel(excel_path, sheet_name='NIR_0')
    
    # 提取标签（第一列：药品类型，第二列：厂家）
    drug_labels = df_uv.iloc[:, 0].values
    manufacturer_labels = df_uv.iloc[:, 1].values
    
    # 提取光谱数据（从第三列开始）
    uv_spectra = df_uv.iloc[:, 2:].values.astype(np.float32)
    nir_spectra = df_nir.iloc[:, 2:].values.astype(np.float32)
    
    print(f"UV spectra shape: {uv_spectra.shape}")
    print(f"NIR spectra shape: {nir_spectra.shape}")
    print(f"Number of unique drugs: {len(np.unique(drug_labels))}")
    print(f"Number of unique manufacturers: {len(np.unique(manufacturer_labels))}")
    
    return uv_spectra, nir_spectra, drug_labels, manufacturer_labels

def preprocess_spectra(spectra, method='snv'):
    """光谱预处理：SNV标准化或Z-score标准化"""
    if method == 'snv':
        # Standard Normal Variate (SNV)
        mean = spectra.mean(axis=1, keepdims=True)
        std = spectra.std(axis=1, keepdims=True)
        spectra_norm = (spectra - mean) / (std + 1e-8)
    elif method == 'zscore':
        # Z-score normalization
        scaler = StandardScaler()
        spectra_norm = scaler.fit_transform(spectra)
    else:
        spectra_norm = spectra
    
    return spectra_norm

# ==================== PI-VAE Models ====================

class GaussianPeakDecoder(nn.Module):
    """高斯峰解码器（用于UV-Vis光谱）"""
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        self.spectrum_dim = spectrum_dim
        
        # 从潜变量映射到峰参数：位置、高度、宽度
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_peaks * 3)  # 每个峰3个参数
        )
        
        # 波长网格（用于生成光谱）
        self.register_buffer('wavelengths', torch.linspace(0, 1, spectrum_dim))
    
    def forward(self, z):
        batch_size = z.size(0)
        params = self.fc_peaks(z)
        params = params.view(batch_size, self.n_peaks, 3)
        
        # 参数约束：位置[0,1], 高度>0, 宽度>0
        positions = torch.sigmoid(params[:, :, 0])  # 峰位置
        heights = torch.abs(params[:, :, 1]) + 0.1  # 峰高度
        widths = torch.abs(params[:, :, 2]) + 0.01  # 峰宽度
        
        # 生成高斯峰
        spectrum = torch.zeros(batch_size, self.spectrum_dim, device=z.device)
        for i in range(self.n_peaks):
            diff = self.wavelengths.unsqueeze(0) - positions[:, i:i+1]
            peak = heights[:, i:i+1] * torch.exp(-0.5 * (diff / widths[:, i:i+1]) ** 2)
            spectrum += peak
        
        return spectrum

class LorentzianPeakDecoder(nn.Module):
    """洛伦兹峰解码器（用于NIR光谱）"""
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
        
        self.register_buffer('wavelengths', torch.linspace(0, 1, spectrum_dim))
    
    def forward(self, z):
        batch_size = z.size(0)
        params = self.fc_peaks(z)
        params = params.view(batch_size, self.n_peaks, 3)
        
        positions = torch.sigmoid(params[:, :, 0])
        heights = torch.abs(params[:, :, 1]) + 0.1
        widths = torch.abs(params[:, :, 2]) + 0.01
        
        # 生成洛伦兹峰
        spectrum = torch.zeros(batch_size, self.spectrum_dim, device=z.device)
        for i in range(self.n_peaks):
            diff = self.wavelengths.unsqueeze(0) - positions[:, i:i+1]
            peak = heights[:, i:i+1] / (1 + (diff / widths[:, i:i+1]) ** 2)
            spectrum += peak
        
        return spectrum

class UV_VAE(nn.Module):
    """UV-Vis VAE with Gaussian prior"""
    def __init__(self, input_dim, latent_dim=32, n_peaks=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean and logvar
        )
        
        # Decoder (Gaussian peaks)
        self.decoder = GaussianPeakDecoder(latent_dim, n_peaks, input_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class NIR_VAE(nn.Module):
    """NIR VAE with Lorentzian prior"""
    def __init__(self, input_dim, latent_dim=32, n_peaks=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        
        self.decoder = LorentzianPeakDecoder(latent_dim, n_peaks, input_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE损失函数：重构误差 + KL散度"""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, epochs=100, lr=1e-3, device='cpu', model_name='VAE'):
    """训练VAE模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    
    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in train_loader:
            if isinstance(batch, tuple):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, train_losses

# ==================== Feature Extraction ====================

def extract_latent_features(model, data_loader, device='cpu'):
    """提取潜变量特征"""
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            mu, _ = model.encode(x)
            latent_features.append(mu.cpu().numpy())
    
    return np.vstack(latent_features)

# ==================== Cascade Classification ====================

def train_l1_classifier(X_train, y_train, X_test, y_test):
    """L1: 药品分类（使用融合特征）"""
    print("\n=== L1: Drug Classification ===")
    
    # 使用SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"L1 Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return svm, y_pred, accuracy

class PLSDAClassifier:
    """PLS-DA分类器包装类"""
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pls = None
        self.unique_classes = None
    
    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        # 将类别标签转换为数值（如果还不是数值）
        y_numeric = np.array([np.where(self.unique_classes == label)[0][0] for label in y])
        self.pls = PLSRegression(n_components=min(self.n_components, len(self.unique_classes)-1))
        self.pls.fit(X, y_numeric)
        return self
    
    def predict(self, X):
        y_pred_continuous = self.pls.predict(X).flatten()
        # 找到最近的类别
        y_pred_class = np.array([self.unique_classes[np.argmin(np.abs(self.unique_classes - val))] 
                                 for val in y_pred_continuous])
        return y_pred_class

def select_best_model(X_train, y_train, cv_method='loocv'):
    """通过交叉验证选择最佳模型（SVM, RF, PLS-DA）"""
    n_components = min(5, len(np.unique(y_train))-1)
    if n_components < 1:
        n_components = 1
    
    models = {
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'PLS-DA': PLSDAClassifier(n_components=n_components)
    }
    
    if cv_method == 'loocv' and len(X_train) <= 50:
        cv = LeaveOneOut()
    else:
        from sklearn.model_selection import StratifiedKFold
        n_splits = min(5, len(np.unique(y_train)), len(X_train) // 2)
        if n_splits < 2:
            n_splits = 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_model_name = None
    best_score = -1
    
    for name, model in models.items():
        scores = []
        try:
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                # 创建新模型实例以避免状态污染
                if name == 'SVM':
                    m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
                elif name == 'RandomForest':
                    m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                else:  # PLS-DA
                    m = PLSDAClassifier(n_components=n_components)
                
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
        except Exception as e:
            print(f"  Warning: {name} failed during CV: {e}")
            scores = [0.0]
        
        avg_score = np.mean(scores) if scores else 0.0
        print(f"  {name} CV Score: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = name
    
    # 训练最佳模型
    if best_model_name == 'SVM':
        best_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    elif best_model_name == 'RandomForest':
        best_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:  # PLS-DA
        best_model = PLSDAClassifier(n_components=n_components)
    
    best_model.fit(X_train, y_train)
    
    return best_model_name, best_model, best_score

def train_l2_classifiers(X_train, y_drug_train, y_manuf_train, 
                         X_test, y_drug_test, y_manuf_test):
    """L2: 厂家分类（为每种药品分别建立模型）"""
    print("\n=== L2: Manufacturer Identification ===")
    
    results = []
    all_predictions = []
    all_true = []
    
    unique_drugs = np.unique(y_drug_train)
    
    for drug in unique_drugs:
        # 获取该药品的训练和测试样本
        train_mask = y_drug_train == drug
        test_mask = y_drug_test == drug
        
        if not np.any(test_mask):
            continue
        
        X_drug_train = X_train[train_mask]
        y_manuf_drug_train = y_manuf_train[train_mask]
        X_drug_test = X_test[test_mask]
        y_manuf_drug_test = y_manuf_test[test_mask]
        
        print(f"\nDrug: {drug} (Train: {len(X_drug_train)}, Test: {len(X_drug_test)})")
        
        if len(X_drug_train) < 2:
            print(f"  Skipping: insufficient training samples")
            continue
        
        # 模型选择
        best_name, best_model, cv_score = select_best_model(
            X_drug_train, y_manuf_drug_train, 
            cv_method='loocv' if len(X_drug_train) <= 50 else 'kfold'
        )
        
        # 测试
        y_pred = best_model.predict(X_drug_test)
        
        accuracy = accuracy_score(y_manuf_drug_test, y_pred)
        
        print(f"  Best Model: {best_name} (CV: {cv_score:.4f}, Test: {accuracy:.4f})")
        
        results.append({
            'Drug': drug,
            'Best_Model': best_name,
            'CV_Score': cv_score,
            'Test_Accuracy': accuracy,
            'Train_Samples': len(X_drug_train),
            'Test_Samples': len(X_drug_test)
        })
        
        all_predictions.extend(y_pred)
        all_true.extend(y_manuf_drug_test)
    
    results_df = pd.DataFrame(results)
    return results_df, all_predictions, all_true

# ==================== Visualization ====================

def plot_pca_vs_vae(uv_features, nir_features, drug_labels, save_path='figures/pca_vs_vae.png'):
    """对比PCA和PI-VAE的潜空间"""
    import os
    import sys
    os.makedirs('figures', exist_ok=True)
    
    # Import plotting style
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
    from plotting_style import (
        create_multi_panel_figure, add_panel_label, format_axes,
        COLOR_TRUE, COLOR_PRED
    )
    
    # PCA
    pca_uv = PCA(n_components=2)
    pca_nir = PCA(n_components=2)
    pca_uv_coords = pca_uv.fit_transform(uv_features)
    pca_nir_coords = pca_nir.fit_transform(nir_features)
    
    # VAE features (already 2D or use PCA on latent)
    vae_uv_2d = PCA(n_components=2).fit_transform(uv_features)
    vae_nir_2d = PCA(n_components=2).fit_transform(nir_features)
    
    # Convert labels to numeric for color mapping
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    drug_y = le.fit_transform(drug_labels)
    
    # Create 2x2 multi-panel figure with reference style
    fig, gs = create_multi_panel_figure(nrows=2, ncols=2, figsize=(14, 12), 
                                        hspace=0.3, wspace=0.3)
    
    # PCA UV (a)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(pca_uv_coords[:, 0], pca_uv_coords[:, 1], 
                          c=drug_y, cmap='tab10', alpha=0.6, s=30, edgecolors='none')
    format_axes(ax1, xlabel='PC1', ylabel='PC2', 
               title='PCA: UV-Vis Spectra')
    add_panel_label(ax1, '(a)', x_offset=-0.12, y_offset=1.02)
    plt.colorbar(scatter1, ax=ax1)
    
    # VAE UV (b)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(vae_uv_2d[:, 0], vae_uv_2d[:, 1], 
                          c=drug_y, cmap='tab10', alpha=0.6, s=30, edgecolors='none')
    format_axes(ax2, xlabel='Latent Dimension 1', ylabel='Latent Dimension 2',
               title='PI-VAE: UV-Vis Latent Space')
    add_panel_label(ax2, '(b)', x_offset=-0.12, y_offset=1.02)
    plt.colorbar(scatter2, ax=ax2)
    
    # PCA NIR (c)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter3 = ax3.scatter(pca_nir_coords[:, 0], pca_nir_coords[:, 1], 
                          c=drug_y, cmap='tab10', alpha=0.6, s=30, edgecolors='none')
    format_axes(ax3, xlabel='PC1', ylabel='PC2',
               title='PCA: NIR Spectra')
    add_panel_label(ax3, '(c)', x_offset=-0.12, y_offset=1.02)
    plt.colorbar(scatter3, ax=ax3)
    
    # VAE NIR (d)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter4 = ax4.scatter(vae_nir_2d[:, 0], vae_nir_2d[:, 1], 
                          c=drug_y, cmap='tab10', alpha=0.6, s=30, edgecolors='none')
    format_axes(ax4, xlabel='Latent Dimension 1', ylabel='Latent Dimension 2',
               title='PI-VAE: NIR Latent Space')
    add_panel_label(ax4, '(d)', x_offset=-0.12, y_offset=1.02)
    plt.colorbar(scatter4, ax=ax4)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()

def plot_reconstruction(model, spectra, labels, n_samples=5, save_path='figures/reconstruction.png'):
    """绘制重构光谱对比"""
    import os
    import sys
    os.makedirs('figures', exist_ok=True)
    
    # Import plotting style
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
    from plotting_style import (
        create_multi_panel_figure, add_panel_label, format_axes,
        COLOR_TRUE, COLOR_PRED, COLOR_NIR
    )
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=n_samples, ncols=2, 
                                        figsize=(14, 3*n_samples), 
                                        hspace=0.3, wspace=0.3)
    
    indices = np.random.choice(len(spectra), n_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x = torch.FloatTensor(spectra[idx:idx+1]).to(device)
            recon, _, _, _ = model(x)
            
            x_np = x.cpu().numpy().flatten()
            recon_np = recon.cpu().numpy().flatten()
            
            # Original vs Reconstructed
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x_np, label='Original', color=COLOR_TRUE, alpha=0.8, linewidth=2)
            ax1.plot(recon_np, label='Reconstructed', color=COLOR_PRED, 
                    alpha=0.8, linestyle='--', linewidth=2)
            label_letter = f"({chr(97+2*i)})" if 2*i < 26 else f"({2*i+1})"
            format_axes(ax1, xlabel='Wavelength Index', ylabel='Intensity',
                       title=f'Sample {idx} (Label: {labels[idx]})')
            ax1.legend(fontsize=10)
            add_panel_label(ax1, label_letter, x_offset=-0.12, y_offset=1.02)
            
            # Residual
            residual = x_np - recon_np
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.plot(residual, color=COLOR_NIR, alpha=0.8, linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            label_letter2 = f"({chr(97+2*i+1)})" if 2*i+1 < 26 else f"({2*i+2})"
            format_axes(ax2, xlabel='Wavelength Index', ylabel='Residual',
                       title='Residual')
            add_panel_label(ax2, label_letter2, x_offset=-0.12, y_offset=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """绘制混淆矩阵"""
    import os
    import sys
    os.makedirs('figures', exist_ok=True)
    
    # Import plotting style
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
    from plotting_style import (
        create_multi_panel_figure, add_panel_label, format_axes,
        get_heatmap_colormap, COLOR_PRED
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create figure with reference style
    fig, gs = create_multi_panel_figure(nrows=1, ncols=1, figsize=(10, 8))
    ax = fig.add_subplot(gs[0, 0])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=get_heatmap_colormap(), 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
    format_axes(ax, xlabel='Predicted Label', ylabel='True Label', title=title)
    add_panel_label(ax, '(a)', x_offset=-0.12, y_offset=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ==================== Main Pipeline ====================

def main():
    """主流程"""
    print("="*60)
    print("PI-VAE Pipeline for Drug and Manufacturer Identification")
    print("="*60)
    
    # 1. 数据加载
    uv_spectra, nir_spectra, drug_labels, manufacturer_labels = load_data()
    
    # 2. 数据预处理
    print("\nPreprocessing spectra...")
    uv_spectra_norm = preprocess_spectra(uv_spectra, method='snv')
    nir_spectra_norm = preprocess_spectra(nir_spectra, method='snv')
    
    # 3. 标签编码
    drug_le = LabelEncoder()
    manuf_le = LabelEncoder()
    drug_labels_encoded = drug_le.fit_transform(drug_labels)
    manuf_labels_encoded = manuf_le.fit_transform(manufacturer_labels)
    
    print(f"\nDrug labels: {drug_le.classes_}")
    print(f"Manufacturer labels: {manuf_le.classes_}")
    
    # 4. 数据划分（80/20分层）
    print("\nSplitting data (80/20 stratified)...")
    indices = np.arange(len(uv_spectra))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=drug_labels_encoded
    )
    
    uv_train, uv_test = uv_spectra_norm[train_idx], uv_spectra_norm[test_idx]
    nir_train, nir_test = nir_spectra_norm[train_idx], nir_spectra_norm[test_idx]
    drug_train, drug_test = drug_labels_encoded[train_idx], drug_labels_encoded[test_idx]
    manuf_train, manuf_test = manuf_labels_encoded[train_idx], manuf_labels_encoded[test_idx]
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # 5. 训练VAE模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # UV-VAE
    uv_dataset = SpectralDataset(uv_train)
    uv_loader = DataLoader(uv_dataset, batch_size=32, shuffle=True)
    
    uv_vae = UV_VAE(input_dim=uv_spectra.shape[1], latent_dim=32, n_peaks=10)
    uv_vae, uv_losses = train_vae(uv_vae, uv_loader, epochs=200, lr=1e-3, 
                                   device=device, model_name='UV-VAE')
    
    # NIR-VAE
    nir_dataset = SpectralDataset(nir_train)
    nir_loader = DataLoader(nir_dataset, batch_size=32, shuffle=True)
    
    nir_vae = NIR_VAE(input_dim=nir_spectra.shape[1], latent_dim=32, n_peaks=10)
    nir_vae, nir_losses = train_vae(nir_vae, nir_loader, epochs=200, lr=1e-3, 
                                     device=device, model_name='NIR-VAE')
    
    # 6. 特征提取
    print("\nExtracting latent features...")
    uv_train_loader = DataLoader(SpectralDataset(uv_train), batch_size=32, shuffle=False)
    uv_test_loader = DataLoader(SpectralDataset(uv_test), batch_size=32, shuffle=False)
    nir_train_loader = DataLoader(SpectralDataset(nir_train), batch_size=32, shuffle=False)
    nir_test_loader = DataLoader(SpectralDataset(nir_test), batch_size=32, shuffle=False)
    
    z_uv_train = extract_latent_features(uv_vae, uv_train_loader, device)
    z_uv_test = extract_latent_features(uv_vae, uv_test_loader, device)
    z_nir_train = extract_latent_features(nir_vae, nir_train_loader, device)
    z_nir_test = extract_latent_features(nir_vae, nir_test_loader, device)
    
    # 特征融合
    X_train = np.hstack([z_nir_train, z_uv_train])
    X_test = np.hstack([z_nir_test, z_uv_test])
    
    print(f"Fused feature dimension: {X_train.shape[1]}")
    
    # 7. 级联分类
    # L1: 药品分类
    l1_model, l1_pred, l1_acc = train_l1_classifier(
        X_train, drug_train, X_test, drug_test
    )
    
    # L2: 厂家分类
    l2_results, l2_pred, l2_true = train_l2_classifiers(
        X_train, drug_train, manuf_train,
        X_test, drug_test, manuf_test
    )
    
    # 8. 可视化
    print("\nGenerating visualizations...")
    
    # PCA vs VAE对比
    plot_pca_vs_vae(uv_spectra_norm, nir_spectra_norm, drug_labels_encoded)
    
    # 重构对比
    plot_reconstruction(uv_vae, uv_test, drug_test, n_samples=5, 
                       save_path='figures/uv_reconstruction.png')
    plot_reconstruction(nir_vae, nir_test, drug_test, n_samples=5, 
                       save_path='figures/nir_reconstruction.png')
    
    # 混淆矩阵
    unique_drugs = np.unique(drug_test)
    plot_confusion_matrix(drug_test, l1_pred, unique_drugs, 
                         'L1: Drug Classification Confusion Matrix',
                         'figures/l1_confusion_matrix.png')
    
    # 9. 保存结果
    print("\nSaving results...")
    l2_results.to_csv('results/l2_classification_results.csv', index=False)
    print("Saved: results/l2_classification_results.csv")
    
    # 10. 消融实验
    print("\n=== Ablation Study ===")
    
    # 10.1 直接28分类 vs 级联
    print("\n10.1 Direct 28-class classification vs Cascade...")
    direct_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    direct_model.fit(X_train, manuf_train)
    direct_pred = direct_model.predict(X_test)
    direct_acc = accuracy_score(manuf_test, direct_pred)
    print(f"Direct 28-class accuracy: {direct_acc:.4f}")
    
    # 级联准确率（L2平均）
    cascade_acc = l2_results['Test_Accuracy'].mean()
    print(f"Cascade average accuracy: {cascade_acc:.4f}")
    print(f"Improvement: {cascade_acc - direct_acc:.4f}")
    
    # 10.2 无物理先验的自动编码器
    print("\n10.2 Standard Autoencoder (no physical prior) vs PI-VAE...")
    
    class StandardAE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z
        
        def encode(self, x):
            return self.encoder(x)
    
    # 训练标准AE
    def train_ae(model, train_loader, epochs=100, lr=1e-3, device='cpu'):
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                if isinstance(batch, tuple):
                    x, _ = batch
                else:
                    x = batch
                x = x.to(device)
                
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader.dataset):.4f}")
        
        return model
    
    print("Training standard AE for UV...")
    uv_ae = StandardAE(uv_spectra.shape[1], 32)
    uv_ae = train_ae(uv_ae, uv_loader, epochs=100, device=device)
    
    print("Training standard AE for NIR...")
    nir_ae = StandardAE(nir_spectra.shape[1], 32)
    nir_ae = train_ae(nir_ae, nir_loader, epochs=100, device=device)
    
    # 提取特征并分类
    def extract_ae_features(model, data_loader, device):
        model.eval()
        features = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    x, _ = batch
                else:
                    x = batch
                x = x.to(device)
                z = model.encode(x)
                features.append(z.cpu().numpy())
        return np.vstack(features)
    
    z_uv_ae_train = extract_ae_features(uv_ae, uv_train_loader, device)
    z_uv_ae_test = extract_ae_features(uv_ae, uv_test_loader, device)
    z_nir_ae_train = extract_ae_features(nir_ae, nir_train_loader, device)
    z_nir_ae_test = extract_ae_features(nir_ae, nir_test_loader, device)
    
    X_ae_train = np.hstack([z_nir_ae_train, z_uv_ae_train])
    X_ae_test = np.hstack([z_nir_ae_test, z_uv_ae_test])
    
    ae_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    ae_model.fit(X_ae_train, drug_train)
    ae_pred = ae_model.predict(X_ae_test)
    ae_acc = accuracy_score(drug_test, ae_pred)
    
    print(f"Standard AE (L1) accuracy: {ae_acc:.4f}")
    print(f"PI-VAE (L1) accuracy: {l1_acc:.4f}")
    print(f"Improvement: {l1_acc - ae_acc:.4f}")
    
    # 保存消融结果
    ablation_results = pd.DataFrame({
        'Method': ['Direct 28-class', 'Cascade', 'Standard AE (L1)', 'PI-VAE (L1)'],
        'Accuracy': [direct_acc, cascade_acc, ae_acc, l1_acc]
    })
    ablation_results.to_csv('results/ablation_study.csv', index=False)
    print("\nSaved: results/ablation_study.csv")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()

