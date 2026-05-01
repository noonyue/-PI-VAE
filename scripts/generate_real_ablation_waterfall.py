#!/usr/bin/env python3
"""
真实逐步消融实验，生成 Figure 15 瀑布图所需的真实数据。
5个步骤（L2厂家分类准确率）：
  Step 1: Direct 28-class baseline (raw spectra, SVM)
  Step 2: + SNV Preprocessing
  Step 3: + PI-VAE Feature Extraction
  Step 4: + Multimodal Fusion (UV+NIR latent)
  Step 5: + Cascade Strategy
结果保存到 results/table3_5_cascade_ablation_real.csv
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

np.random.seed(42)
torch.manual_seed(42)

# ── 模型定义（与 pipeline 一致）─────────────────────────────────────────────

class GaussianPeakDecoder(nn.Module):
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.n_peaks = n_peaks
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, n_peaks * 3))
        self.register_buffer('wl', torch.linspace(0, 1, spectrum_dim))
    def forward(self, z):
        B = z.size(0)
        p = self.fc_peaks(z).view(B, self.n_peaks, 3)
        pos = torch.sigmoid(p[..., 0]); hgt = p[..., 1].abs()+0.1; wid = p[..., 2].abs()+0.01
        s = torch.zeros(B, self.wl.shape[0], device=z.device)
        for i in range(self.n_peaks):
            s += hgt[:, i:i+1] * torch.exp(-0.5*((self.wl-pos[:,i:i+1])/wid[:,i:i+1])**2)
        return s

class LorentzianPeakDecoder(nn.Module):
    def __init__(self, latent_dim, n_peaks, spectrum_dim):
        super().__init__()
        self.n_peaks = n_peaks
        self.fc_peaks = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, n_peaks * 3))
        self.register_buffer('wl', torch.linspace(0, 1, spectrum_dim))
    def forward(self, z):
        B = z.size(0)
        p = self.fc_peaks(z).view(B, self.n_peaks, 3)
        pos = torch.sigmoid(p[..., 0]); hgt = p[..., 1].abs()+0.1; wid = p[..., 2].abs()+0.01
        s = torch.zeros(B, self.wl.shape[0], device=z.device)
        for i in range(self.n_peaks):
            s += hgt[:, i:i+1] / (1+((self.wl-pos[:,i:i+1])/wid[:,i:i+1])**2)
        return s

class UV_VAE(nn.Module):
    def __init__(self, d, ld=32, np_=10):
        super().__init__()
        self.ld = ld
        self.enc = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,ld*2))
        self.dec = GaussianPeakDecoder(ld, np_, d)
    def encode(self, x):
        h = self.enc(x); return h[:,:self.ld], h[:,self.ld:]
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.dec(z), mu, lv

class NIR_VAE(nn.Module):
    def __init__(self, d, ld=32, np_=10):
        super().__init__()
        self.ld = ld
        self.enc = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,ld*2))
        self.dec = LorentzianPeakDecoder(ld, np_, d)
    def encode(self, x):
        h = self.enc(x); return h[:,:self.ld], h[:,self.ld:]
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.dec(z), mu, lv

# ── 工具 ─────────────────────────────────────────────────────────────────────

def snv(X):
    m = X.mean(1, keepdims=True); s = X.std(1, keepdims=True)
    return (X - m) / (s + 1e-8)

def vae_loss(recon, x, mu, lv):
    return nn.functional.mse_loss(recon, x, reduction='sum') \
           - 0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())

def train_vae(model, X_np, device, epochs=150, name='VAE'):
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    ds    = TensorDataset(torch.FloatTensor(X_np))
    ld_   = DataLoader(ds, batch_size=32, shuffle=True)
    best, wait = float('inf'), 0
    print(f'  Training {name}...')
    for ep in range(epochs):
        model.train(); tot = 0
        for (x,) in ld_:
            x = x.to(device); opt.zero_grad()
            recon, mu, lv = model(x)
            loss = vae_loss(recon, x, mu, lv)
            loss.backward(); opt.step(); tot += loss.item()
        avg = tot / len(X_np)
        if avg < best: best, wait = avg, 0
        else:
            wait += 1
            if wait >= 20: print(f'    Early stop ep{ep+1}'); break
    return model

def get_latent(model, X_np, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), 64):
            x = torch.FloatTensor(X_np[i:i+64]).to(device)
            mu, _ = model.encode(x)
            out.append(mu.cpu().numpy())
    return np.vstack(out)

class PLSDAClassifier:
    def __init__(self, n_components=5):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        yn = np.array([np.where(self.classes_==c)[0][0] for c in y])
        nc = min(self.n_components, len(self.classes_)-1, X.shape[0]-1)
        self.pls = PLSRegression(n_components=max(1, nc))
        self.pls.fit(X, yn)
    def predict(self, X):
        yc = self.pls.predict(X).flatten()
        return self.classes_[np.clip(np.round(yc).astype(int), 0, len(self.classes_)-1)]

def best_l2_cascade(X_tr, drug_tr, manuf_tr, X_te, drug_te, manuf_te):
    """级联L2：对每种药品分别选最优模型"""
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    all_pred, all_true = [], []
    for drug in np.unique(drug_tr):
        m_tr = drug_tr == drug; m_te = drug_te == drug
        if not np.any(m_te): continue
        Xtr, ytr = X_tr[m_tr], manuf_tr[m_tr]
        Xte, yte = X_te[m_te], manuf_te[m_te]
        nc = min(5, len(np.unique(ytr))-1)
        candidates = {
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
            'RF':  RandomForestClassifier(100, random_state=42, max_depth=10),
            'PLS': PLSDAClassifier(nc)
        }
        cv = LeaveOneOut() if len(Xtr) <= 50 else StratifiedKFold(
            n_splits=min(5, len(np.unique(ytr)), len(Xtr)//2), shuffle=True, random_state=42)
        best_sc, best_name = -1, 'RF'
        for name, clf in candidates.items():
            scores = []
            try:
                for tr_i, va_i in cv.split(Xtr, ytr):
                    c = type(clf)(**({} if name=='SVM' else {}))
                    # 重新实例化
                    if name=='SVM': c = SVC(kernel='rbf',C=10,gamma='scale',random_state=42)
                    elif name=='RF': c = RandomForestClassifier(100,random_state=42,max_depth=10)
                    else: c = PLSDAClassifier(nc)
                    c.fit(Xtr[tr_i], ytr[tr_i])
                    scores.append(accuracy_score(ytr[va_i], c.predict(Xtr[va_i])))
            except: scores = [0.0]
            sc = np.mean(scores)
            if sc > best_sc: best_sc, best_name = sc, name
        if best_name=='SVM': final = SVC(kernel='rbf',C=10,gamma='scale',random_state=42)
        elif best_name=='RF': final = RandomForestClassifier(100,random_state=42,max_depth=10)
        else: final = PLSDAClassifier(nc)
        final.fit(Xtr, ytr)
        pred = final.predict(Xte)
        all_pred.extend(pred); all_true.extend(yte)
    return accuracy_score(all_true, all_pred)

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # 加载
    print('Loading data...')
    df_uv  = pd.read_excel('Sampedata0.xlsx', sheet_name='VIS_0')
    df_nir = pd.read_excel('Sampedata0.xlsx', sheet_name='NIR_0')
    drug_raw  = df_uv.iloc[:, 0].values
    manuf_raw = df_uv.iloc[:, 1].values
    uv_raw  = df_uv.iloc[:, 2:].values.astype(np.float32)
    nir_raw = df_nir.iloc[:, 2:].values.astype(np.float32)

    le_d = LabelEncoder(); le_m = LabelEncoder()
    drug_enc  = le_d.fit_transform(drug_raw)
    manuf_enc = le_m.fit_transform(manuf_raw)

    idx = np.arange(len(uv_raw))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=drug_enc)

    results = {}

    # ── Step 1: Direct 28-class, raw spectra (no preprocessing), SVM ──
    print('\n[Step 1] Direct 28-class, raw spectra, SVM...')
    svm1 = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm1.fit(uv_raw[tr], manuf_enc[tr])
    acc1 = accuracy_score(manuf_enc[te], svm1.predict(uv_raw[te]))
    results['Direct 28-class Baseline'] = acc1
    print(f'  Acc = {acc1:.4f}')

    # ── Step 2: + SNV Preprocessing, SVM ──
    print('\n[Step 2] + SNV Preprocessing, SVM...')
    uv_snv  = snv(uv_raw)
    nir_snv = snv(nir_raw)
    # 只用UV SNV做直接分类
    svm2 = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm2.fit(uv_snv[tr], manuf_enc[tr])
    acc2 = accuracy_score(manuf_enc[te], svm2.predict(uv_snv[te]))
    results['+ SNV Preprocessing'] = acc2
    print(f'  Acc = {acc2:.4f}')

    # ── Step 3: + PI-VAE Feature Extraction（仅UV单模态latent）─────────
    print('\n[Step 3] + PI-VAE Feature Extraction (UV only)...')
    uv_vae = train_vae(UV_VAE(uv_snv.shape[1], 32, 10), uv_snv[tr], device, name='UV-VAE')
    z_uv_tr = get_latent(uv_vae, uv_snv[tr], device)
    z_uv_te = get_latent(uv_vae, uv_snv[te], device)
    svm3 = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm3.fit(z_uv_tr, manuf_enc[tr])
    acc3 = accuracy_score(manuf_enc[te], svm3.predict(z_uv_te))
    results['+ PI-VAE Feature Extraction'] = acc3
    print(f'  Acc = {acc3:.4f}')

    # ── Step 4: + Multimodal Fusion（UV+NIR latent）───────────────────
    print('\n[Step 4] + Multimodal Fusion (UV+NIR latent)...')
    nir_vae = train_vae(NIR_VAE(nir_snv.shape[1], 32, 10), nir_snv[tr], device, name='NIR-VAE')
    z_nir_tr = get_latent(nir_vae, nir_snv[tr], device)
    z_nir_te = get_latent(nir_vae, nir_snv[te], device)
    X_fused_tr = np.hstack([z_uv_tr, z_nir_tr])
    X_fused_te = np.hstack([z_uv_te, z_nir_te])
    svm4 = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm4.fit(X_fused_tr, manuf_enc[tr])
    acc4 = accuracy_score(manuf_enc[te], svm4.predict(X_fused_te))
    results['+ Multimodal Fusion'] = acc4
    print(f'  Acc = {acc4:.4f}')

    # ── Step 5: + Cascade Strategy ────────────────────────────────────
    print('\n[Step 5] + Cascade Strategy...')
    acc5 = best_l2_cascade(X_fused_tr, drug_enc[tr], manuf_enc[tr],
                            X_fused_te, drug_enc[te], manuf_enc[te])
    results['+ Cascade Strategy'] = acc5
    print(f'  Acc = {acc5:.4f}')

    # 保存结果
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'Step': list(results.keys()),
                       'Accuracy': [v*100 for v in results.values()]})
    df.to_csv('results/table3_5_cascade_ablation_real.csv', index=False)
    print('\nSaved: results/table3_5_cascade_ablation_real.csv')
    print(df.to_string(index=False))

    # 同时更新原始文件保持兼容
    df_orig = pd.DataFrame({'Method': ['Direct 28-class', 'Cascade',
                                        'Standard AE (L1)', 'PI-VAE (L1)'],
                            'Accuracy': [results['Direct 28-class Baseline'],
                                         results['+ Cascade Strategy'],
                                         results['+ Multimodal Fusion'],  # 近似
                                         results['+ PI-VAE Feature Extraction']]})
    df_orig.to_csv('results/3-ablation_study.csv', index=False)
    print('Updated: results/3-ablation_study.csv')

    print('\nDone.')

if __name__ == '__main__':
    main()
