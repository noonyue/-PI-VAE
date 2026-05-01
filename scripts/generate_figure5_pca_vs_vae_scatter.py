#!/usr/bin/env python3
"""
Generate Figure 5: PCA vs PI-VAE Latent Space Scatter Comparison (2×2 layout)
Panels: (a) PCA-UV, (b) PCA-NIR, (c) PI-VAE-UV, (d) PI-VAE-NIR
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(42)
torch.manual_seed(42)

# ── Load data ──
print("Loading data...")
df = pd.read_excel('Sampedata0.xlsx', sheet_name='VIS_0', header=None)
X_uv_raw = df.iloc[:, 2:].values
drug_labels = df.iloc[:, 0].values

df_nir = pd.read_excel('Sampedata0.xlsx', sheet_name='NIR_0', header=None)
X_nir_raw = df_nir.iloc[:, 2:].values

# SNV normalization
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

X_uv = snv(X_uv_raw).astype(np.float32)
X_nir = snv(X_nir_raw).astype(np.float32)

# ── PCA (2D projection) ──
print("Computing PCA projections...")
pca_uv = PCA(n_components=2, random_state=42)
pca_nir = PCA(n_components=2, random_state=42)
Z_pca_uv = pca_uv.fit_transform(X_uv)
Z_pca_nir = pca_nir.fit_transform(X_nir)

# ── PI-VAE models ──
class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_peaks=12):
        super().__init__()
        self.n_peaks = n_peaks
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_peaks * 3)
        )
    def forward(self, z):
        params = self.fc(z)
        mu = torch.sigmoid(params[:, :self.n_peaks]) * self.output_dim
        A = torch.relu(params[:, self.n_peaks:2*self.n_peaks])
        sigma = torch.relu(params[:, 2*self.n_peaks:]) + 1.0
        x_grid = torch.linspace(0, self.output_dim - 1, self.output_dim, device=z.device)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        mu_exp = mu.unsqueeze(2)
        A_exp = A.unsqueeze(2)
        sigma_exp = sigma.unsqueeze(2)
        peaks = A_exp * torch.exp(-0.5 * ((x_grid - mu_exp) / sigma_exp) ** 2)
        return peaks.sum(dim=1)

class LorentzianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_peaks=12):
        super().__init__()
        self.n_peaks = n_peaks
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_peaks * 3)
        )
    def forward(self, z):
        params = self.fc(z)
        mu = torch.sigmoid(params[:, :self.n_peaks]) * self.output_dim
        A = torch.relu(params[:, self.n_peaks:2*self.n_peaks])
        gamma = torch.relu(params[:, 2*self.n_peaks:]) + 0.5
        x_grid = torch.linspace(0, self.output_dim - 1, self.output_dim, device=z.device)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        mu_exp = mu.unsqueeze(2)
        A_exp = A.unsqueeze(2)
        gamma_exp = gamma.unsqueeze(2)
        peaks = A_exp * (gamma_exp ** 2) / ((x_grid - mu_exp) ** 2 + gamma_exp ** 2)
        return peaks.sum(dim=1)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, peak_type, n_peaks):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        if peak_type == 'gaussian':
            self.dec = GaussianDecoder(latent_dim, input_dim, n_peaks)
        else:
            self.dec = LorentzianDecoder(latent_dim, input_dim, n_peaks)
    def encode(self, x):
        h = self.enc(x)
        return h[:, :h.size(1)//2], h[:, h.size(1)//2:]
    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        xr = self.dec(z)
        return xr, mu, lv

def train_vae(X, peak_type, epochs=200, latent_dim=32, n_peaks=12):
    model = VAE(X.shape[1], latent_dim, peak_type, n_peaks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X).to(device)
    for ep in range(1, epochs + 1):
        model.train()
        xr, mu, lv = model(Xt)
        recon = nn.functional.mse_loss(xr, Xt, reduction='sum') / len(Xt)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum() / len(Xt)
        loss = recon + kl
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  Epoch {ep}/{epochs} - Loss: {loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        _, mu, _ = model(Xt)
    return mu.cpu().numpy()

print("Training UV-VAE (Gaussian)...")
Z_vae_uv_full = train_vae(X_uv, peak_type='gaussian', epochs=200, latent_dim=32)
print("Training NIR-VAE (Lorentzian)...")
Z_vae_nir_full = train_vae(X_nir, peak_type='lorentzian', epochs=200, latent_dim=32)

# Use first 2 dims for visualization
Z_vae_uv = Z_vae_uv_full[:, :2]
Z_vae_nir = Z_vae_nir_full[:, :2]

# ── Plotting ──
ORANGE = '#E87722'
BLUE = '#1565C0'
drug_names = ['CIM', 'FMD', 'GLD', 'GSR', 'HCT', 'IBU', 'MHE', 'MHL', 'MHR']
unique_drugs = sorted(np.unique(drug_labels))
n_drugs = len(unique_drugs)
cmap = plt.cm.tab10
colors = [cmap(i % 10) for i in range(n_drugs)]  # Handle more than 10 drugs

fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.28)

panels = [
    (gs[0, 0], Z_pca_uv,  '(a)', 'PCA — UV-Vis Subspace'),
    (gs[0, 1], Z_pca_nir, '(b)', 'PCA — NIR Subspace'),
    (gs[1, 0], Z_vae_uv,  '(c)', 'PI-VAE — UV-Vis Latent Space\n(Gaussian prior)'),
    (gs[1, 1], Z_vae_nir, '(d)', 'PI-VAE — NIR Latent Space\n(Lorentzian prior)'),
]

for slot, Z, label, title in panels:
    ax = fig.add_subplot(slot)
    for i, drug in enumerate(unique_drugs):
        mask = (drug_labels == drug)
        drug_label = drug_names[i] if i < len(drug_names) else f'Drug{drug}'
        ax.scatter(Z[mask, 0], Z[mask, 1], c=[colors[i]], s=50, alpha=0.75,
                   edgecolors='white', linewidths=0.5, label=drug_label)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    if slot == gs[1, 1]:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  fontsize=9, framealpha=0.9, title='Drug')

plt.suptitle('Figure 5  Feature Space Comparison: PCA vs PI-VAE',
             fontsize=13, fontweight='bold', y=0.995)
plt.savefig('figures/figure5_pca_vs_vae_scatter.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("[OK] Saved: figures/figure5_pca_vs_vae_scatter.png")
