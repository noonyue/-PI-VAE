"""Figure 10: UV-VAE and NIR-VAE training loss curves (3-component decomposition)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

# ── data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
df_uv  = pd.read_excel('Sampedata0.xlsx', sheet_name='VIS_0', header=None)
df_nir = pd.read_excel('Sampedata0.xlsx', sheet_name='NIR_0', header=None)
drug_labels = df_uv.iloc[:, 0].values
X_uv  = df_uv.iloc[:, 2:].values.astype(np.float32)
X_nir = df_nir.iloc[:, 2:].values.astype(np.float32)

def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd

X_uv  = snv(X_uv)
X_nir = snv(X_nir)

device = 'cpu'

# ── VAE architecture ──────────────────────────────────────────────────────────
class PhysicsDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim, peak_type='gaussian', n_peaks=12):
        super().__init__()
        self.out_dim    = out_dim
        self.peak_type  = peak_type
        self.n_peaks    = n_peaks
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, n_peaks * 3)
        )
        x = torch.linspace(0, 1, out_dim)
        self.register_buffer('x', x)

    def forward(self, z):
        params = self.fc(z)
        params = params.view(-1, self.n_peaks, 3)
        pos  = torch.sigmoid(params[..., 0])
        hgt  = torch.sigmoid(params[..., 1])
        wid  = torch.sigmoid(params[..., 2]) * 0.15 + 0.01
        x    = self.x.unsqueeze(0).unsqueeze(0)
        pos  = pos.unsqueeze(-1)
        hgt  = hgt.unsqueeze(-1)
        wid  = wid.unsqueeze(-1)
        if self.peak_type == 'gaussian':
            peaks = hgt * torch.exp(-0.5 * ((x - pos) / wid) ** 2)
        else:
            peaks = hgt / (1 + ((x - pos) / wid) ** 2)
        return peaks.sum(dim=1)

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, peak_type, n_peaks):
        super().__init__()
        self.enc_fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),   nn.ReLU(),
        )
        self.mu_fc  = nn.Linear(128, latent_dim)
        self.lv_fc  = nn.Linear(128, latent_dim)
        self.dec    = PhysicsDecoder(latent_dim, in_dim, peak_type, n_peaks)

    def encode(self, x):
        h  = self.enc_fc(x)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, lv = self.encode(x)
        z  = self.reparameterize(mu, lv)
        xr = self.dec(z)
        return xr, mu, lv

def train_vae(X, peak_type, epochs=300, latent_dim=32, n_peaks=12, log_every=10):
    model = VAE(X.shape[1], latent_dim, peak_type, n_peaks).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt    = torch.tensor(X).to(device)
    history = {'total': [], 'recon': [], 'kl': [], 'epoch': []}
    for ep in range(1, epochs + 1):
        model.train()
        xr, mu, lv = model(Xt)
        recon = nn.functional.mse_loss(xr, Xt, reduction='sum') / len(Xt)
        kl    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum() / len(Xt)
        loss  = recon + kl
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % log_every == 0 or ep == 1:
            history['epoch'].append(ep)
            history['total'].append(loss.item())
            history['recon'].append(recon.item())
            history['kl'].append(kl.item())
    return history

print("Training UV-VAE (Gaussian)...")
hist_uv  = train_vae(X_uv,  peak_type='gaussian',   epochs=300)
print("Training NIR-VAE (Lorentzian)...")
hist_nir = train_vae(X_nir, peak_type='lorentzian', epochs=300)

# ── plotting ──────────────────────────────────────────────────────────────────
ORANGE = '#E87722'
BLUE   = '#1565C0'
GREEN  = '#2E7D32'
GRAY   = '#757575'

fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor('white')
gs  = GridSpec(1, 2, figure=fig, wspace=0.32)

panel_cfg = [
    (gs[0, 0], hist_uv,  'UV-VAE (Gaussian decoder)',       ORANGE),
    (gs[0, 1], hist_nir, 'NIR-VAE (Lorentzian decoder)',    BLUE),
]

for idx, (slot, hist, title, clr) in enumerate(panel_cfg):
    ax = fig.add_subplot(slot)
    ep = np.array(hist['epoch'])
    ax.plot(ep, hist['total'], color=clr,     lw=2.4,  label='Total Loss',     zorder=3)
    ax.plot(ep, hist['recon'], color=clr,     lw=1.8,  ls='--', alpha=0.75,
            label='Recon Loss',    zorder=2)
    ax.plot(ep, hist['kl'],    color='#C62828', lw=2.0, ls='-.',
            label='KL Divergence', zorder=4)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (per sample, log scale)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10, which='both')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, 305)
    # vertical guide: phase boundary at epoch 100
    ax.axvline(100, color=GRAY, lw=1.0, ls=':', alpha=0.6)
    ax.text(102, ax.get_ylim()[0] * 2 if idx == 0 else 1,
            'Phase II', fontsize=8, color=GRAY, va='bottom')
    label = '(a)' if idx == 0 else '(b)'
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

plt.suptitle('Figure 10  Training Dynamics of PI-VAE',
             fontsize=13, fontweight='bold', y=1.02)
plt.savefig('figures/figure10_training_curves.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("[OK] Saved: figures/figure10_training_curves.png")
