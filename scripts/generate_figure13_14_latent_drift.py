"""Figure 13 & 14: Latent space drift under SNR noise injection (UV and NIR)."""
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

np.random.seed(42)
torch.manual_seed(42)

# ── data ──────────────────────────────────────────────────────────────────────
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

# ── model ─────────────────────────────────────────────────────────────────────
class PhysicsDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim, peak_type='gaussian', n_peaks=12):
        super().__init__()
        self.out_dim   = out_dim
        self.peak_type = peak_type
        self.n_peaks   = n_peaks
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                nn.Linear(128, n_peaks * 3))
        self.register_buffer('x', torch.linspace(0, 1, out_dim))

    def forward(self, z):
        params = self.fc(z).view(-1, self.n_peaks, 3)
        pos = torch.sigmoid(params[..., 0]).unsqueeze(-1)
        hgt = torch.sigmoid(params[..., 1]).unsqueeze(-1)
        wid = (torch.sigmoid(params[..., 2]) * 0.15 + 0.01).unsqueeze(-1)
        x   = self.x.unsqueeze(0).unsqueeze(0)
        if self.peak_type == 'gaussian':
            peaks = hgt * torch.exp(-0.5 * ((x - pos) / wid) ** 2)
        else:
            peaks = hgt / (1 + ((x - pos) / wid) ** 2)
        return peaks.sum(dim=1)

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, peak_type, n_peaks=12):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU())
        self.mu_fc = nn.Linear(128, latent_dim)
        self.lv_fc = nn.Linear(128, latent_dim)
        self.dec   = PhysicsDecoder(latent_dim, in_dim, peak_type, n_peaks)

    def encode(self, x):
        h = self.enc(x)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.dec(self.reparameterize(mu, lv)), mu, lv

def train_vae(X, peak_type, epochs=300, latent_dim=32):
    model = VAE(X.shape[1], latent_dim, peak_type).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt    = torch.tensor(X).to(device)
    for ep in range(1, epochs + 1):
        model.train()
        xr, mu, lv = model(Xt)
        recon = nn.functional.mse_loss(xr, Xt, reduction='sum') / len(Xt)
        kl    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum() / len(Xt)
        (recon + kl).backward()
        opt.step(); opt.zero_grad()
    return model

print("Training UV-VAE...")
vae_uv  = train_vae(X_uv,  'gaussian')
print("Training NIR-VAE...")
vae_nir = train_vae(X_nir, 'lorentzian')

def add_snr_noise(X, snr_db):
    sig_pow  = np.mean(X ** 2, axis=1, keepdims=True)
    noi_pow  = sig_pow / (10 ** (snr_db / 10))
    noise    = np.random.randn(*X.shape).astype(np.float32) * np.sqrt(noi_pow)
    return X + noise

SNR_LIST = [50, 40, 30, 20, 10]

def compute_drift(model, X_clean, snr_list):
    model.eval()
    with torch.no_grad():
        Xc  = torch.tensor(X_clean)
        mu0, _ = model.encode(Xc)
        mu0     = mu0.numpy()
    drifts = []
    for snr in snr_list:
        Xn = add_snr_noise(X_clean, snr)
        with torch.no_grad():
            mun, _ = model.encode(torch.tensor(Xn))
            mun = mun.numpy()
        per_sample_drift = np.linalg.norm(mun - mu0, axis=1) / (np.std(mu0, axis=0).mean() + 1e-8)
        drifts.append(per_sample_drift)
    return drifts   # list of (N,) arrays

print("Computing latent drift...")
drifts_uv  = compute_drift(vae_uv,  X_uv,  SNR_LIST)
drifts_nir = compute_drift(vae_nir, X_nir, SNR_LIST)

# ── figure factory ─────────────────────────────────────────────────────────────
ORANGE = '#E87722'
BLUE   = '#1565C0'
COLORS = ['#1565C0', '#388E3C', '#F57F17', '#AD1457', '#4527A0']

def make_fig(drifts, snr_list, modality, color, fig_num, fname):
    means  = [d.mean()  for d in drifts]
    stds   = [d.std()   for d in drifts]
    p25    = [np.percentile(d, 25) for d in drifts]
    p75    = [np.percentile(d, 75) for d in drifts]

    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor('white')
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # (a) mean ± std line chart
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(snr_list))
    ax1.plot(x, means, '-o', color=color, lw=2.2, ms=7, zorder=3)
    ax1.fill_between(x,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color=color, alpha=0.18, label='Mean ± 1 SD')
    ax1.axhline(0.5, color='#B71C1C', ls='--', lw=1.2, label='0.5 SD threshold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s} dB' for s in snr_list], fontsize=10)
    ax1.set_xlabel('Input SNR', fontsize=12)
    ax1.set_ylabel('Latent Drift (std units)', fontsize=12)
    ax1.set_title(f'(a)  Mean Latent Drift\n{modality}', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(labelsize=10)
    ax1.text(-0.13, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=13, fontweight='bold', va='top')

    # (b) boxplot per SNR
    ax2 = fig.add_subplot(gs[0, 1])
    bp  = ax2.boxplot(drifts, labels=[f'{s}' for s in snr_list],
                      patch_artist=True, widths=0.5,
                      medianprops=dict(color='white', lw=2.0))
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax2.axhline(0.5, color='#B71C1C', ls='--', lw=1.2, label='0.5 SD threshold')
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Per-sample Drift (std units)', fontsize=12)
    ax2.set_title(f'(b)  Drift Distribution per SNR\n{modality}', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.tick_params(labelsize=10)
    ax2.text(-0.13, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=13, fontweight='bold', va='top')

    plt.suptitle(f'Figure {fig_num}  {modality} Latent Space Stability Under Noise',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {fname}")

make_fig(drifts_uv,  SNR_LIST, 'UV-Vis VAE (Gaussian)',    ORANGE, 13, 'figures/figure13_uv_latent_drift.png')
make_fig(drifts_nir, SNR_LIST, 'NIR VAE (Lorentzian)',     BLUE,   14, 'figures/figure14_nir_latent_drift.png')
