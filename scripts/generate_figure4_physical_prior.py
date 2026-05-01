"""
Figure 4: Physical Prior Validation
Combined: UV reconstruction + NIR reconstruction + Residual comparison

Layout: 3 rows x 2 columns
- Row 1: (a) UV Original vs Reconstructed  |  (b) UV Residual
- Row 2: (c) NIR Original vs Reconstructed |  (d) NIR Residual
- Row 3: (e) Lorentzian vs Gaussian residual curves  |  (f) RMSE boxplot

Output: figures/figure4_physical_prior_validation.png
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi_vae_pipeline import (
    load_data, preprocess_spectra, SpectralDataset,
    UV_VAE, NIR_VAE, train_vae,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import (
    setup_style, add_panel_label, format_axes,
    COLOR_TRUE, COLOR_PRED, COLOR_NIR,
)

setup_style()

COLOR_LORENTZ = '#1F77B4'   # blue
COLOR_GAUSS   = '#D62728'   # red
COLOR_RESID   = '#9467BD'   # purple for residual fill


def reconstruct(model, data, device):
    loader = torch.utils.data.DataLoader(
        SpectralDataset(data), batch_size=64, shuffle=False)
    recons = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            rec, _, _, _ = model(x.to(device))
            recons.append(rec.cpu().numpy())
    return np.vstack(recons)


def rmse_per_sample(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def draw_recon_pair(ax_recon, ax_resid,
                    original, reconstructed,
                    wav_label, sample_label,
                    color_recon, panel_a, panel_b):
    """Draw one reconstruction + residual panel pair."""
    n = len(original)
    x = np.arange(n)

    # (left) original vs reconstructed
    ax_recon.plot(x, original,      color=COLOR_TRUE,   lw=1.8,
                  alpha=0.9, label='Original')
    ax_recon.plot(x, reconstructed, color=color_recon,  lw=1.8,
                  alpha=0.85, linestyle='--', label='Reconstructed')
    ax_recon.fill_between(x, original, reconstructed,
                          color=color_recon, alpha=0.08)
    ax_recon.set_title(sample_label, fontsize=10, fontweight='bold', pad=6)
    ax_recon.set_xlabel(f'Wavelength Index ({wav_label})', fontsize=9)
    ax_recon.set_ylabel('Intensity (SNV)', fontsize=9)
    ax_recon.legend(fontsize=8.5, loc='upper right')
    ax_recon.grid(True, alpha=0.25, linestyle='--')
    ax_recon.tick_params(length=3)
    add_panel_label(ax_recon, panel_a, x_offset=-0.13, y_offset=1.04)

    # (right) residual
    residual = original - reconstructed
    ax_resid.plot(x, residual, color=COLOR_RESID, lw=1.5, alpha=0.85)
    ax_resid.axhline(0, color='gray', lw=0.9, linestyle='--')
    ax_resid.fill_between(x, residual, 0,
                          where=(residual >= 0),
                          color=COLOR_RESID, alpha=0.12)
    ax_resid.fill_between(x, residual, 0,
                          where=(residual < 0),
                          color=COLOR_RESID, alpha=0.12)

    rmse_val = np.sqrt(np.mean(residual ** 2))
    ax_resid.set_title(f'Residual  (RMSE = {rmse_val:.4f})',
                       fontsize=10, fontweight='bold', pad=6)
    ax_resid.set_xlabel(f'Wavelength Index ({wav_label})', fontsize=9)
    ax_resid.set_ylabel('Residual', fontsize=9)
    ax_resid.grid(True, alpha=0.25, linestyle='--')
    ax_resid.tick_params(length=3)
    add_panel_label(ax_resid, panel_b, x_offset=-0.13, y_offset=1.04)


def main():
    os.makedirs('figures', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load & preprocess ─────────────────────────────────────────────────
    uv_raw, nir_raw, drug_labels, _ = load_data()
    uv  = preprocess_spectra(uv_raw,  method='snv')
    nir = preprocess_spectra(nir_raw, method='snv')

    # ── Train / test split (80/20, stratified) ────────────────────────────
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(uv))
    idx_tr, idx_te = train_test_split(idx_all, test_size=0.20,
                                      random_state=42,
                                      stratify=drug_labels)

    uv_tr,  uv_te  = uv[idx_tr],  uv[idx_te]
    nir_tr, nir_te = nir[idx_tr], nir[idx_te]
    drug_labels_te = drug_labels[idx_te]

    # ── Train VAE models on TRAIN set ─────────────────────────────────────
    uv_loader  = torch.utils.data.DataLoader(
        SpectralDataset(uv_tr),  batch_size=32, shuffle=True)
    nir_loader = torch.utils.data.DataLoader(
        SpectralDataset(nir_tr), batch_size=32, shuffle=True)

    print('Training UV-VAE (Gaussian decoder)...')
    uv_vae = UV_VAE(input_dim=uv_tr.shape[1], latent_dim=32, n_peaks=12)
    uv_vae, _ = train_vae(uv_vae, uv_loader, epochs=300,
                           device=device, model_name='UV-VAE')

    print('Training NIR-VAE (Lorentzian decoder)...')
    nir_vae = NIR_VAE(input_dim=nir_tr.shape[1], latent_dim=32, n_peaks=12)
    nir_vae, _ = train_vae(nir_vae, nir_loader, epochs=300,
                            device=device, model_name='NIR-VAE')

    print('Training NIR-Gaussian ablation (Gaussian decoder on NIR)...')
    nir_gauss_vae = UV_VAE(input_dim=nir_tr.shape[1], latent_dim=32, n_peaks=12)
    nir_gauss_vae, _ = train_vae(nir_gauss_vae, nir_loader, epochs=300,
                                  device=device, model_name='NIR-Gaussian')

    # ── Evaluate on TEST set (generalization RMSE) ────────────────────────
    rec_uv_te        = reconstruct(uv_vae,        uv_te,  device)
    rec_nir_lor_te   = reconstruct(nir_vae,       nir_te, device)
    rec_nir_gauss_te = reconstruct(nir_gauss_vae, nir_te, device)

    rmse_lor   = rmse_per_sample(nir_te, rec_nir_lor_te)
    rmse_gauss = rmse_per_sample(nir_te, rec_nir_gauss_te)
    rmse_uv_te = rmse_per_sample(uv_te,  rec_uv_te)

    # ── Compute residual autocorrelations (lag 1) for all test samples ────
    def lag1_autocorr(residual_matrix):
        """Compute lag-1 autocorrelation per sample row."""
        acorrs = []
        for row in residual_matrix:
            r = row - row.mean()
            if r.std() < 1e-10:
                acorrs.append(0.0)
                continue
            acorrs.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        return np.array(acorrs)

    resid_matrix_lor   = nir_te - rec_nir_lor_te
    resid_matrix_gauss = nir_te - rec_nir_gauss_te

    acorr_lor   = lag1_autocorr(resid_matrix_lor)
    acorr_gauss = lag1_autocorr(resid_matrix_gauss)

    print(f'Lorentzian residual autocorr: {acorr_lor.mean():.4f} ± {acorr_lor.std():.4f}')
    print(f'Gaussian   residual autocorr: {acorr_gauss.mean():.4f} ± {acorr_gauss.std():.4f}')

    # Select representative test samples (median RMSE)
    uv_idx  = np.argsort(rmse_uv_te)[len(uv_te) // 2]
    nir_idx = np.argsort(rmse_lor)[len(nir_te) // 2]

    uv_orig  = uv_te[uv_idx]
    uv_recon = rec_uv_te[uv_idx]
    nir_orig  = nir_te[nir_idx]
    nir_recon = rec_nir_lor_te[nir_idx]

    drug_uv  = drug_labels_te[uv_idx]
    drug_nir = drug_labels_te[nir_idx]

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.text(0.50, 0.995,
             'Figure 4.  Physical Prior Validation: '
             'Spectral Reconstruction & Residual Analysis',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.94, bottom=0.06,
                           left=0.07, right=0.97,
                           hspace=0.52, wspace=0.32)

    # ── Row 1: UV reconstruction ──────────────────────────────────────────
    ax_uv_recon = fig.add_subplot(gs[0, 0])
    ax_uv_resid = fig.add_subplot(gs[0, 1])
    draw_recon_pair(
        ax_uv_recon, ax_uv_resid,
        uv_orig, uv_recon,
        wav_label='200–800 nm',
        sample_label=f'UV-Vis Reconstruction  [Sample #{uv_idx}, Drug: {drug_uv}]',
        color_recon=COLOR_PRED,
        panel_a='(a)', panel_b='(b)'
    )

    # ── Row 2: NIR reconstruction ─────────────────────────────────────────
    ax_nir_recon = fig.add_subplot(gs[1, 0])
    ax_nir_resid = fig.add_subplot(gs[1, 1])
    draw_recon_pair(
        ax_nir_recon, ax_nir_resid,
        nir_orig, nir_recon,
        wav_label='780–2500 nm',
        sample_label=f'NIR Reconstruction  [Sample #{nir_idx}, Drug: {drug_nir}]',
        color_recon=COLOR_NIR,
        panel_a='(c)', panel_b='(d)'
    )

    # ── Row 3: Lorentzian vs Gaussian comparison ──────────────────────────
    ax_curve  = fig.add_subplot(gs[2, 0])
    ax_box    = fig.add_subplot(gs[2, 1])

    # (e) Residual curves — test set sample
    resid_lor   = nir_orig - rec_nir_lor_te[nir_idx]
    resid_gauss = nir_orig - rec_nir_gauss_te[nir_idx]
    x_nir = np.arange(nir.shape[1])

    ax_curve.plot(x_nir, resid_lor,   color=COLOR_LORENTZ, lw=1.8,
                  alpha=0.85, label='Lorentzian (NIR-VAE)')
    ax_curve.plot(x_nir, resid_gauss, color=COLOR_GAUSS,   lw=1.8,
                  alpha=0.80, linestyle='--', label='Gaussian (ablation)')
    ax_curve.axhline(0, color='gray', lw=0.9, linestyle='--')
    ax_curve.set_title('NIR Residual: Lorentzian vs Gaussian Prior',
                       fontsize=10, fontweight='bold', pad=6)
    ax_curve.set_xlabel('Wavelength Index (NIR)', fontsize=9)
    ax_curve.set_ylabel('Residual', fontsize=9)
    ax_curve.legend(fontsize=9)
    ax_curve.grid(True, alpha=0.25, linestyle='--')
    ax_curve.tick_params(length=3)
    add_panel_label(ax_curve, '(e)', x_offset=-0.13, y_offset=1.04)

    # (f) RMSE boxplot — objective comparison (test set)
    bp = ax_box.boxplot(
        [rmse_lor, rmse_gauss],
        labels=['Lorentzian\n(NIR-VAE)', 'Gaussian\n(ablation)'],
        patch_artist=True, widths=0.5,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )
    bp['boxes'][0].set_facecolor(COLOR_LORENTZ)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_GAUSS)
    bp['boxes'][1].set_alpha(0.7)

    # Mean annotations
    for i, rmse_arr in enumerate([rmse_lor, rmse_gauss], 1):
        ax_box.text(i, rmse_arr.max() + 0.005,
                    f'Mean={rmse_arr.mean():.3f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Physical interpretability note
    ax_box.text(0.50, 0.04,
                'RMSE difference is marginal (small-sample limit).\n'
                'Lorentzian prior is adopted for physical interpretability:\n'
                'peak parameters correspond to real NIR vibrational modes.',
                transform=ax_box.transAxes, ha='center', va='bottom',
                fontsize=7.5, color='#555555', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
                          edgecolor='#CCCCCC', alpha=0.88))

    ax_box.set_title('Test-Set RMSE Distribution\n(Lorentzian vs Gaussian decoder)',
                     fontsize=10, fontweight='bold', pad=6)
    ax_box.set_ylabel('RMSE (test set)', fontsize=9)
    ax_box.grid(axis='y', alpha=0.3, linestyle='--')
    ax_box.tick_params(length=3)
    add_panel_label(ax_box, '(f)', x_offset=-0.13, y_offset=1.04)

    # Save result CSV
    stats = pd.DataFrame({
        'Model':        ['Lorentzian', 'Gaussian'],
        'RMSE_mean':    [rmse_lor.mean(),    rmse_gauss.mean()],
        'RMSE_median':  [np.median(rmse_lor), np.median(rmse_gauss)],
        'RMSE_std':     [rmse_lor.std(),     rmse_gauss.std()],
    })
    stats.to_csv('results/6-prior_fitting_stats.csv', index=False)
    print('Saved: results/6-prior_fitting_stats.csv')

    out_path = 'figures/figure4_physical_prior_validation.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
