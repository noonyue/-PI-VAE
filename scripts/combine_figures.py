"""
Combine individual figures into comprehensive composite figures for paper.

Scheme A: 6 comprehensive figures
1. Complete Pipeline Overview (preprocessing + feature evolution + waterfall)
2. Reconstruction Quality (UV + NIR reconstruction)
3. Classification Performance (L1 confusion + L2 boundary + L2 heatmap)
4. Physical Prior Validation (residual + ablation radar)
5. Training & Robustness (training curves + noise stress)
6. Latent Disentanglement (UV + NIR latent perturbation)
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/composite", exist_ok=True)


def load_image(path):
    """Load image and return as numpy array"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping...")
        return None
    img = Image.open(path)
    return np.array(img)


def combine_pipeline_overview():
    """Figure 1: Complete Pipeline Overview"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top row: Preprocessing + Feature Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/preprocessing_effect.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) Preprocessing Effect\n(Raw vs SNV)", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1:])
    img2 = load_image("figures/feature_space_evolution.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) Feature Space Evolution\n(Raw → PCA → PI-VAE)", fontsize=12, fontweight='bold', pad=10)
    
    # Bottom row: PCA vs VAE + Waterfall
    ax3 = fig.add_subplot(gs[1, :2])
    img3 = load_image("figures/pca_vs_vae.png")
    if img3 is not None:
        ax3.imshow(img3)
        ax3.axis('off')
        ax3.set_title("(C) PCA vs PI-VAE Latent Space", fontsize=12, fontweight='bold', pad=10)
    
    ax4 = fig.add_subplot(gs[1, 2])
    img4 = load_image("figures/stepwise_accuracy_waterfall.png")
    if img4 is not None:
        ax4.imshow(img4)
        ax4.axis('off')
        ax4.set_title("(D) Stepwise Accuracy Gain", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 1: Complete Pipeline Overview\n(Data Preprocessing → Feature Extraction → Performance)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig1_pipeline_overview.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig1_pipeline_overview.png")
    plt.close()


def combine_reconstruction_quality():
    """Figure 2: Reconstruction Quality"""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.2, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/uv_reconstruction.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) UV-Vis Reconstruction\n(Gaussian Peak Decoder)", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = load_image("figures/nir_reconstruction.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) NIR Reconstruction\n(Lorentzian Peak Decoder)", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 2: Reconstruction Quality\n(Physical Prior Decoders)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig2_reconstruction_quality.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig2_reconstruction_quality.png")
    plt.close()


def combine_classification_performance():
    """Figure 3: Classification Performance"""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/l1_confusion_matrix.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) L1: Drug Classification\nConfusion Matrix", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = load_image("figures/l2_decision_boundary_zoom.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) L2: Manufacturer Decision\nBoundary (t-SNE)", fontsize=12, fontweight='bold', pad=10)
    
    ax3 = fig.add_subplot(gs[0, 2])
    img3 = load_image("figures/l2_model_performance_heatmap.png")
    if img3 is not None:
        ax3.imshow(img3)
        ax3.axis('off')
        ax3.set_title("(C) L2: Model Selection\n(Per-Drug CV Accuracy)", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 3: Classification Performance\n(Cascade Strategy: L1 Drug → L2 Manufacturer)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig3_classification_performance.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig3_classification_performance.png")
    plt.close()


def combine_physical_prior_validation():
    """Figure 4: Physical Prior Validation"""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/spectral_residual_analysis.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) Residual Analysis\n(Lorentzian vs Gaussian Prior)", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = load_image("figures/ablation_radar.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) Ablation Study\n(Standard AE vs Gaussian VAE vs PI-VAE)", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 4: Physical Prior Validation\n(Quantitative Evidence for Physically-Informed Design)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig4_physical_prior_validation.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig4_physical_prior_validation.png")
    plt.close()


def combine_training_robustness():
    """Figure 5: Training & Robustness"""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/training_loss_curve.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) Training Convergence\n(Reconstruction + KL Loss)", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = load_image("figures/robustness_stress.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) Noise Robustness Test\n(PI-VAE vs Raw+SVM, SNR 50→10 dB)", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 5: Training Stability & Robustness\n(Model Convergence & Noise Resistance)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig5_training_robustness.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig5_training_robustness.png")
    plt.close()


def combine_latent_disentanglement():
    """Figure 6: Latent Disentanglement"""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = load_image("figures/latent_shift_uv.png")
    if img1 is not None:
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title("(A) UV-Vis Latent Perturbation\n(Peak Position/Width Control)", fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = load_image("figures/latent_shift_nir.png")
    if img2 is not None:
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title("(B) NIR Latent Perturbation\n(Physical Interpretation)", fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle("Figure 6: Latent Space Disentanglement\n(Opening the Black Box: Physical Meaning of Latent Variables)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("figures/composite/fig6_latent_disentanglement.png", dpi=300, bbox_inches='tight')
    print("Saved: figures/composite/fig6_latent_disentanglement.png")
    plt.close()


def main():
    ensure_dirs()
    print("Combining figures into comprehensive composite figures (Scheme A)...\n")
    
    combine_pipeline_overview()
    combine_reconstruction_quality()
    combine_classification_performance()
    combine_physical_prior_validation()
    combine_training_robustness()
    combine_latent_disentanglement()
    
    print("\nAll composite figures generated successfully!")
    print("Output directory: figures/composite/")


if __name__ == "__main__":
    main()

