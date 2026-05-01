# PI-VAE Spectral Analysis System - Submission Package

## Overview

This package contains the complete code and data for reproducing the results presented in our paper on Physics-Informed Variational Autoencoder (PI-VAE) for pharmaceutical drug and manufacturer identification using UV-Vis and NIR spectroscopy.

**Key Features:**
- Physics-informed feature extraction with Gaussian (UV-Vis) and Lorentzian (NIR) priors
- Cascade classification system (L1: drug type, L2: manufacturer)
- Two system versions: original PI-VAE and upgraded system with Transformer encoder

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Environment

```bash
python check_environment.py
```

### 3. Run Analysis

**Option A: Original PI-VAE System (Recommended for reproduction)**
```bash
python pi_vae_pipeline.py
```

**Option B: Upgraded System with Transformer**
```bash
python train_upgraded_system.py
```

## Data Description

**File:** `Sampedata0.xlsx`

This Excel file contains spectral data with two sheets:
- **VIS_0**: UV-Vis spectral data (200-800 nm)
- **NIR_0**: NIR spectral data (1000-2500 nm)

**Data Format:**
- Column 1: Drug type label (9 classes)
- Column 2: Manufacturer label (28 classes)
- Column 3+: Spectral intensity values at each wavelength

**Dataset Statistics:**
- Total samples: 360
- Drug types: 9 (CIM, FMD, GLD, GSR, HCT, IBU, MHE, MHL, MHR)
- Manufacturers: 28 (distributed across drug types)
- Train/test split: 80/20 stratified by drug type

## File Structure

```
submission_package/
├── README_SUBMISSION.md          # This file
├── requirements.txt              # Python dependencies
├── check_environment.py          # Environment verification
├── Sampedata0.xlsx              # Original spectral data
├── pi_vae_pipeline.py           # Main pipeline (original system)
├── train_upgraded_system.py     # Upgraded system training
├── configs/
│   └── upgraded_config.yaml     # Model configuration
├── models/                      # Model architecture
│   ├── upgraded_pi_vae.py
│   ├── transformer_encoder.py
│   ├── physics_loss.py
│   └── contrastive_loss.py
├── utils/                       # Utility modules
│   └── augmentation.py
├── explainability/              # Explainability tools
│   ├── __init__.py
│   ├── peak_interpreter.py
│   ├── attention_visualizer.py
│   └── shap_analyzer.py
└── scripts/                     # Analysis scripts
    ├── plotting_style.py
    ├── output_path_helper.py
    ├── benchmark_l1_l2_models.py
    ├── benchmark_l1_cascade_fused.py
    ├── benchmark_l2_models.py
    └── generate_paper_figures_tables.py
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**:
  - CPU: Any modern processor
  - GPU: NVIDIA GPU (optional, recommended for faster training)
  - RAM: At least 8GB
  - Disk: At least 2GB free space

**Key Dependencies:**
- PyTorch >= 1.8.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.6.0
- openpyxl (for Excel file reading)

## Expected Outputs

Running `pi_vae_pipeline.py` will generate:

**Directories:**
- `figures/`: Visualization outputs (PCA vs VAE, reconstructions, confusion matrices, etc.)
- `results/`: CSV files with classification results and performance metrics

**Key Outputs:**
- L1 drug classification: 100% accuracy
- L2 manufacturer classification: 97.22% accuracy (cascade approach)
- Spectral reconstruction quality metrics
- Ablation study results
- Robustness analysis under noise

## Reproducibility Notes

1. **Random Seeds**: All scripts use fixed random seeds (42) for reproducibility
2. **Data Preprocessing**: SNV (Standard Normal Variate) normalization is applied
3. **Train/Test Split**: 80/20 stratified split by drug labels
4. **Model Selection**: L2 uses LOOCV for small samples, StratifiedKFold otherwise
5. **Device**: Code automatically detects and uses CUDA GPU if available

## Benchmarking Scripts

Additional analysis scripts are provided in `scripts/`:

- `benchmark_l1_l2_models.py`: Compare multiple models for L1 drug classification
- `benchmark_l1_cascade_fused.py`: L1 cascade with fused features
- `benchmark_l2_models.py`: L2 manufacturer classification benchmarks
- `generate_paper_figures_tables.py`: Generate all paper figures and tables

## Citation

If you use this code or data in your research, please cite:

```
[Citation information will be added upon publication]
```

## Contact

For questions or issues, please contact:

- **Author**: [Author name]
- **Email**: [Author email]
- **Institution**: [Institution name]

## License

[License information to be added]

---

**Last Updated**: 2026-05-01
