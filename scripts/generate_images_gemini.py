#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Gemini Imagen API 生成学术论文配图
代理: http://127.0.0.1:33210
"""

import os
import base64
import json
import requests
from pathlib import Path

API_KEY = "AIzaSyBcTEBoO1httzE2V4gOuXFRqimbzcAEVuc"
PROXY = "http://127.0.0.1:33210"
OUT_DIR = Path("d:/work/class/GEN_MODEL/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.proxies = {"http": PROXY, "https": PROXY}
SESSION.verify = False  # 代理可能用自签证书

IMAGEN_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/imagen-4.0-generate-001:predict"
    f"?key={API_KEY}"
)

PROMPTS = {
    "fig_framework": (
        "Scientific academic diagram, white background, vector-style illustration. "
        "Title: PI-VAE Cascade Classification Framework. "
        "Left section labeled Input: two spectral signal waveforms side by side, "
        "one orange Gaussian-shaped curve labeled UV-Vis Spectra, one blue Lorentzian-shaped "
        "curve labeled NIR Spectra. "
        "Middle section labeled PI-VAE Encoder: two parallel MLP encoder blocks (3-layer "
        "neural network nodes depicted as circles and arrows), each outputting a latent vector z. "
        "Below each encoder, a physics-informed decoder: orange decoder shows Gaussian peak "
        "reconstruction with mu sigma parameters annotated, blue decoder shows Lorentzian peak "
        "reconstruction with Gamma parameter annotated. Between encoder and decoder: a small "
        "beta-VAE loss box with KL divergence symbol. "
        "Right section labeled Cascade Classifier: a feature fusion box merging z_NIR, z_UV and "
        "raw features into a single vector. Below it, two-stage classification tree: L1 node "
        "(SVM, RBF kernel) branching to 9 drug-type leaf nodes, then L2 nodes branching to 28 "
        "manufacturer leaf nodes. Accuracy badges: L1 100% and L2 97.22%. "
        "Style: clean flat design, sans-serif font, orange for UV pathway, blue for NIR pathway, "
        "green for latent space, gray for classification. Arrows show data flow. No shadows."
    ),
    "fig_pipeline": (
        "Academic flowchart diagram, white background, minimal flat design. "
        "Title: Spectral Data Analysis Pipeline. "
        "Step 1 Data Acquisition box: icon of pharmaceutical tablet with two spectral curves "
        "(UV and NIR) emanating from it. Label: Raw Spectra, 9 drug types, 28 manufacturers, 357 samples. "
        "Arrow down to Step 2 Preprocessing box: SNV normalization formula shown as small equation. "
        "Arrow down to Step 3 PI-VAE Training box: split into two parallel columns (orange UV-VAE, "
        "blue NIR-VAE), each showing encoder to latent space to physics decoder cycle. "
        "Arrow down to Step 4 Feature Fusion box: concatenation symbol merging z_UV 32-dim, "
        "z_NIR 32-dim, and raw spectra into one feature vector. "
        "Arrow down to Step 5 Cascade Classification box: L1 SVM block to L2 per-drug model "
        "selection block with LOOCV validation loop shown. "
        "Arrow down to Step 6 Evaluation box: confusion matrix thumbnail, accuracy bar charts, "
        "robustness test curves. "
        "Color: orange and blue for dual modalities, gray for pipeline structure. "
        "Arrows: thick directional. Clean minimalist line icons."
    ),
    "fig_graphical_abstract": (
        "Professional academic graphical abstract, landscape orientation 16:9, white background "
        "with subtle light gray panel separations. "
        "Left panel Problem Statement: simple illustration of a shelf of pharmaceutical drug "
        "bottles with question mark overlay. Small caption: Drug counterfeiting and manufacturer "
        "traceability challenge. "
        "Center panel Method PI-VAE: two spectral waveforms entering a funnel-shaped encoder "
        "(orange UV, blue NIR). Inside funnel: neural network node grid. Output: two compact "
        "latent vectors z_UV and z_NIR floating in an abstract 2D latent space scatter plot "
        "with clusters of colored dots representing 9 drug classes, clearly separated. "
        "Below: physics priors shown as Gaussian and Lorentzian curve icons with formula labels. "
        "Right panel Results: two circular accuracy badges: large badge L1 Drug ID 100% Accuracy "
        "in gold orange, smaller badge L2 Manufacturer ID 97.22% in blue. Below badges: small "
        "confusion matrix heatmap thumbnail with clean diagonal pattern. Caption: 28 Manufacturers "
        "Identified. Bottom strip: miniature comparison bar chart PI-VAE vs SVM vs PCA vs CNN, "
        "PI-VAE bar tallest in blue. "
        "Style: Nature or Science journal graphical abstract style. Clean, modern, no clutter. "
        "Font: Helvetica. Color palette: orange #E87E3E, blue #2E6DB4, white, light gray."
    ),
    "fig2_feature_space": (
        "Two-panel scientific comparison figure, white background, academic journal style. "
        "Panel (a) labeled PCA Feature Space: 2D scatter plot, axes labeled PC1 and PC2. "
        "Data points: 9 distinct clusters one per drug type, colored with a categorical palette "
        "from warm to cool spectrum, but clusters are partially overlapping showing limited "
        "separability. Cluster boundaries drawn as soft ellipses. Legend: 9 drug class labels "
        "CIM FMD GLD GSR HCT IBU MHE MHL MHR. Title above: PCA Projection. Small annotation "
        "arrow pointing to overlap region: Low separability. "
        "Panel (b) labeled PI-VAE Latent Space: same layout but 2D scatter of latent dimensions. "
        "Identical 9 drug classes shown but now clusters are tight, compact, well-separated with "
        "clear inter-cluster gaps. Cluster boundaries as sharp ellipses. Same color palette as (a) "
        "for direct comparison. Title: PI-VAE Latent Space. Small annotation: High separability "
        "silhouette score 0.82. "
        "Layout: side-by-side horizontal panels with shared color legend at bottom. "
        "Figure caption: Fig. 2. Comparison of PCA and PI-VAE feature representations for 9 drug "
        "classes. Style: Nature Communications or Analytical Chemistry journal figure style. "
        "Axes: clean, minimal ticks. No gridlines. Font: 10pt sans-serif. "
        "Panel labels bold (a) (b) in top-left corner."
    ),
}


def generate_image(prompt: str, filename: str) -> None:
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9" if "abstract" in filename or "pipeline" in filename else "4:3",
            "safetyFilterLevel": "block_few",
            "personGeneration": "dont_allow",
        },
    }
    print(f"[INFO] Generating {filename} ...")
    try:
        resp = SESSION.post(
            IMAGEN_URL,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        b64 = data["predictions"][0]["bytesBase64Encoded"]
        out_path = OUT_DIR / filename
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"[OK]   Saved -> {out_path}")
    except requests.HTTPError as e:
        print(f"[ERR]  HTTP {e.response.status_code}: {e.response.text[:500]}")
    except Exception as e:
        print(f"[ERR]  {type(e).__name__}: {e}")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    tasks = [
        (PROMPTS["fig_framework"],        "ai_fig1_framework.png"),
        (PROMPTS["fig_pipeline"],         "ai_fig1_pipeline.png"),
        (PROMPTS["fig_graphical_abstract"],"ai_graphical_abstract.png"),
        (PROMPTS["fig2_feature_space"],   "ai_fig2_feature_space.png"),
    ]

    for prompt, fname in tasks:
        generate_image(prompt, fname)

    print("\n[DONE] All images saved to:", OUT_DIR)
