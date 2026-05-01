"""
Explainability Module for Upgraded PI-VAE System
可解释性模块

This module provides tools for interpreting and explaining the upgraded PI-VAE model:
- SHAP analysis for feature importance
- Attention weight visualization
- Peak parameter interpretation
"""

from .shap_analyzer import SHAPAnalyzer
from .attention_visualizer import AttentionVisualizer
from .peak_interpreter import PeakInterpreter

__all__ = [
    'SHAPAnalyzer',
    'AttentionVisualizer',
    'PeakInterpreter'
]
