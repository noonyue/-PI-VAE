"""
Unified plotting style library for PI-VAE figures.
Adapts reference figure style (TabPFN paper) to PI-VAE spectral analysis content.

Color Scheme:
- Orange: Raw data, ground truth, baseline methods
- Blue: PI-VAE model results, predictions
- Other colors: Adapted for spectral data (UV/NIR distinction)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# ==================== Color Theme ====================

# Reference style: Orange for ground truth, Blue for predictions
COLOR_TRUE = '#FF7F0E'  # Orange - for raw/true/ground truth
COLOR_PRED = '#1F77B4'  # Blue - for model predictions/results
COLOR_PRED_LIGHT = '#8FB9D6'  # Light blue - for tuned/alternative results

# PI-VAE specific colors
COLOR_UV = '#1F77B4'  # Blue for UV
COLOR_NIR = '#D62728'  # Red for NIR
COLOR_PI_VAE = '#1F77B4'  # Blue for PI-VAE results
COLOR_BASELINE = '#FF7F0E'  # Orange for baseline methods

# Model comparison colors
COLOR_DEFAULT = '#1F77B4'  # Dark blue for default
COLOR_TUNED = '#8FB9D6'  # Light blue for tuned
COLOR_OTHER = '#2CA02C'  # Green for other methods

# Background and grid
COLOR_GRID = '#E0E0E0'
COLOR_FACE = 'white'

# ==================== Typography ====================

FONT_FAMILY = 'sans-serif'
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS = 11
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10
FONT_SIZE_SUBTITLE = 13
FONT_WEIGHT_BOLD = 'bold'

# ==================== Line Styles ====================

LINE_WIDTH = 2.0
LINE_WIDTH_THIN = 1.5
LINE_WIDTH_THICK = 2.5
LINE_STYLE_DASHED = '--'
LINE_STYLE_DOTTED = ':'
LINE_STYLE_SOLID = '-'
ALPHA_FILL = 0.3
ALPHA_LINE = 0.8

# ==================== Grid and Spacing ====================

GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
TICK_LENGTH = 4
TICK_WIDTH = 1.0

# ==================== Figure Setup ====================

def setup_style():
    """Setup matplotlib style for reference figure appearance."""
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZE_AXIS,
        'axes.labelsize': FONT_SIZE_LABEL,
        'axes.titlesize': FONT_SIZE_SUBTITLE,
        'xtick.labelsize': FONT_SIZE_AXIS,
        'ytick.labelsize': FONT_SIZE_AXIS,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
        'axes.linewidth': 1.0,
        'grid.alpha': GRID_ALPHA,
        'grid.linestyle': GRID_LINESTYLE,
        'figure.facecolor': COLOR_FACE,
        'axes.facecolor': COLOR_FACE,
    })

def create_multi_panel_figure(nrows=1, ncols=1, figsize=(12, 8), 
                              hspace=0.35, wspace=0.35):
    """Create figure with multi-panel layout and panel labels."""
    fig = plt.figure(figsize=figsize, facecolor=COLOR_FACE)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, 
                          hspace=hspace, wspace=wspace)
    return fig, gs

def add_panel_label(ax, label, x_offset=-0.15, y_offset=1.05,
                   fontsize=18, fontweight='bold'):
    """Add panel label like (a), (b), (c) etc."""
    ax.text(x_offset, y_offset, label, transform=ax.transAxes,
           fontsize=fontsize, fontweight=fontweight,
           va='bottom', ha='right')

# ==================== Plotting Utilities ====================

def plot_with_confidence_interval(ax, x, y_mean, y_std=None, y_lower=None, 
                                  y_upper=None, color=COLOR_PRED, 
                                  label=None, linestyle='-', 
                                  alpha_fill=ALPHA_FILL):
    """Plot line with confidence interval shading."""
    ax.plot(x, y_mean, color=color, label=label, 
           linewidth=LINE_WIDTH, linestyle=linestyle, alpha=ALPHA_LINE)
    
    if y_std is not None:
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
    elif y_lower is None or y_upper is None:
        return
    
    ax.fill_between(x, y_lower, y_upper, color=color, 
                    alpha=alpha_fill, linewidth=0)

def add_diagonal_line(ax, xlim=None, ylim=None, color='gray', 
                     linestyle='--', linewidth=1.5, alpha=0.5, 
                     label='Equal performance'):
    """Add diagonal line for equal performance in scatter plots."""
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    
    ax.plot([min_val, max_val], [min_val, max_val], 
           color=color, linestyle=linestyle, linewidth=linewidth,
           alpha=alpha, label=label)

def format_axes(ax, xlabel=None, ylabel=None, title=None, 
               grid=True, xlim=None, ylim=None):
    """Format axes with consistent styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, fontweight=FONT_WEIGHT_BOLD)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight=FONT_WEIGHT_BOLD)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_SUBTITLE, 
                    fontweight=FONT_WEIGHT_BOLD, pad=15)
    
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.tick_params(length=TICK_LENGTH, width=TICK_WIDTH)

# ==================== Bar Chart Utilities ====================

def create_grouped_bar_chart(ax, categories, data_dict, width=0.25,
                            xlabel=None, ylabel=None, title=None):
    """Create grouped bar chart with reference style."""
    x = np.arange(len(categories))
    
    colors = [COLOR_DEFAULT, COLOR_TUNED] if len(data_dict) == 2 else \
             [COLOR_DEFAULT, COLOR_TUNED, COLOR_OTHER][:len(data_dict)]
    
    for i, (label, values) in enumerate(data_dict.items()):
        offset = width * (i - (len(data_dict) - 1) / 2)
        ax.bar(x + offset, values, width, label=label, 
              color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

# ==================== Heatmap Utilities ====================

def get_heatmap_colormap(reverse=False):
    """Get colormap for heatmaps matching reference style."""
    if reverse:
        return plt.cm.Blues_r
    return plt.cm.Oranges if reverse else plt.cm.Blues

# ==================== Statistical Annotations ====================

def add_statistical_annotation(ax, text, x=0.05, y=0.95, 
                              transform=None, fontsize=10):
    """Add statistical annotation (e.g., Wilcoxon P-value)."""
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, transform=transform, fontsize=fontsize,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def add_confidence_interval_text(ax, value, ci_lower, ci_upper, 
                                x, y, format_str='.3f'):
    """Add confidence interval text annotation."""
    text = f"{value:{format_str}} (95% CI: {ci_lower:{format_str}}-{ci_upper:{format_str}})"
    ax.text(x, y, text, fontsize=FONT_SIZE_LEGEND, 
           ha='center', va='bottom')

# ==================== Waterfall Chart ====================

def create_waterfall_chart(ax, values, labels, title=None):
    """Create waterfall chart with reference style."""
    deltas = [values[0]] + [values[i] - values[i-1] 
                            for i in range(1, len(values))]
    
    running = values[0]
    xs = range(len(values))
    bottoms = []
    heights = []
    colors = []

    for i, d in enumerate(deltas):
        if i == 0:
            bottoms.append(0)
            heights.append(d)
            colors.append(COLOR_BASELINE)  # Orange for baseline
            running = d
        else:
            bottoms.append(running if d >= 0 else running + d)
            heights.append(abs(d))
            colors.append(COLOR_PRED if d >= 0 else COLOR_NIR)  # Blue/Red
            running += d
    
    bars = ax.bar(xs, heights, bottom=bottoms, color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for x, b, h, v in zip(xs, bottoms, heights, values):
        ax.text(x, b + h + 0.01, f'{v:.3f}', ha='center', 
               va='bottom', fontsize=FONT_SIZE_LEGEND, fontweight='bold')
    
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    format_axes(ax, ylabel='Accuracy', title=title)
    ax.set_ylim(0, max(values) * 1.1)

# ==================== Initialization ====================

# Call setup when module is imported
setup_style()
