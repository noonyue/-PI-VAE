#!/usr/bin/env python3
"""
Generate Figure 15: Stepwise Accuracy Waterfall Chart
Shows the marginal contribution of each module to the final system performance.
Data loaded from: results/table3_5_cascade_ablation_real.csv (real experiment data)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ── Data: load real stepwise accuracy from experiment CSV ──
_df = pd.read_csv('results/table3_5_cascade_ablation_real.csv')
steps = list(zip(_df['Step'].tolist(), _df['Accuracy'].tolist()))

labels = [s[0] for s in steps]
values = [s[1] for s in steps]
deltas = [0] + [values[i] - values[i-1] for i in range(1, len(values))]

# ── Plotting ──
ORANGE = '#E87722'
BLUE = '#1565C0'
GREEN = '#2E7D32'
GRAY = '#757575'
RED = '#C62828'

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.patch.set_facecolor('white')

x_pos = np.arange(len(labels))
bar_width = 0.6

# Draw baseline bar
ax.bar(0, values[0], width=bar_width, color=GRAY, alpha=0.7, 
       edgecolor='black', linewidth=1.5, label='Baseline')

# Draw incremental bars
for i in range(1, len(values)):
    delta = deltas[i]
    if delta >= 0:
        bottom = values[i-1]
        height = delta
        color = ORANGE if i == len(values)-1 else BLUE
    else:
        bottom = values[i]      # bar top is previous value, draws down
        height = -delta
        color = RED
    ax.bar(i, height, width=bar_width, bottom=bottom,
           color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add delta annotation (signed)
    mid_y = values[i-1] + delta / 2
    sign = '+' if delta >= 0 else ''
    ax.text(i, mid_y, f'{sign}{delta:.1f}pp', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

# Add final value annotations on top of each bar
for i, val in enumerate(values):
    ax.text(i, val + 1.5, f'{val:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')

# Styling
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10, ha='center')
ax.set_ylabel('L2 Manufacturer Classification Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(-0.7, len(labels) - 0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, ls='--')
ax.set_axisbelow(True)

# Add horizontal reference line at final performance (dynamic from data)
final_acc = values[-1]
ax.axhline(final_acc, color=ORANGE, ls=':', lw=2, alpha=0.6, zorder=0)
ax.text(len(labels) - 0.5, final_acc + 1.0, f'Final System: {final_acc:.2f}%',
        fontsize=10, color=ORANGE, fontweight='bold', style='italic')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=GRAY, edgecolor='black', label='Baseline (Direct 28-class)'),
    mpatches.Patch(facecolor=BLUE, edgecolor='black', label='Incremental Modules (+)'),
    mpatches.Patch(facecolor=RED, edgecolor='black', label='Performance Decrease (−)'),
    mpatches.Patch(facecolor=ORANGE, edgecolor='black', label='Cascade Strategy (Final)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)

plt.title('Figure 15  System Performance Decomposition: Stepwise Accuracy Waterfall',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/figure15_waterfall.png', dpi=180, bbox_inches='tight', facecolor='white')
print("[OK] Saved: figures/figure15_waterfall.png")
