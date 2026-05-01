#!/usr/bin/env python3
"""
Generate Table 3.5: Cascade Classification Ablation Results
Data loaded from: results/table3_5_cascade_ablation_real.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure Chinese font
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# Load real experiment data
df = pd.read_csv('results/table3_5_cascade_ablation_real.csv')
steps = df['Step'].tolist()
accs  = df['Accuracy'].tolist()

baseline = accs[0]
data = []
for step, acc in zip(steps, accs):
    delta = acc - baseline
    delta_str = '—' if step == steps[0] else f'+{delta:.2f} pp'
    data.append([step, f'{acc:.2f}%', delta_str])

headers = ['方法/步骤', 'L2 Accuracy', '相对基线提升']

# Create figure
fig, ax = plt.subplots(figsize=(10, 3.5))
fig.patch.set_facecolor('white')
ax.axis('off')

# Create table
table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                 colWidths=[0.50, 0.25, 0.25])

# Styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Header styling
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#1565C0')
    cell.set_text_props(weight='bold', color='white', fontsize=12)
    cell.set_edgecolor('white')
    cell.set_linewidth(1.5)

# Data row styling
row_colors = ['#F5F5F5', '#FFFFFF'] * (len(data) // 2 + 1)
for i in range(len(data)):
    for j in range(len(headers)):
        cell = table[(i + 1, j)]
        # Highlight last row (final result)
        if i == len(data) - 1:
            cell.set_facecolor('#FFF3E0')
            cell.set_text_props(weight='bold', fontsize=11)
        else:
            cell.set_facecolor(row_colors[i])
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(1.0)
        if j > 0:
            cell.set_text_props(ha='center')

plt.title('表 3.5  级联分类消融实验结果',
          fontsize=13, fontweight='bold', pad=20, loc='center')
plt.tight_layout()
plt.savefig('figures/table3_5_cascade_ablation.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("[OK] Saved: figures/table3_5_cascade_ablation.png")
