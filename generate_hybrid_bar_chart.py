import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Data from user request
metrics = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']
scores = [88.84, 98.72, 98.53, 93.52]

# Set up global style to match the LaTeX/TikZ feel
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(7, 4))

# Bar properties
bar_width = 0.3
x_pos = np.arange(len(metrics))

# Colors to match the provided image (a nice blueish purple)
bar_color = '#6d74ff'

# Plot the bars with zorder=3 so they appear above the grid lines
bars = ax.bar(x_pos, scores, width=bar_width, color=bar_color, zorder=3)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    # Position text slightly above the bar
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f'{yval:.2f}', 
            ha='center', va='bottom', fontsize=9)

# Set axes limits
ax.set_ylim(85, 100) # Based on data minimum ~88 and maximum ~98
ax.set_xlim(-0.5, len(metrics) - 0.5)

# Set axes labels
ax.set_ylabel('Score (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)

# Customize grid (dotted grid lines)
ax.grid(True, linestyle=':', alpha=0.7, zorder=0)

# Make ticks point inward to match the classical look
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)

# Add minor ticks for Y axis and enable grid for them
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
ax.grid(True, which='minor', linestyle=':', alpha=0.3, zorder=0)

# Layout
plt.tight_layout()

# Save the plot
output_path = 'plots/hybrid_model_metrics_bar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart successfully generated and saved to: {output_path}")
