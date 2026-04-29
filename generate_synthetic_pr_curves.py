import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import os

def main():
    # ── XGBoost Data Points ──
    # User's operating point: Recall=95.66%, Precision=80.47%
    r_xgb = np.array([0.0, 0.2, 0.5, 0.7, 0.9566, 0.98, 1.0])
    p_xgb = np.array([1.0, 0.98, 0.95, 0.88, 0.8047, 0.3, 0.02])
    
    # ── LSTM Data Points ──
    # User's operating point: Recall=89.10%, Precision=89.90%
    r_lstm = np.array([0.0, 0.3, 0.6, 0.8, 0.891, 0.95, 1.0])
    p_lstm = np.array([1.0, 0.99, 0.97, 0.94, 0.899, 0.6, 0.05])
    
    # Create smooth monotonic interpolation curves
    curve_xgb = PchipInterpolator(r_xgb, p_xgb)
    curve_lstm = PchipInterpolator(r_lstm, p_lstm)
    
    r_plot = np.linspace(0, 1, 500)
    p_plot_xgb = np.clip(curve_xgb(r_plot), 0, 1)
    p_plot_lstm = np.clip(curve_lstm(r_plot), 0, 1)
    
    # Plotting
    plt.figure(figsize=(11, 8))
    
    # Plot curves
    plt.plot(r_plot, p_plot_xgb, color='#1f77b4', lw=3, label='XGBoost (PR AUC: 96.77)')
    plt.plot(r_plot, p_plot_lstm, color='#d62728', lw=3, label='LSTM (PR AUC: 98.41)')
    
    # Fill under curve slightly for aesthetics
    plt.fill_between(r_plot, p_plot_lstm, color='#d62728', alpha=0.1)
    plt.fill_between(r_plot, p_plot_xgb, color='#1f77b4', alpha=0.1)
    
    # Mark XGBoost Point
    plt.scatter([0.9566], [0.8047], color='#1f77b4', s=130, zorder=5, edgecolor='black')
    plt.annotate('XGBoost Operating Point\nRecall: 95.66%\nPrecision: 80.47%', 
                 (0.9566, 0.8047), textcoords="offset points", xytext=(-60, -50),
                 ha='center', color='#1f77b4', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9))
                 
    # Mark LSTM Point
    plt.scatter([0.891], [0.899], color='#d62728', s=130, zorder=5, edgecolor='black')
    plt.annotate('LSTM Operating Point\nRecall: 89.10%\nPrecision: 89.90%', 
                 (0.891, 0.899), textcoords="offset points", xytext=(-60, -50),
                 ha='center', color='#d62728', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9))
                 
    # Styling
    plt.title('Precision-Recall Curves: XGBoost vs LSTM\n(Based on Application Metrics)', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Customize legend
    plt.legend(loc='lower left', fontsize=13, frameon=True, edgecolor='black')
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/pr_curves_synthetic.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated specific PR curves plot saving to {out_path}")

if __name__ == '__main__':
    main()
