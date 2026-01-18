#!/usr/bin/env python3
"""
Generate a summary figure of the world model findings.
Shows local vs global planning results and probing evidence.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def create_summary_figure():
    """Create a 2x2 summary figure."""
    fig = plt.figure(figsize=(14, 12))

    # Create grid with custom spacing
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # =========================================
    # Panel A: Local vs Global Planning Results
    # =========================================
    ax1 = fig.add_subplot(gs[0, 0])

    categories = ['Safety\n(Local)', 'Optimality\n(Global)']
    success_rates = [80, 10]
    colors = ['#4CAF50', '#f44336']

    bars = ax1.bar(categories, success_rates, color=colors, edgecolor='black', linewidth=2)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add value labels
    for bar, val in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Correct Choice Rate (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_title('A. Planning Task Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')

    # Add annotations
    ax1.annotate('Pattern\nRecognition', xy=(0, 80), xytext=(0.3, 65),
                fontsize=10, ha='center', color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    ax1.annotate('Requires\nSimulation', xy=(1, 10), xytext=(0.7, 25),
                fontsize=10, ha='center', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    # =========================================
    # Panel B: Probing Results
    # =========================================
    ax2 = fig.add_subplot(gs[0, 1])

    probe_targets = ['Distance\nto zones', 'Blocking\ndetection', 'Blue→Green\ndistance', 'Total path\nvia blue']
    r2_scores = [0.85, 0.95, 0.41, 0.44]  # Using averages from results
    bar_colors = ['#4CAF50', '#4CAF50', '#f44336', '#f44336']

    bars = ax2.bar(probe_targets, r2_scores, color=bar_colors, edgecolor='black', linewidth=2)

    # Add threshold line
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Weak threshold')

    # Add value labels
    for bar, val in zip(bars, r2_scores):
        label = f'{val:.2f}' if val < 1 else f'{int(val*100)}%'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Probe Performance (R² / Accuracy)', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('B. Linear Probing Results', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')

    # Add annotations
    ax2.text(0.5, 1.02, 'Immediate Features', ha='center', fontsize=10,
            color='darkgreen', fontweight='bold', transform=ax2.transAxes)
    ax2.text(2.5, 0.55, 'Computed Features', ha='center', fontsize=10,
            color='darkred', fontweight='bold')

    # =========================================
    # Panel C: Directional Bias Evidence
    # =========================================
    ax3 = fig.add_subplot(gs[1, 0])

    configs = ['Original\n(Optimal=Lower-left)', 'Swapped\n(Optimal=Upper-right)']
    optimal_rates = [5, 95]

    bars = ax3.bar(configs, optimal_rates, color=['#FF9800', '#FF9800'],
                  edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, optimal_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax3.set_ylabel('"Optimal" Choice Rate (%)', fontsize=12)
    ax3.set_ylim(0, 110)
    ax3.set_title('C. Equidistant Control Test', fontsize=14, fontweight='bold')

    # Add explanation box
    box_text = "Agent goes upper-right\n~95% regardless of\nwhich blue is optimal"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.5, 0.3, box_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center', bbox=props)

    ax3.annotate('Bias, not planning!', xy=(0.5, 50), xytext=(0.5, 75),
                fontsize=11, ha='center', fontweight='bold', color='#D84315',
                arrowprops=dict(arrowstyle='->', color='#D84315', lw=2))

    # =========================================
    # Panel D: Mechanism Diagram
    # =========================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')

    # Title
    ax4.text(5, 9.5, 'D. Mechanism Summary', fontsize=14, fontweight='bold',
            ha='center', va='top')

    # Safety box (green)
    safety_box = patches.FancyBboxPatch((0.5, 5.5), 4, 3.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#E8F5E9', edgecolor='#4CAF50',
                                        linewidth=2)
    ax4.add_patch(safety_box)
    ax4.text(2.5, 8.5, 'Safety Planning', fontsize=12, fontweight='bold',
            ha='center', color='#2E7D32')
    ax4.text(2.5, 7.5, 'Observation', fontsize=10, ha='center')
    ax4.text(2.5, 7.0, '↓', fontsize=14, ha='center')
    ax4.text(2.5, 6.5, '"obstacle pattern"', fontsize=10, ha='center', style='italic')
    ax4.text(2.5, 6.0, '↓', fontsize=14, ha='center')
    ax4.text(2.5, 5.5, 'Turn to other goal', fontsize=10, ha='center')

    # Optimality box (red)
    opt_box = patches.FancyBboxPatch((5.5, 5.5), 4, 3.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FFEBEE', edgecolor='#f44336',
                                     linewidth=2)
    ax4.add_patch(opt_box)
    ax4.text(7.5, 8.5, 'Optimality Planning', fontsize=12, fontweight='bold',
            ha='center', color='#C62828')
    ax4.text(7.5, 7.5, 'Would need:', fontsize=10, ha='center')
    ax4.text(7.5, 6.8, 'Simulate future state', fontsize=9, ha='center', style='italic')
    ax4.text(7.5, 6.3, 'Compute new distance', fontsize=9, ha='center', style='italic')
    ax4.text(7.5, 5.8, 'Compare options', fontsize=9, ha='center', style='italic')

    # Checkmark and X
    ax4.text(4.2, 6.5, '✓', fontsize=24, ha='center', color='#4CAF50', fontweight='bold')
    ax4.text(9.2, 6.5, '✗', fontsize=24, ha='center', color='#f44336', fontweight='bold')

    # Bottom text
    ax4.text(5, 4.5, 'Pattern Recognition', fontsize=11, ha='center',
            fontweight='bold', color='#4CAF50')
    ax4.text(5, 3.8, 'vs', fontsize=10, ha='center', color='gray')
    ax4.text(5, 3.1, 'Future Simulation', fontsize=11, ha='center',
            fontweight='bold', color='#f44336')

    # Conclusion box
    conclusion_box = patches.FancyBboxPatch((1, 0.5), 8, 2,
                                            boxstyle="round,pad=0.1",
                                            facecolor='#FFF3E0', edgecolor='#FF9800',
                                            linewidth=2)
    ax4.add_patch(conclusion_box)
    ax4.text(5, 2.0, 'Conclusion: Behavioral Heuristics, Not World Model',
            fontsize=11, fontweight='bold', ha='center', color='#E65100')
    ax4.text(5, 1.2, 'Agent learns reactive patterns, fails when\ngenuine multi-step reasoning is required',
            fontsize=10, ha='center', color='#BF360C')

    # Main title
    fig.suptitle('World Model vs Behavioral Heuristics in DeepLTL Agent',
                fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path = Path('world_model_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

    return output_path


if __name__ == '__main__':
    create_summary_figure()
