#!/usr/bin/env python3
"""Generate clean, separate figures for the report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_planning_performance_figure():
    """Bar chart comparing local vs global planning performance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Safety Planning\n(Local)', 'Optimality Planning\n(Global)']
    success_rates = [80, 10]
    colors = ['#4CAF50', '#f44336']

    bars = ax.bar(categories, success_rates, color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance level')

    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel('Correct Choice Rate (%)', fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_title('Planning Task Performance', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    plt.savefig('figures/planning_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/planning_performance.png")


def create_probing_results_figure():
    """Bar chart showing probing results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    probe_targets = ['Distance\nto zones', 'Blocking\ndetection', 'Blue→Green\ndistance', 'Total path\nvia blue']
    scores = [0.85, 0.95, 0.41, 0.44]
    colors = ['#4CAF50', '#4CAF50', '#f44336', '#f44336']

    bars = ax.bar(probe_targets, scores, color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (R²=0.5)')

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Probe Performance (R² / Accuracy)', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_title('Linear Probing Results', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)

    # Add category labels
    ax.text(0.5, -0.18, 'Immediate Features', ha='center', fontsize=12,
            color='#2E7D32', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, -0.23, '(encoded)', ha='center', fontsize=10,
            color='#2E7D32', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('figures/probing_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/probing_results.png")


if __name__ == '__main__':
    create_planning_performance_figure()
    create_probing_results_figure()
