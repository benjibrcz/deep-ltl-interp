#!/usr/bin/env python3
"""Generate combined example figures for the report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def create_combined_safety_figure():
    """Create a 2x2 grid of safety test examples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for i, ax in enumerate(axes.flat):
        img_path = f'paper_safety_results/run_{i+1}.png'
        if Path(img_path).exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

    fig.suptitle('Safety Planning Test: (F green | F yellow) & G !blue\nAgent must choose unblocked goal',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/safety_examples_grid.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/safety_examples_grid.png")

def create_combined_optimality_figure():
    """Create a 2x2 grid of optimality test examples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for i, ax in enumerate(axes.flat):
        img_path = f'paper_optimality_results/run_{i+1}.png'
        if Path(img_path).exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

    fig.suptitle('Optimality Planning Test: F blue THEN F green\nAgent should choose blue closer to green (optimal path)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/optimality_examples_grid.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/optimality_examples_grid.png")

def create_comparison_figure():
    """Create a side-by-side comparison of safety vs optimality."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Safety example
    if Path('paper_safety_results/run_1.png').exists():
        img = mpimg.imread('paper_safety_results/run_1.png')
        axes[0].imshow(img)
        axes[0].set_title('Safety Planning (Local)\n80% Success', fontsize=14, fontweight='bold', color='green')
        axes[0].axis('off')

    # Optimality example
    if Path('paper_optimality_results/run_1.png').exists():
        img = mpimg.imread('paper_optimality_results/run_1.png')
        axes[1].imshow(img)
        axes[1].set_title('Optimality Planning (Global)\n10% Optimal Choice', fontsize=14, fontweight='bold', color='red')
        axes[1].axis('off')

    fig.suptitle('Local vs Global Planning: Key Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/local_vs_global_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/local_vs_global_comparison.png")

if __name__ == '__main__':
    create_combined_safety_figure()
    create_combined_optimality_figure()
    create_comparison_figure()
