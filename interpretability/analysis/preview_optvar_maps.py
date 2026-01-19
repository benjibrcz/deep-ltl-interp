#!/usr/bin/env python3
"""
Preview optvar map layouts to verify they're correct before running eval.
"""

import os
import sys
sys.path.insert(0, 'src')

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import projections

import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from safety_gymnasium.utils.registration import make

# Import built-in visualization
from visualize.zones import draw_zones, draw_path, draw_diamond, setup_axis, FancyAxes


COLORS = ['blue', 'green', 'yellow', 'magenta']

COLOR_PAIRS = [
    (a, b) for a in COLORS for b in COLORS if a != b
]


def get_unwrapped(env):
    """Get the base environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def create_env(int_color, goal_color, layout_seed):
    """Create environment with specified colors and layout seed."""
    # Custom config must include agent_name since we're overriding the registered config
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    env = make('PointLtl2-v0.optvar', config=config)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    return env


def plot_map(ax, env, int_color, goal_color, layout_seed):
    """Plot the map configuration using built-in visualization."""
    env.reset()
    unwrapped = get_unwrapped(env)

    agent_pos = unwrapped.agent_pos[:2]
    zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

    # Use built-in axis setup
    setup_axis(ax)

    # Find intermediate and goal zones
    int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
    goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]

    # Compute myopic vs optimal
    int_analysis = []
    for name, pos in int_zones:
        d_agent = np.linalg.norm(pos - agent_pos)
        d_goal = min(np.linalg.norm(gpos - pos) for _, gpos in goal_zones)
        int_analysis.append({
            'name': name,
            'pos': pos,
            'd_agent': d_agent,
            'd_goal': d_goal,
            'total': d_agent + d_goal,
        })

    myopic_zone = min(int_analysis, key=lambda x: x['d_agent'])
    optimal_zone = min(int_analysis, key=lambda x: x['total'])

    is_contested = myopic_zone['name'] != optimal_zone['name']

    # Draw zones using built-in function
    draw_zones(ax, zone_positions)

    # Draw agent position (orange diamond)
    draw_diamond(ax, agent_pos, color='orange')

    # Draw optimal path (agent -> optimal -> goal) - green dashed
    goal_pos = goal_zones[0][1]
    optimal_path = [agent_pos.tolist(), optimal_zone['pos'].tolist(), goal_pos.tolist()]
    draw_path(ax, optimal_path, color='#4caf50', linewidth=3, style='dashed')

    # Draw myopic path (agent -> myopic -> goal) - orange dotted
    myopic_path = [agent_pos.tolist(), myopic_zone['pos'].tolist(), goal_pos.tolist()]
    draw_path(ax, myopic_path, color='#ff9800', linewidth=3, style='dotted')

    # Info text
    opt_total = optimal_zone['total']
    myo_total = myopic_zone['total']

    contested_text = "CONTESTED" if is_contested else "NOT CONTESTED"
    contested_color = "#4caf50" if is_contested else "#f44336"

    ax.set_title(f"{int_color}→{goal_color}\n"
                f"Opt: {opt_total:.2f} | Myo: {myo_total:.2f}",
                fontsize=10, color=contested_color, fontweight='bold')

    return is_contested


def main():
    random.seed(42)
    np.random.seed(42)

    print("Generating map previews...")

    # Register FancyAxes projection
    projections.register_projection(FancyAxes)

    # Generate 12 maps with different color pairs and seeds
    n_maps = 12
    cols = 4
    rows = 3

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    contested_count = 0

    for i in range(n_maps):
        int_color, goal_color = random.choice(COLOR_PAIRS)
        layout_seed = 42 + i * 7

        print(f"  Map {i+1}: {int_color}→{goal_color}, seed={layout_seed}")

        ax = fig.add_subplot(rows, cols, i + 1,
                           projection='fancy_box_axes',
                           edgecolor='gray', linewidth=0.5)

        env = create_env(int_color, goal_color, layout_seed)
        is_contested = plot_map(ax, env, int_color, goal_color, layout_seed)
        env.close()

        if is_contested:
            contested_count += 1

    fig.suptitle(f"Optvar Map Preview\n"
                f"Structure: 2 intermediate zones, 1 goal zone, 2 distractors\n"
                f"Green dashed = optimal path | Orange dotted = myopic path",
                fontsize=14, fontweight='bold')

    plt.tight_layout(pad=3)

    output_dir = 'interpretability/results/optimality_analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/map_preview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_file}")
    print(f"Contested maps: {contested_count}/{n_maps}")


if __name__ == '__main__':
    main()
