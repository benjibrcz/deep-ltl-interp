#!/usr/bin/env python3
"""
Preview equidistant optimality map layouts to verify they're correct before running eval.

In these maps, both intermediate zones are equidistant from the agent,
so the only way to choose optimally is to consider the full path to the goal.
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
    """Create equidistant environment with specified colors and layout seed."""
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    env = make('PointLtl2-v0.opteq', config=config)
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

    # Compute distances and totals
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

    optimal_zone = min(int_analysis, key=lambda x: x['total'])
    suboptimal_zone = max(int_analysis, key=lambda x: x['total'])

    # Check equidistant constraint
    d_diff = abs(int_analysis[0]['d_agent'] - int_analysis[1]['d_agent'])
    is_equidistant = d_diff < 0.1

    # Draw zones using built-in function
    draw_zones(ax, zone_positions)

    # Draw agent position (orange diamond)
    draw_diamond(ax, agent_pos, color='orange')

    # Draw optimal path (agent -> optimal -> goal) - green dashed
    goal_pos = goal_zones[0][1]
    optimal_path = [agent_pos.tolist(), optimal_zone['pos'].tolist(), goal_pos.tolist()]
    draw_path(ax, optimal_path, color='#4caf50', linewidth=3, style='dashed')

    # Draw suboptimal path (agent -> suboptimal -> goal) - red dotted
    suboptimal_path = [agent_pos.tolist(), suboptimal_zone['pos'].tolist(), goal_pos.tolist()]
    draw_path(ax, suboptimal_path, color='#f44336', linewidth=3, style='dotted')

    # Draw equidistant circles (both intermediates should be on same circle)
    avg_d_agent = (int_analysis[0]['d_agent'] + int_analysis[1]['d_agent']) / 2
    circle = plt.Circle(agent_pos, avg_d_agent, fill=False, color='gray',
                       linestyle='--', linewidth=1, alpha=0.5)
    ax.add_patch(circle)

    # Info text
    opt_total = optimal_zone['total']
    subopt_total = suboptimal_zone['total']

    eq_text = "EQUIDIST" if is_equidistant else f"NOT EQ (Δ={d_diff:.2f})"
    eq_color = "#4caf50" if is_equidistant else "#f44336"

    ax.set_title(f"{int_color}→{goal_color}\n"
                f"d_agent: {int_analysis[0]['d_agent']:.2f}, {int_analysis[1]['d_agent']:.2f}\n"
                f"Opt: {opt_total:.2f} | Sub: {subopt_total:.2f}",
                fontsize=9, color=eq_color, fontweight='bold')

    return is_equidistant


def main():
    random.seed(42)
    np.random.seed(42)

    print("Generating equidistant map previews...")

    # Register FancyAxes projection
    projections.register_projection(FancyAxes)

    # Generate 12 maps with different color pairs and seeds
    n_maps = 12
    cols = 4
    rows = 3

    fig = plt.figure(figsize=(4 * cols, 4.5 * rows))

    equidistant_count = 0

    for i in range(n_maps):
        int_color, goal_color = random.choice(COLOR_PAIRS)
        layout_seed = 42 + i * 7

        print(f"  Map {i+1}: {int_color}→{goal_color}, seed={layout_seed}")

        ax = fig.add_subplot(rows, cols, i + 1,
                           projection='fancy_box_axes',
                           edgecolor='gray', linewidth=0.5)

        env = create_env(int_color, goal_color, layout_seed)
        is_equidistant = plot_map(ax, env, int_color, goal_color, layout_seed)
        env.close()

        if is_equidistant:
            equidistant_count += 1

    fig.suptitle(f"Equidistant Optimality Map Preview\n"
                f"Both intermediates at SAME distance from agent (gray circle)\n"
                f"Green dashed = optimal path | Red dotted = suboptimal path",
                fontsize=14, fontweight='bold')

    plt.tight_layout(pad=3)

    output_dir = 'interpretability/results/optimality_analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/map_preview_equidistant.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_file}")
    print(f"Equidistant maps: {equidistant_count}/{n_maps}")


if __name__ == '__main__':
    main()
