#!/usr/bin/env python3
"""
Equidistant Optimality Test: Both blue zones same distance from agent

This tests whether the agent has learned any chained distance computation.
When both blues are equidistant from the agent, a myopic policy would choose
randomly (50/50), while a planning policy would prefer the blue closer to green.

Configuration:
- Agent at (0.1, 0) - equidistant from both blues
- Blue1 at (-0.8, -1) - distance to agent: 1.35, distance to green: 1.08 (OPTIMAL)
- Blue2 at (1, 1) - distance to agent: 1.35, distance to green: 3.76 (SUBOPTIMAL)
- Green at (-1.4, -1.9)

If agent chooses randomly: 50% optimal
If agent plans ahead: >50% optimal (prefers blue closer to green)
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

import preprocessing
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from ltl.automata import LDBASequence
from ltl.logic import Assignment

import safety_gymnasium
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

torch.set_grad_enabled(False)

ZONE_RADIUS = 0.4

# Equidistant configuration
# Agent placed on perpendicular bisector of line between Blue1 and Blue2
EQUIDISTANT_CONFIG = {
    'agent_start': np.array([0.1, 0.0]),  # Equidistant from both blues
    'green': np.array([-1.4, -1.9]),
    'blue1': np.array([-0.8, -1]),      # Closer to green (OPTIMAL)
    'blue2': np.array([1, 1]),           # Farther from green (SUBOPTIMAL)
}


class SimpleAgent:
    def __init__(self, model, propositions):
        self.model = model
        self.propositions = propositions

    def get_action(self, obs, deterministic=True):
        if not isinstance(obs, list):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            dist, value = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()


def create_optimality_task(propositions):
    """F blue & F green - reach both."""
    blue = Assignment.single_proposition('blue', propositions).to_frozen()
    reach_blue = frozenset([blue])
    green = Assignment.single_proposition('green', propositions).to_frozen()
    reach_green = frozenset([green])
    no_avoid = frozenset()
    return LDBASequence([(reach_blue, no_avoid), (reach_green, no_avoid)])


def get_env_internals(env):
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def modify_agent_position(env, new_pos):
    """Modify the agent's starting position after reset."""
    builder = get_env_internals(env)
    if hasattr(builder, 'agent_pos'):
        builder.agent_pos[0] = new_pos[0]
        builder.agent_pos[1] = new_pos[1]


def run_equidistant_test(env, agent, equidistant_pos, max_steps=400):
    """Run one episode with agent starting equidistant from both blues."""
    obs, info = env.reset(), {}
    builder = get_env_internals(env)

    # Override agent position to be equidistant
    modify_agent_position(env, equidistant_pos)
    agent_start = equidistant_pos.copy()

    # Get zone positions
    zone_positions = {}
    if hasattr(builder, 'zone_positions'):
        for name, pos in builder.zone_positions.items():
            zone_positions[name] = pos[:2].copy()

    blue_zones = {k: v for k, v in zone_positions.items() if k.startswith('blue')}
    green_pos = zone_positions.get('green_zone0', None)

    if green_pos is None or len(blue_zones) < 2:
        return None

    blue_positions = list(blue_zones.values())

    # Calculate distances
    dist_agent_to_blue = [np.linalg.norm(agent_start - b) for b in blue_positions]
    dist_blue_to_green = [np.linalg.norm(b - green_pos) for b in blue_positions]

    # Optimal is the one closer to green
    optimal_blue_idx = np.argmin(dist_blue_to_green)

    # Run episode
    trajectory = [agent_start.copy()]
    done = False
    first_blue_reached = None
    steps = 0

    while not done and steps < max_steps:
        action = agent.get_action(obs, deterministic=True)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]

        obs, reward, done, info = env.step(action)
        steps += 1

        agent_pos = builder.agent_pos[:2].copy()
        trajectory.append(agent_pos.copy())

        if first_blue_reached is None:
            for i, blue in enumerate(blue_positions):
                if np.linalg.norm(agent_pos - blue) < ZONE_RADIUS:
                    first_blue_reached = i
                    break

    chose_optimal = first_blue_reached == optimal_blue_idx if first_blue_reached is not None else None

    return {
        'agent_start': agent_start,
        'green_pos': green_pos,
        'blue_positions': blue_positions,
        'optimal_blue_idx': optimal_blue_idx,
        'dist_agent_to_blue': dist_agent_to_blue,
        'dist_blue_to_green': dist_blue_to_green,
        'trajectory': trajectory,
        'first_blue_reached': first_blue_reached,
        'chose_optimal': chose_optimal,
        'steps': steps,
    }


def plot_result(result, filename, run_num):
    fig, ax = plt.subplots(figsize=(8, 8))

    colors_map = {'blue': '#2196f3', 'green': '#4caf50'}

    # Plot blue zones
    for i, blue in enumerate(result['blue_positions']):
        is_optimal = (i == result['optimal_blue_idx'])
        edgecolor = 'green' if is_optimal else 'orange'
        linewidth = 3 if is_optimal else 2

        circle = patches.Circle(blue, ZONE_RADIUS,
                               facecolor=colors_map['blue'], alpha=0.7,
                               edgecolor=edgecolor, linewidth=linewidth)
        ax.add_patch(circle)

        label = f"OPTIMAL\n(d={result['dist_blue_to_green'][i]:.2f} to green)" if is_optimal else f"SUBOPTIMAL\n(d={result['dist_blue_to_green'][i]:.2f} to green)"
        ax.annotate(label, blue, textcoords="offset points", xytext=(0, -40),
                   ha='center', fontsize=9, color=edgecolor, fontweight='bold')

    # Plot green zone
    green = result['green_pos']
    circle = patches.Circle(green, ZONE_RADIUS,
                           facecolor=colors_map['green'], alpha=0.7,
                           edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)

    # Plot trajectory
    traj = np.array(result['trajectory'])
    path_color = '#2ecc71' if result['chose_optimal'] else '#e67e22'
    ax.plot(traj[:, 0], traj[:, 1], color=path_color, linewidth=2.5, alpha=0.8)

    # Mark start
    ax.scatter(traj[0, 0], traj[0, 1], s=200, c='orange', marker='D',
              zorder=10, edgecolors='black', linewidths=2)

    # Draw equidistant lines from start to both blues
    for i, blue in enumerate(result['blue_positions']):
        ax.plot([traj[0, 0], blue[0]], [traj[0, 1], blue[1]],
               'k--', alpha=0.3, linewidth=1)
        mid = (traj[0] + blue) / 2
        ax.annotate(f'd={result["dist_agent_to_blue"][i]:.2f}', mid,
                   fontsize=8, alpha=0.7)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    title = f"EQUIDISTANT TEST - Run {run_num}\n"
    title += f"Both blues at distance {result['dist_agent_to_blue'][0]:.2f} from agent\n"

    if result['first_blue_reached'] is not None:
        if result['chose_optimal']:
            title += "CHOSE OPTIMAL (closer to green)"
            color = 'green'
        else:
            title += "CHOSE SUBOPTIMAL (farther from green)"
            color = 'orange'
    else:
        title += "Did not reach blue"
        color = 'gray'

    ax.set_title(title, fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='planning_from_baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--out_dir', default='equidistant_results')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name = 'PointLtl2-v0'

    # Calculate equidistant position
    blue1 = EQUIDISTANT_CONFIG['blue1']
    blue2 = EQUIDISTANT_CONFIG['blue2']
    green = EQUIDISTANT_CONFIG['green']
    equidistant_pos = EQUIDISTANT_CONFIG['agent_start']

    # Verify equidistance
    d1 = np.linalg.norm(equidistant_pos - blue1)
    d2 = np.linalg.norm(equidistant_pos - blue2)
    d1_green = np.linalg.norm(blue1 - green)
    d2_green = np.linalg.norm(blue2 - green)

    print("="*60)
    print("EQUIDISTANT OPTIMALITY TEST")
    print("="*60)
    print(f"\nAgent position: {equidistant_pos}")
    print(f"\nBlue zones:")
    print(f"  Blue1 {blue1}: agent→blue = {d1:.3f}, blue→green = {d1_green:.2f} (OPTIMAL)")
    print(f"  Blue2 {blue2}: agent→blue = {d2:.3f}, blue→green = {d2_green:.2f} (SUBOPTIMAL)")
    print(f"\nDistance difference: |{d1:.3f} - {d2:.3f}| = {abs(d1-d2):.4f}")
    print(f"\nIf myopic (random): ~50% optimal")
    print(f"If planning: >50% optimal")
    print("="*60)

    # Create sampler
    def optimality_sampler(props):
        return lambda: create_optimality_task(props)

    # Create environment (using fixed env as base)
    base_env = safety_gymnasium.make('PointLtl2-v0.fixed')
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    propositions = base_env.get_propositions()
    sample_task = optimality_sampler(propositions)
    env = SequenceWrapper(base_env, sample_task)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    env = RemoveTruncWrapper(env)

    # Load model
    print(f"\nLoading model: {env_name} / {args.exp}")
    config = model_configs[env_name]
    model_store = ModelStore(env_name, args.exp, args.seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    print(f"\nRunning {args.n_runs} equidistant test episodes...")

    results = []
    for run in range(args.n_runs):
        env.reset(seed=args.seed + run)
        result = run_equidistant_test(env, agent, equidistant_pos, args.max_steps)

        if result:
            results.append(result)
            choice = "OPTIMAL" if result['chose_optimal'] else "SUBOPTIMAL" if result['first_blue_reached'] is not None else "NONE"
            print(f"  Run {run+1}: {choice}")
            plot_result(result, out_dir / f'run_{run+1}.png', run+1)

    env.close()

    # Summary
    print("\n" + "="*60)
    print("EQUIDISTANT TEST RESULTS")
    print("="*60)

    n_total = len(results)
    n_optimal = sum(1 for r in results if r['chose_optimal'] == True)
    n_suboptimal = sum(1 for r in results if r['chose_optimal'] == False)
    n_none = sum(1 for r in results if r['first_blue_reached'] is None)

    print(f"\nTotal runs: {n_total}")
    print(f"\nBlue zone choices:")
    print(f"  OPTIMAL (closer to green):    {n_optimal}/{n_total} ({100*n_optimal/n_total:.1f}%)")
    print(f"  SUBOPTIMAL (farther from green): {n_suboptimal}/{n_total} ({100*n_suboptimal/n_total:.1f}%)")
    print(f"  No blue reached:              {n_none}/{n_total}")

    # Statistical interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    if n_optimal + n_suboptimal > 0:
        opt_rate = n_optimal / (n_optimal + n_suboptimal)
        print(f"\nOptimal choice rate: {100*opt_rate:.1f}%")

        if opt_rate > 0.65:
            print("\n>>> EVIDENCE OF PLANNING!")
            print("    Agent prefers blue closer to green even when equidistant.")
        elif opt_rate < 0.35:
            print("\n>>> ANTI-PLANNING?")
            print("    Agent prefers blue FARTHER from green. Unexpected!")
        else:
            print("\n>>> NO CLEAR PREFERENCE")
            print("    Agent chooses roughly randomly when equidistant.")
            print("    Confirms myopic behavior - no chained distance computation.")

    # Create summary plot
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Optimal\n(closer to green)', 'Suboptimal\n(farther from green)', 'None']
    counts = [n_optimal, n_suboptimal, n_none]
    colors = ['#2ecc71', '#e67e22', '#95a5a6']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=n_total/2, color='red', linestyle='--', label='Random baseline (50%)')

    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Equidistant Test: Which Blue Zone Does Agent Choose?\n(n={n_total} trials, both blues at d={d1:.2f} from agent)',
                fontsize=12, fontweight='bold')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / n_total if n_total > 0 else 0
        ax.annotate(f'{count}\n({pct:.0f}%)',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {out_dir}/")


if __name__ == '__main__':
    main()
