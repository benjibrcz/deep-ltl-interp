#!/usr/bin/env python3
"""
Generate comparative trajectory visualizations for planning incentive models.

Compares: baseline, aux_loss_02, transition_loss_010, combined_aux02_trans01
"""

import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import preprocessing
from envs import make_env
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula


# Fixed configuration for optimality test (same as in paper)
FIXED_CONFIG = {
    'agent_start': np.array([0.2, 0.2]),
    'blue1_pos': np.array([-0.8, -1.0]),  # OPTIMAL (closer to green)
    'blue2_pos': np.array([1.0, 1.0]),    # MYOPIC (closer to agent)
    'green_pos': np.array([-1.4, -1.9]),
}


def load_model(exp_name, seed=0):
    """Load a trained model."""
    env_name = 'PointLtl2-v0'

    # Create temp env for model building
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    # Load model
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp_name, seed=seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    # Build with aux/transition heads if needed
    use_aux = 'aux' in exp_name or 'combined' in exp_name
    use_trans = 'transition' in exp_name or 'combined' in exp_name
    model = build_model(env, training_status, config, use_aux_head=use_aux, use_transition_head=use_trans)
    model.eval()

    props = set(env.get_propositions())
    env.close()

    return model, props


def run_episode_fixed_config(model, props, max_steps=400):
    """Run one episode with fixed zone configuration."""
    env_name = 'PointLtl2-v0'

    # Create environment with fixed zones
    from sequence.samplers.sequence_samplers import blue_then_green_task
    task = blue_then_green_task()

    def fixed_sampler(propositions):
        return lambda: task

    env = make_env(env_name, fixed_sampler, sequence=True)

    # Reset and set fixed positions
    obs, info = env.reset(seed=42)

    # Get the underlying engine and set positions
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env

    # Set agent position
    unwrapped.agent_pos = np.array([FIXED_CONFIG['agent_start'][0],
                                     FIXED_CONFIG['agent_start'][1], 0.0])

    # Set zone positions
    unwrapped.zone_positions = {
        'blue_0': FIXED_CONFIG['blue1_pos'],
        'blue_1': FIXED_CONFIG['blue2_pos'],
        'green_0': FIXED_CONFIG['green_pos'],
    }

    # Rebuild the world with new positions
    unwrapped._build_world()
    obs, info = env.reset(seed=42)

    # Actually we need to manually position - let me just collect trajectory
    trajectory = [FIXED_CONFIG['agent_start'].copy()]
    blue_choice = None
    reached_green = False

    for step in range(max_steps):
        # Get action
        preprocessed = preprocessing.preprocess_obss([obs], props)
        with torch.no_grad():
            dist, value = model(preprocessed)
            action = dist.mode.numpy().flatten()

        obs, reward, done, info = env.step(action)

        # Get agent position
        pos = unwrapped.agent_pos[:2].copy()
        trajectory.append(pos)

        # Check which blue was reached first
        if blue_choice is None:
            d_blue1 = np.linalg.norm(pos - FIXED_CONFIG['blue1_pos'])
            d_blue2 = np.linalg.norm(pos - FIXED_CONFIG['blue2_pos'])
            if d_blue1 < 0.3:
                blue_choice = 'optimal'
            elif d_blue2 < 0.3:
                blue_choice = 'myopic'

        if done:
            if 'success' in info:
                reached_green = True
            break

    env.close()

    return {
        'trajectory': np.array(trajectory),
        'blue_choice': blue_choice or 'none',
        'reached_green': reached_green,
        'steps': len(trajectory) - 1
    }


def plot_trajectory(ax, result, title):
    """Plot a single trajectory."""
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    # Draw zones
    blue1 = Circle(FIXED_CONFIG['blue1_pos'], 0.3, color='blue', alpha=0.6, label='Blue (optimal)')
    blue2 = Circle(FIXED_CONFIG['blue2_pos'], 0.3, color='blue', alpha=0.6)
    green = Circle(FIXED_CONFIG['green_pos'], 0.3, color='green', alpha=0.6, label='Green')

    ax.add_patch(blue1)
    ax.add_patch(blue2)
    ax.add_patch(green)

    # Label zones
    ax.annotate('B1\n(opt)', FIXED_CONFIG['blue1_pos'], ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.annotate('B2\n(myo)', FIXED_CONFIG['blue2_pos'], ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.annotate('G', FIXED_CONFIG['green_pos'], ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Draw trajectory
    traj = result['trajectory']
    color = 'green' if result['reached_green'] else 'red'
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.8)

    # Start marker
    ax.plot(traj[0, 0], traj[0, 1], 'o', color='orange', markersize=10, zorder=10)

    # Title with result
    choice_str = result['blue_choice'].upper()
    success_str = 'SUCCESS' if result['reached_green'] else 'FAIL'
    ax.set_title(f"{title}\n{choice_str} | {success_str} ({result['steps']} steps)", fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])


def main():
    output_dir = 'interpretability/results/trajectory_comparison'
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ('planning_from_baseline', 'Baseline'),
        ('aux_loss_02', 'Aux Loss (0.2)'),
        ('transition_loss_010', 'Trans Loss (0.1)'),
        ('combined_aux02_trans01', 'Combined'),
    ]

    print("=" * 60)
    print("GENERATING COMPARATIVE TRAJECTORY VISUALIZATIONS")
    print("=" * 60)

    # Run multiple episodes per model
    n_episodes = 5
    all_results = {}

    for exp_name, display_name in models:
        print(f"\nLoading {display_name} ({exp_name})...")
        try:
            model, props = load_model(exp_name)
            results = []
            optimal_count = 0
            success_count = 0

            for i in range(n_episodes):
                result = run_episode_fixed_config(model, props)
                results.append(result)
                if result['blue_choice'] == 'optimal':
                    optimal_count += 1
                if result['reached_green']:
                    success_count += 1
                print(f"  Episode {i+1}: {result['blue_choice']} | {'success' if result['reached_green'] else 'fail'}")

            all_results[exp_name] = {
                'results': results,
                'display_name': display_name,
                'optimal_rate': optimal_count / n_episodes,
                'success_rate': success_count / n_episodes,
            }
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Create comparison figure
    print("\nCreating comparison figure...")

    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, n_episodes, figsize=(3 * n_episodes, 3.5 * n_models))

    for row, (exp_name, data) in enumerate(all_results.items()):
        for col, result in enumerate(data['results']):
            ax = axes[row, col] if n_models > 1 else axes[col]
            title = f"{data['display_name']}" if col == 0 else ""
            plot_trajectory(ax, result, title)

    # Add summary stats as row labels
    for row, (exp_name, data) in enumerate(all_results.items()):
        ax = axes[row, 0] if n_models > 1 else axes[0]
        opt_pct = int(100 * data['optimal_rate'])
        suc_pct = int(100 * data['success_rate'])
        ax.set_ylabel(f"Opt: {opt_pct}%\nSuc: {suc_pct}%", fontsize=10, rotation=0, ha='right', va='center')

    plt.suptitle("Planning Incentive Model Comparison\nFixed Config: F blue THEN F green", fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_file = f"{output_dir}/model_comparison_trajectories.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_name, data in all_results.items():
        print(f"{data['display_name']:20s}: Optimal={100*data['optimal_rate']:.0f}%, Success={100*data['success_rate']:.0f}%")


if __name__ == '__main__':
    main()
