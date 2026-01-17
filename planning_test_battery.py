#!/usr/bin/env python3
"""
LOCAL vs GLOBAL Planning Test Battery

Tests whether an agent uses local (myopic/greedy) or global (optimal) planning.

LOCAL Planning Tests:
- Scenarios where greedy behavior is optimal
- Nearest intermediate goal IS the best choice
- Tests: agent should succeed with high path efficiency

GLOBAL Planning Tests:
- Scenarios where greedy behavior is suboptimal
- Nearest intermediate goal leads to longer total path
- Tests: does agent choose optimal or myopic path?

Metrics:
- Success rate: Did agent complete the task?
- Path efficiency: Actual path length / optimal path length
- Optimal choice rate: Chose globally optimal intermediate goal?
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class TestConfig:
    """Configuration for a planning test scenario."""
    name: str
    description: str
    category: str  # 'local' or 'global'
    agent_start: np.ndarray
    blue1: np.ndarray  # First blue zone position
    blue2: np.ndarray  # Second blue zone position
    green: np.ndarray  # Final goal position

    def __post_init__(self):
        self.agent_start = np.array(self.agent_start)
        self.blue1 = np.array(self.blue1)
        self.blue2 = np.array(self.blue2)
        self.green = np.array(self.green)

    @property
    def distances(self):
        """Compute all relevant distances."""
        d_agent_b1 = np.linalg.norm(self.agent_start - self.blue1)
        d_agent_b2 = np.linalg.norm(self.agent_start - self.blue2)
        d_b1_green = np.linalg.norm(self.blue1 - self.green)
        d_b2_green = np.linalg.norm(self.blue2 - self.green)

        path_via_b1 = d_agent_b1 + d_b1_green
        path_via_b2 = d_agent_b2 + d_b2_green

        return {
            'd_agent_b1': d_agent_b1,
            'd_agent_b2': d_agent_b2,
            'd_b1_green': d_b1_green,
            'd_b2_green': d_b2_green,
            'path_via_b1': path_via_b1,
            'path_via_b2': path_via_b2,
            'optimal_path': min(path_via_b1, path_via_b2),
            'optimal_blue': 0 if path_via_b1 <= path_via_b2 else 1,
            'nearer_blue': 0 if d_agent_b1 <= d_agent_b2 else 1,
        }


# LOCAL Planning Tests - Greedy is Optimal
LOCAL_TESTS = [
    TestConfig(
        name='aligned_simple',
        description='Both blues on the path to green, nearer is better',
        category='local',
        agent_start=[0.0, 0.0],
        blue1=[-0.8, 0.0],   # Nearer, on path to green
        blue2=[-1.6, 0.0],   # Farther, also on path
        green=[-2.5, 0.0],
    ),
    TestConfig(
        name='aligned_diagonal',
        description='Blues aligned diagonally toward green',
        category='local',
        agent_start=[0.0, 0.0],
        blue1=[-0.6, -0.6],  # Nearer
        blue2=[-1.2, -1.2],  # Farther
        green=[-2.0, -2.0],
    ),
    TestConfig(
        name='one_near_one_offpath',
        description='One blue near and on-path, other off-path',
        category='local',
        agent_start=[0.0, 0.0],
        blue1=[-0.8, 0.0],   # Near and on-path
        blue2=[0.0, 1.5],    # Far and off-path
        green=[-2.0, 0.0],
    ),
    TestConfig(
        name='corner_to_corner',
        description='Agent in one corner, goal in opposite, nearer blue better',
        category='local',
        agent_start=[1.5, 1.5],
        blue1=[0.5, 0.5],    # Nearer, on diagonal path
        blue2=[1.5, -0.5],   # Farther off-path
        green=[-1.5, -1.5],
    ),
]

# GLOBAL Planning Tests - Greedy is Suboptimal
GLOBAL_TESTS = [
    TestConfig(
        name='opposite_direction',
        description='Nearer blue is opposite direction from green',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[1.0, 0.0],    # Nearer but opposite from green
        blue2=[-1.0, 0.0],   # Farther but toward green
        green=[-2.5, 0.0],
        # Myopic: 1.0 + 3.5 = 4.5
        # Optimal: 1.0 + 1.5 = 2.5
    ),
    TestConfig(
        name='tempting_trap',
        description='Very close blue that leads to long total path',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[0.3, 0.0],    # Very close! Only 0.3 away
        blue2=[-1.5, 0.0],   # 1.5 from agent
        green=[-2.5, 0.0],
        # Myopic: 0.3 + 2.8 = 3.1
        # Optimal: 1.5 + 1.0 = 2.5
    ),
    TestConfig(
        name='diagonal_trap',
        description='Nearest blue is diagonal away from goal',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[0.8, 0.8],    # 1.13 from agent, diagonal
        blue2=[-1.0, -0.5],  # 1.12 from agent
        green=[-2.0, -1.5],
        # blue1→green: 3.6, blue2→green: 1.4
    ),
    TestConfig(
        name='mild_suboptimal',
        description='Slight inefficiency from myopic choice (~25%)',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[0.7, 0.5],    # 0.86 from agent
        blue2=[-0.9, 0.0],   # 0.9 from agent
        green=[-2.0, 0.0],
        # Path via b1: 0.86 + 2.89 = 3.75
        # Path via b2: 0.9 + 1.1 = 2.0
    ),
    TestConfig(
        name='severe_trap',
        description='Myopic leads to 80%+ longer path',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[0.5, 0.5],    # 0.71 from agent
        blue2=[-0.8, -0.8],  # 1.13 from agent
        green=[-2.5, -2.5],
        # Path via b1: 0.71 + 4.24 = 4.95
        # Path via b2: 1.13 + 2.40 = 3.53
    ),
    TestConfig(
        name='equidistant_blues',
        description='Blues equidistant from agent, but different from green',
        category='global',
        agent_start=[0.0, 0.0],
        blue1=[1.0, 0.0],    # 1.0 from agent
        blue2=[0.0, 1.0],    # 1.0 from agent (equidistant)
        green=[-2.0, 1.0],   # b2 is closer to green
        # Path via b1: 1.0 + 3.16 = 4.16
        # Path via b2: 1.0 + 2.0 = 3.0
    ),
]


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


def create_task(propositions):
    """Create F blue THEN F green task."""
    blue = Assignment.single_proposition('blue', propositions).to_frozen()
    green = Assignment.single_proposition('green', propositions).to_frozen()
    return LDBASequence([
        (frozenset([blue]), frozenset()),
        (frozenset([green]), frozenset())
    ])


def get_env_internals(env):
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def run_test(env, agent, test_config: TestConfig, max_steps=400):
    """Run a single test scenario."""
    obs, _ = env.reset(), {}
    builder = get_env_internals(env)

    # Get zone positions from environment
    zone_positions = {}
    if hasattr(builder, 'zone_positions'):
        for name, pos in builder.zone_positions.items():
            zone_positions[name] = pos.copy()

    blue_zones = {k: v for k, v in zone_positions.items() if k.startswith('blue')}
    green_pos = zone_positions.get('green_zone0', None)

    if green_pos is None or len(blue_zones) < 2:
        return None

    blue_positions = list(blue_zones.values())
    agent_start = builder.agent_pos[:2].copy()

    # Calculate which blue is optimal vs myopic
    d_to_blues = [np.linalg.norm(agent_start - b) for b in blue_positions]
    d_blues_to_green = [np.linalg.norm(b - green_pos) for b in blue_positions]
    paths = [d_to_blues[i] + d_blues_to_green[i] for i in range(len(blue_positions))]

    optimal_idx = np.argmin(paths)
    myopic_idx = np.argmin(d_to_blues)
    optimal_path = paths[optimal_idx]

    # Run episode
    trajectory = [agent_start.copy()]
    first_blue_reached = None
    reached_green = False
    steps = 0

    while steps < max_steps:
        action = agent.get_action(obs, deterministic=True).flatten()
        if action.shape == (1,):
            action = action[0]

        obs, reward, done, info = env.step(action)
        steps += 1

        pos = builder.agent_pos[:2].copy()
        trajectory.append(pos.copy())

        # Check blue zones
        if first_blue_reached is None:
            for i, blue in enumerate(blue_positions):
                if np.linalg.norm(pos - blue) < ZONE_RADIUS:
                    first_blue_reached = i
                    break

        # Check green
        if np.linalg.norm(pos - green_pos) < ZONE_RADIUS:
            reached_green = True

        if done:
            break

    # Compute actual path length
    trajectory = np.array(trajectory)
    actual_path = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

    return {
        'test_name': test_config.name,
        'category': test_config.category,
        'success': reached_green,
        'steps': steps,
        'optimal_idx': optimal_idx,
        'myopic_idx': myopic_idx,
        'first_blue_reached': first_blue_reached,
        'chose_optimal': first_blue_reached == optimal_idx if first_blue_reached is not None else None,
        'optimal_path': optimal_path,
        'actual_path': actual_path,
        'path_efficiency': optimal_path / actual_path if actual_path > 0 else 0,
        'trajectory': trajectory,
        'blue_positions': blue_positions,
        'green_pos': green_pos,
        'agent_start': agent_start,
    }


def run_battery(agent, env, tests, n_runs=5):
    """Run the full test battery."""
    all_results = []

    for test_config in tests:
        test_results = []
        for run in range(n_runs):
            env.reset(seed=42 + run)
            result = run_test(env, agent, test_config)
            if result:
                test_results.append(result)
        all_results.extend(test_results)

    return all_results


def summarize_results(results, category=None):
    """Summarize results, optionally filtered by category."""
    if category:
        results = [r for r in results if r['category'] == category]

    if not results:
        return {}

    n_total = len(results)
    n_success = sum(1 for r in results if r['success'])
    n_optimal = sum(1 for r in results if r['chose_optimal'] == True)
    n_myopic = sum(1 for r in results if r['chose_optimal'] == False)
    n_no_choice = sum(1 for r in results if r['first_blue_reached'] is None)

    efficiencies = [r['path_efficiency'] for r in results if r['success']]

    return {
        'category': category or 'all',
        'n_total': n_total,
        'success_rate': n_success / n_total if n_total > 0 else 0,
        'optimal_rate': n_optimal / (n_optimal + n_myopic) if (n_optimal + n_myopic) > 0 else 0,
        'myopic_rate': n_myopic / (n_optimal + n_myopic) if (n_optimal + n_myopic) > 0 else 0,
        'mean_efficiency': np.mean(efficiencies) if efficiencies else 0,
        'std_efficiency': np.std(efficiencies) if efficiencies else 0,
    }


def plot_summary(local_summary, global_summary, output_dir, model_name):
    """Create summary visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of optimal vs myopic rates
    ax = axes[0]
    x = np.arange(2)
    width = 0.35

    optimal_rates = [local_summary['optimal_rate'], global_summary['optimal_rate']]
    myopic_rates = [local_summary['myopic_rate'], global_summary['myopic_rate']]

    bars1 = ax.bar(x - width/2, optimal_rates, width, label='Optimal Choice', color='#4caf50')
    bars2 = ax.bar(x + width/2, myopic_rates, width, label='Myopic Choice', color='#ff9800')

    ax.set_ylabel('Rate')
    ax.set_title(f'Planning Behavior: {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(['LOCAL Tests\n(greedy=optimal)', 'GLOBAL Tests\n(greedy≠optimal)'])
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # Path efficiency
    ax = axes[1]
    efficiencies = [local_summary['mean_efficiency'], global_summary['mean_efficiency']]
    stds = [local_summary['std_efficiency'], global_summary['std_efficiency']]

    bars = ax.bar(x, efficiencies, 0.5, yerr=stds, color=['#2196f3', '#9c27b0'], capsize=5)
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Perfect efficiency')
    ax.set_ylabel('Path Efficiency (optimal/actual)')
    ax.set_title('Path Efficiency by Test Category')
    ax.set_xticks(x)
    ax.set_xticklabels(['LOCAL Tests', 'GLOBAL Tests'])
    ax.set_ylim(0, 1.2)
    ax.legend()

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f'planning_battery_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Local vs Global Planning Test Battery')
    parser.add_argument('--exp', default='planning_from_baseline', help='Experiment name')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=5, help='Runs per test config')
    parser.add_argument('--out_dir', default='planning_battery_results')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name = 'PointLtl2-v0'

    # Create environment
    def task_sampler(props):
        return lambda: create_task(props)

    base_env = safety_gymnasium.make(f'{env_name}.fixed')
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    propositions = base_env.get_propositions()
    env = SequenceWrapper(base_env, task_sampler(propositions))
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)

    # Load model
    print(f"Loading model: {env_name} / {args.exp} / seed={args.seed}")
    config = model_configs[env_name]
    model_store = ModelStore(env_name, args.exp, args.seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    print("\n" + "="*60)
    print("LOCAL vs GLOBAL PLANNING TEST BATTERY")
    print("="*60)

    # Analyze test configurations
    print("\nTEST CONFIGURATIONS:")
    print("-"*60)

    print("\nLOCAL Tests (greedy should be optimal):")
    for test in LOCAL_TESTS:
        d = test.distances
        print(f"  {test.name}: nearer_blue={d['nearer_blue']}, optimal_blue={d['optimal_blue']}")
        print(f"    (paths: via_b1={d['path_via_b1']:.2f}, via_b2={d['path_via_b2']:.2f})")

    print("\nGLOBAL Tests (greedy is suboptimal):")
    for test in GLOBAL_TESTS:
        d = test.distances
        inefficiency = (max(d['path_via_b1'], d['path_via_b2']) / d['optimal_path'] - 1) * 100
        print(f"  {test.name}: nearer_blue={d['nearer_blue']}, optimal_blue={d['optimal_blue']}")
        print(f"    (inefficiency if myopic: +{inefficiency:.0f}%)")

    # Run tests
    print("\n" + "="*60)
    print("RUNNING TESTS...")
    print("="*60)

    local_results = run_battery(agent, env, LOCAL_TESTS, args.n_runs)
    global_results = run_battery(agent, env, GLOBAL_TESTS, args.n_runs)

    all_results = local_results + global_results

    # Summarize
    local_summary = summarize_results(all_results, 'local')
    global_summary = summarize_results(all_results, 'global')
    overall_summary = summarize_results(all_results)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nLOCAL Tests (n={local_summary['n_total']}):")
    print(f"  Success rate: {local_summary['success_rate']:.1%}")
    print(f"  Optimal choice rate: {local_summary['optimal_rate']:.1%}")
    print(f"  Path efficiency: {local_summary['mean_efficiency']:.3f} ± {local_summary['std_efficiency']:.3f}")

    print(f"\nGLOBAL Tests (n={global_summary['n_total']}):")
    print(f"  Success rate: {global_summary['success_rate']:.1%}")
    print(f"  Optimal choice rate: {global_summary['optimal_rate']:.1%}")
    print(f"  Myopic choice rate: {global_summary['myopic_rate']:.1%}")
    print(f"  Path efficiency: {global_summary['mean_efficiency']:.3f} ± {global_summary['std_efficiency']:.3f}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if global_summary['optimal_rate'] > 0.5:
        print("\n>>> EVIDENCE OF GLOBAL PLANNING:")
        print("    Agent chooses intermediate goals based on downstream efficiency")
    elif global_summary['optimal_rate'] < 0.3:
        print("\n>>> MYOPIC (LOCAL) BEHAVIOR:")
        print("    Agent chooses nearest intermediate goal, ignoring downstream cost")
    else:
        print("\n>>> MIXED BEHAVIOR:")
        print("    Agent shows inconsistent planning strategy")

    # Create visualization
    plot_summary(local_summary, global_summary, out_dir, args.exp)

    # Save detailed results
    import json
    with open(out_dir / f'results_{args.exp}.json', 'w') as f:
        json.dump({
            'local_summary': local_summary,
            'global_summary': global_summary,
            'overall_summary': overall_summary,
        }, f, indent=2)

    print(f"\nResults saved to: {out_dir}/")
    env.close()


if __name__ == '__main__':
    main()
