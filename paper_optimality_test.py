#!/usr/bin/env python3
"""
Exact Paper Optimality Test: Replicating Figure 1b from DeepLTL paper

Task: F blue THEN F green (sequential)
- Agent must reach blue FIRST, then reach green
- There are TWO blue zones - agent must choose which one
- Optimal: Choose blue zone closer to GREEN (even if farther from agent)
- Myopic: Choose blue zone closer to AGENT

Fixed configuration:
- Green zone at (-1.4, -1.9) - final goal
- Blue zones (2) at (-0.8, -1) and (1, 1) - intermediate goals
- Agent starts at (0.2, 0.2)

Blue1 (-0.8, -1): closer to green (distance ~1.08), farther from agent (~1.56)
Blue2 (1, 1): farther from green (distance ~3.76), closer to agent (~1.13)

Optimal path: agent → blue1 → green (total ~2.64)
Myopic path: agent → blue2 → green (total ~4.89)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import preprocessing
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.samplers import CurriculumSampler, curricula
from ltl.automata import LDBASequence
from ltl.logic import Assignment

# Import safety gymnasium directly to use the fixed environment
import safety_gymnasium
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

torch.set_grad_enabled(False)

ZONE_RADIUS = 0.4

# Fixed zone positions from the paper (optimality config)
FIXED_CONFIG = {
    'agent_start': np.array([0.2, 0.2]),
    'green': np.array([-1.4, -1.9]),
    'yellow': np.array([1.4, 2.3]),
    'blue1': np.array([-0.8, -1]),      # Closer to green (optimal choice)
    'blue2': np.array([1, 1]),           # Closer to agent (myopic choice)
    'magenta': np.array([-1.7, 1.2]),
}


class SimpleAgent:
    """Simple agent that uses the model with sequence observations."""
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
    """
    Create the F blue THEN F green task (sequential).
    Agent MUST reach blue first, THEN reach green.

    This tests optimality: which blue does the agent choose,
    knowing it will need to reach green afterward?
    """
    # Step 1: Reach blue (any blue zone satisfies this)
    blue_assignment = Assignment.single_proposition('blue', propositions).to_frozen()
    reach_blue = frozenset([blue_assignment])

    # Step 2: Reach green (only after blue is reached)
    green_assignment = Assignment.single_proposition('green', propositions).to_frozen()
    reach_green = frozenset([green_assignment])

    # No avoid constraints
    no_avoid = frozenset()

    # Sequential: reach blue THEN reach green
    return LDBASequence([(reach_blue, no_avoid), (reach_green, no_avoid)])


def make_fixed_env(sampler):
    """Create the fixed environment matching the paper."""
    base_env = safety_gymnasium.make('PointLtl2-v0.fixed')
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)

    propositions = base_env.get_propositions()
    sample_task = sampler(propositions)
    env = SequenceWrapper(base_env, sample_task)
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)
    return env


def get_env_internals(env):
    """Get the Builder layer of the environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def run_optimality_test(env, agent, max_steps=400):
    """Run one episode of the optimality test."""
    obs, info = env.reset(), {}
    builder = get_env_internals(env)

    # Get zone positions
    zone_positions = {}
    if hasattr(builder, 'zone_positions'):
        for name, pos in builder.zone_positions.items():
            zone_positions[name] = pos.copy()

    # Get agent start position
    agent_start = builder.agent_pos[:2].copy() if hasattr(builder, 'agent_pos') else None

    # Extract specific zones
    blue_zones = {k: v for k, v in zone_positions.items() if k.startswith('blue')}
    green_pos = zone_positions.get('green_zone0', None)

    if green_pos is None or len(blue_zones) < 2:
        print(f"Warning: Missing zones. Green: {green_pos}, Blue: {blue_zones}")
        return None

    # Calculate distances
    blue_positions = list(blue_zones.values())

    # Distance from agent to each blue
    dist_agent_to_blue = [np.linalg.norm(agent_start - b) for b in blue_positions]

    # Distance from each blue to green
    dist_blue_to_green = [np.linalg.norm(b - green_pos) for b in blue_positions]

    # Optimal blue is the one closer to green
    optimal_blue_idx = np.argmin(dist_blue_to_green)
    myopic_blue_idx = np.argmin(dist_agent_to_blue)

    optimal_blue = blue_positions[optimal_blue_idx]
    myopic_blue = blue_positions[myopic_blue_idx]

    # Run episode
    trajectory = [agent_start.copy()]
    done = False
    first_blue_reached = None
    reached_green = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.get_action(obs, deterministic=True)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]

        obs, reward, done, info = env.step(action)
        steps += 1

        # Get current position
        agent_pos = builder.agent_pos[:2].copy()
        trajectory.append(agent_pos.copy())

        # Check which blue zone was reached first
        if first_blue_reached is None:
            for i, blue in enumerate(blue_positions):
                if np.linalg.norm(agent_pos - blue) < ZONE_RADIUS:
                    first_blue_reached = i
                    break

        # Check if green reached
        if np.linalg.norm(agent_pos - green_pos) < ZONE_RADIUS:
            reached_green = True

    chose_optimal = first_blue_reached == optimal_blue_idx if first_blue_reached is not None else None

    return {
        'agent_start': agent_start,
        'green_pos': green_pos,
        'blue_positions': blue_positions,
        'optimal_blue_idx': optimal_blue_idx,
        'myopic_blue_idx': myopic_blue_idx,
        'dist_agent_to_blue': dist_agent_to_blue,
        'dist_blue_to_green': dist_blue_to_green,
        'trajectory': trajectory,
        'first_blue_reached': first_blue_reached,
        'reached_green': reached_green,
        'chose_optimal': chose_optimal,
        'steps': steps,
        'magenta_pos': zone_positions.get('magenta_zone0', None),
        'yellow_pos': zone_positions.get('yellow_zone0', None),
        'success': reached_green,
    }


def plot_result(result, filename, run_num):
    """Plot the trajectory showing the optimality test."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors_map = {'blue': '#2196f3', 'green': '#4caf50', 'yellow': '#fdd835', 'magenta': 'violet'}

    # Plot yellow zone (not part of task, just context)
    if result['yellow_pos'] is not None:
        circle = patches.Circle(result['yellow_pos'], ZONE_RADIUS,
                               facecolor=colors_map['yellow'], alpha=0.4,
                               edgecolor='orange', linewidth=1, linestyle='--')
        ax.add_patch(circle)

    # Plot magenta zone (not part of task)
    if result['magenta_pos'] is not None:
        circle = patches.Circle(result['magenta_pos'], ZONE_RADIUS,
                               facecolor=colors_map['magenta'], alpha=0.4,
                               edgecolor='purple', linewidth=1, linestyle='--')
        ax.add_patch(circle)

    # Plot blue zones with labels
    for i, blue in enumerate(result['blue_positions']):
        is_optimal = (i == result['optimal_blue_idx'])
        is_myopic = (i == result['myopic_blue_idx'])

        edgecolor = 'green' if is_optimal else 'orange'
        linewidth = 3 if is_optimal else 2

        circle = patches.Circle(blue, ZONE_RADIUS,
                               facecolor=colors_map['blue'], alpha=0.7,
                               edgecolor=edgecolor, linewidth=linewidth)
        ax.add_patch(circle)

        # Add label
        label = "OPTIMAL\n(closer to green)" if is_optimal else "MYOPIC\n(closer to agent)"
        ax.annotate(label, blue, textcoords="offset points", xytext=(0, -35),
                   ha='center', fontsize=8, color=edgecolor, fontweight='bold')

    # Plot green zone (final goal)
    green = result['green_pos']
    circle = patches.Circle(green, ZONE_RADIUS,
                           facecolor=colors_map['green'], alpha=0.7,
                           edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)
    ax.annotate("GOAL", green, textcoords="offset points", xytext=(0, -30),
               ha='center', fontsize=9, color='darkgreen', fontweight='bold')

    # Plot trajectory
    traj = np.array(result['trajectory'])
    path_color = '#2ecc71' if result['chose_optimal'] else '#e67e22'
    ax.plot(traj[:, 0], traj[:, 1], color=path_color, linewidth=2.5, alpha=0.8)

    # Mark start (orange diamond)
    ax.scatter(traj[0, 0], traj[0, 1], s=200, c='orange', marker='D',
              zorder=10, edgecolors='black', linewidths=2)

    # Mark end
    ax.scatter(traj[-1, 0], traj[-1, 1], s=150, c=path_color, marker='o',
              zorder=10, edgecolors='black', linewidths=2)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Title
    title = f"Run {run_num}: F blue THEN F green\n"

    if result['first_blue_reached'] is not None:
        if result['chose_optimal']:
            title += "CHOSE OPTIMAL BLUE (closer to green) - PLANNING!"
            color = 'green'
        else:
            title += "CHOSE MYOPIC BLUE (closer to agent)"
            color = 'orange'
    else:
        title += "Did not reach blue"
        color = 'gray'

    if result['reached_green']:
        title += f"\nReached green in {result['steps']} steps"

    ax.set_title(title, fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='planning_from_baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--out_dir', default='paper_optimality_results')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name = 'PointLtl2-v0'

    # Create the optimality task sampler
    def optimality_sampler(props):
        return lambda: create_optimality_task(props)

    # Create fixed environment
    env = make_fixed_env(optimality_sampler)

    # Load model
    print(f"Loading model: {env_name} / {args.exp} / seed={args.seed}")
    config = model_configs[env_name]
    model_store = ModelStore(env_name, args.exp, args.seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    # Calculate expected distances
    agent_start = FIXED_CONFIG['agent_start']
    green = FIXED_CONFIG['green']
    blue1 = FIXED_CONFIG['blue1']  # Closer to green
    blue2 = FIXED_CONFIG['blue2']  # Closer to agent

    dist_agent_blue1 = np.linalg.norm(agent_start - blue1)
    dist_agent_blue2 = np.linalg.norm(agent_start - blue2)
    dist_blue1_green = np.linalg.norm(blue1 - green)
    dist_blue2_green = np.linalg.norm(blue2 - green)

    optimal_total = dist_agent_blue1 + dist_blue1_green
    myopic_total = dist_agent_blue2 + dist_blue2_green

    print(f"\nRunning {args.n_runs} optimality test episodes on FIXED environment...")
    print(f"Task: F blue THEN F green")
    print(f"\nFixed configuration (from paper Figure 1b):")
    print(f"  Agent start: {agent_start}")
    print(f"  Blue1 {blue1}: agent→blue={dist_agent_blue1:.2f}, blue→green={dist_blue1_green:.2f}")
    print(f"  Blue2 {blue2}: agent→blue={dist_agent_blue2:.2f}, blue→green={dist_blue2_green:.2f}")
    print(f"  Green: {green}")
    print(f"\n  OPTIMAL path (via Blue1): {optimal_total:.2f} total distance")
    print(f"  MYOPIC path (via Blue2):  {myopic_total:.2f} total distance")
    print(f"  Savings from optimal: {myopic_total - optimal_total:.2f} ({100*(myopic_total-optimal_total)/myopic_total:.0f}%)")
    print("="*60)

    results = []

    for run in range(args.n_runs):
        print(f"\n--- Run {run+1}/{args.n_runs} ---")

        # Reset
        env.reset(seed=args.seed + run)

        result = run_optimality_test(env, agent, args.max_steps)

        if result:
            results.append(result)
            blue_choice = "OPTIMAL (blue1)" if result['chose_optimal'] else "MYOPIC (blue2)" if result['first_blue_reached'] is not None else "None"
            print(f"  Blue choice: {blue_choice}")
            print(f"  Reached green: {result['reached_green']}")
            print(f"  Steps: {result['steps']}")

            # Plot each run
            plot_result(result, out_dir / f'run_{run+1}.png', run+1)

    env.close()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - PAPER OPTIMALITY TEST")
    print("="*60)

    n_total = len(results)
    n_optimal = sum(1 for r in results if r['chose_optimal'] == True)
    n_myopic = sum(1 for r in results if r['chose_optimal'] == False)
    n_no_blue = sum(1 for r in results if r['first_blue_reached'] is None)
    n_reached_green = sum(1 for r in results if r['reached_green'])

    print(f"\nTotal runs: {n_total}")
    print(f"\nBlue zone choice (first subgoal):")
    print(f"  OPTIMAL (closer to green): {n_optimal}/{n_total} ({100*n_optimal/n_total:.1f}%)")
    print(f"  MYOPIC (closer to agent):  {n_myopic}/{n_total} ({100*n_myopic/n_total:.1f}%)")
    print(f"  No blue reached:           {n_no_blue}/{n_total} ({100*n_no_blue/n_total:.1f}%)")

    print(f"\nTask completion:")
    print(f"  Reached green (success): {n_reached_green}/{n_total} ({100*n_reached_green/n_total:.1f}%)")

    if n_optimal > n_myopic:
        print(f"\n>>> EVIDENCE OF PLANNING: Agent chooses blue closer to NEXT goal!")
    elif n_optimal < n_myopic:
        print(f"\n>>> MYOPIC BEHAVIOR: Agent chooses blue closer to ITSELF")
    else:
        print(f"\n>>> MIXED BEHAVIOR")

    print(f"\nResults saved to: {out_dir}/")


if __name__ == '__main__':
    main()
