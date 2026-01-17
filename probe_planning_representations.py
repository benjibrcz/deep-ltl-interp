#!/usr/bin/env python3
"""
Probe for planning-related representations in the DeepLTL agent.

This script collects activations from different layers of the model and trains
linear probes to decode planning-relevant information:

1. Blocking Detection: Can we decode whether each goal is blocked?
2. Distance Encoding: Can we decode distances to each zone?
3. Chained Distance: Can we decode blue→green distances for optimality planning?
4. Value Decomposition: Does value function encode planning information?

Architecture layers we probe:
- env_embedding: Physical state encoding (position, lidar)
- ltl_embedding: Task/goal encoding from LTL net
- combined_embedding: env + ltl concatenated
- actor_hidden: Hidden state of actor network
- value: Value function output
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
COLORS = ['blue', 'green', 'yellow', 'magenta']


class ActivationCollector:
    """Collects activations from model layers during rollouts."""

    def __init__(self, model, propositions):
        self.model = model
        self.propositions = propositions
        self.activations = {}
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        # We'll capture activations by modifying forward pass instead
        pass

    def get_activations(self, obs):
        """Get all intermediate activations for a single observation."""
        if not isinstance(obs, list):
            obs = [obs]

        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)

        # Compute env embedding
        if self.model.env_net is not None:
            env_embedding = self.model.env_net(preprocessed.features)
        else:
            env_embedding = preprocessed.features

        # Compute LTL embedding
        ltl_embedding = self.model.ltl_net(preprocessed.seq)

        # Combined embedding
        combined_embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        # Actor hidden state (after encoder)
        actor_hidden = self.model.actor.enc(combined_embedding)

        # Value
        value = self.model.critic(combined_embedding).squeeze(1)

        # Action distribution
        dist = self.model.actor(combined_embedding)
        action_mean = dist.dist.mean  # MixedDistribution stores underlying dist in .dist

        return {
            'env_embedding': env_embedding.detach().numpy(),
            'ltl_embedding': ltl_embedding.detach().numpy(),
            'combined_embedding': combined_embedding.detach().numpy(),
            'actor_hidden': actor_hidden.detach().numpy(),
            'value': value.detach().numpy(),
            'action_mean': action_mean.detach().numpy(),
        }


def get_env_internals(env):
    """Get the Builder layer of the environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def compute_blocking_labels(agent_pos, goal_pos, blocker_positions, threshold=0.5):
    """
    Compute whether goal is blocked by any blocker zone.

    A goal is "blocked" if a blocker lies roughly on the path from agent to goal.
    Simple heuristic: blocker is blocking if it's closer to the line segment
    from agent to goal than the blocker radius.
    """
    if len(blocker_positions) == 0:
        return False

    agent = np.array(agent_pos)
    goal = np.array(goal_pos)

    # Direction from agent to goal
    direction = goal - agent
    dist_to_goal = np.linalg.norm(direction)
    if dist_to_goal < 0.01:
        return False
    direction = direction / dist_to_goal

    for blocker in blocker_positions:
        blocker = np.array(blocker)
        # Vector from agent to blocker
        to_blocker = blocker - agent
        # Project onto direction
        proj_length = np.dot(to_blocker, direction)
        # Only consider blockers between agent and goal
        if proj_length < 0 or proj_length > dist_to_goal:
            continue
        # Distance from blocker to the line
        proj_point = agent + proj_length * direction
        dist_to_line = np.linalg.norm(blocker - proj_point)
        if dist_to_line < ZONE_RADIUS + threshold:
            return True

    return False


def create_task_sampler(task_type, propositions):
    """Create a task sampler for a specific task type."""

    if task_type == 'safety':
        # (F green | F yellow) & G !blue
        green = Assignment.single_proposition('green', propositions).to_frozen()
        yellow = Assignment.single_proposition('yellow', propositions).to_frozen()
        reach = frozenset([green, yellow])
        blue = Assignment.single_proposition('blue', propositions).to_frozen()
        avoid = frozenset([blue])
        task = LDBASequence([(reach, avoid)], repeat_last=100)

    elif task_type == 'optimality':
        # F blue & F green (reach both)
        blue = Assignment.single_proposition('blue', propositions).to_frozen()
        reach_blue = frozenset([blue])
        green = Assignment.single_proposition('green', propositions).to_frozen()
        reach_green = frozenset([green])
        no_avoid = frozenset()
        task = LDBASequence([(reach_blue, no_avoid), (reach_green, no_avoid)])

    elif task_type == 'simple_reach':
        # F green (simple reach)
        green = Assignment.single_proposition('green', propositions).to_frozen()
        reach = frozenset([green])
        no_avoid = frozenset()
        task = LDBASequence([(reach, no_avoid)])

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return lambda: task


def collect_probing_data(env, model, propositions, task_type, n_episodes=100, max_steps=200):
    """Collect activations and ground truth labels for probing."""

    collector = ActivationCollector(model, propositions)

    data = defaultdict(list)

    for ep in range(n_episodes):
        obs, info = env.reset(), {}
        builder = get_env_internals(env)

        # Get zone positions
        zone_positions = {}
        if hasattr(builder, 'zone_positions'):
            for name, pos in builder.zone_positions.items():
                color = name.split('_')[0]
                if color not in zone_positions:
                    zone_positions[color] = []
                zone_positions[color].append(pos[:2].copy())

        for step in range(max_steps):
            # Get agent position
            agent_pos = builder.agent_pos[:2].copy()

            # Get activations
            activations = collector.get_activations(obs)

            # Compute ground truth labels
            labels = {}

            # Distances to each color's zones
            for color in COLORS:
                if color in zone_positions and len(zone_positions[color]) > 0:
                    dists = [np.linalg.norm(agent_pos - z) for z in zone_positions[color]]
                    labels[f'dist_{color}_min'] = min(dists)
                    labels[f'dist_{color}_mean'] = np.mean(dists)
                    for i, d in enumerate(dists):
                        labels[f'dist_{color}_{i}'] = d

            # Blocking labels (for safety task)
            blue_zones = zone_positions.get('blue', [])
            for goal_color in ['green', 'yellow']:
                if goal_color in zone_positions:
                    for i, goal_pos in enumerate(zone_positions[goal_color]):
                        blocked = compute_blocking_labels(agent_pos, goal_pos, blue_zones)
                        labels[f'{goal_color}_{i}_blocked'] = float(blocked)

            # Chained distances (for optimality task)
            # Distance from each blue zone to green
            if 'blue' in zone_positions and 'green' in zone_positions:
                green_pos = zone_positions['green'][0] if zone_positions['green'] else None
                if green_pos is not None:
                    for i, blue_pos in enumerate(zone_positions['blue']):
                        blue_to_green = np.linalg.norm(np.array(blue_pos) - np.array(green_pos))
                        labels[f'blue_{i}_to_green'] = blue_to_green
                        # Total path through this blue
                        agent_to_blue = np.linalg.norm(agent_pos - np.array(blue_pos))
                        labels[f'total_via_blue_{i}'] = agent_to_blue + blue_to_green

            # Store data
            for key, val in activations.items():
                data[f'act_{key}'].append(val.flatten())
            for key, val in labels.items():
                data[f'label_{key}'].append(val)

            data['agent_pos'].append(agent_pos.copy())
            data['step'].append(step)
            data['episode'].append(ep)

            # Take action
            with torch.no_grad():
                obs_list = [obs] if not isinstance(obs, list) else obs
                preprocessed = preprocessing.preprocess_obss(obs_list, propositions)
                dist, _ = model(preprocessed)
                action = dist.mode.numpy().flatten()

            obs, reward, done, info = env.step(action)

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(f"  Collected episode {ep + 1}/{n_episodes}")

    # Convert to numpy arrays
    result = {}
    for key, values in data.items():
        try:
            result[key] = np.array(values)
        except:
            result[key] = values

    return result


def train_linear_probe(X, y, probe_type='regression'):
    """Train a linear probe and return performance metrics."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if probe_type == 'regression':
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        return {
            'model': model,
            'r2': score,
            'y_test': y_test,
            'y_pred': y_pred,
            'type': 'regression'
        }
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return {
            'model': model,
            'accuracy': score,
            'y_test': y_test,
            'y_pred': y_pred,
            'type': 'classification'
        }


def run_probing_analysis(data, output_dir):
    """Run all probing analyses and save results."""

    results = {}

    # Get activation layer names
    act_keys = [k for k in data.keys() if k.startswith('act_')]
    label_keys = [k for k in data.keys() if k.startswith('label_')]

    print(f"\nActivation layers: {[k.replace('act_', '') for k in act_keys]}")
    print(f"Labels available: {[k.replace('label_', '') for k in label_keys]}")

    # Probe each activation layer for each label
    for act_key in act_keys:
        layer_name = act_key.replace('act_', '')
        X = data[act_key]
        print(f"\n--- Probing {layer_name} (dim={X.shape[1]}) ---")

        layer_results = {}

        for label_key in label_keys:
            label_name = label_key.replace('label_', '')
            y = data[label_key]

            # Skip if constant
            if np.std(y) < 1e-6:
                continue

            # Determine probe type
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:
                probe_type = 'classification'
            else:
                probe_type = 'regression'

            result = train_linear_probe(X, y, probe_type)
            layer_results[label_name] = result

            metric = 'accuracy' if probe_type == 'classification' else 'r2'
            score = result.get(metric, 0)
            print(f"  {label_name}: {metric}={score:.3f}")

        results[layer_name] = layer_results

    # Save results
    np.save(output_dir / 'probe_results.npy', results, allow_pickle=True)

    return results


def plot_probe_results(results, output_dir):
    """Create visualizations of probing results."""

    # Collect all metrics
    layers = list(results.keys())

    # Get all labels that appear in any layer
    all_labels = set()
    for layer_results in results.values():
        all_labels.update(layer_results.keys())
    all_labels = sorted(all_labels)

    # Create heatmap of R² scores for regression probes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter for distance labels (regression)
    dist_labels = [l for l in all_labels if 'dist' in l or 'total_via' in l or 'to_green' in l]

    if dist_labels:
        matrix = np.zeros((len(layers), len(dist_labels)))
        for i, layer in enumerate(layers):
            for j, label in enumerate(dist_labels):
                if label in results[layer]:
                    matrix[i, j] = results[layer][label].get('r2', 0)

        im = axes[0].imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_xticks(range(len(dist_labels)))
        axes[0].set_xticklabels(dist_labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticks(range(len(layers)))
        axes[0].set_yticklabels(layers)
        axes[0].set_title('Distance Probes (R² score)')
        plt.colorbar(im, ax=axes[0])

    # Filter for blocking labels (classification)
    block_labels = [l for l in all_labels if 'blocked' in l]

    if block_labels:
        matrix = np.zeros((len(layers), len(block_labels)))
        for i, layer in enumerate(layers):
            for j, label in enumerate(block_labels):
                if label in results[layer]:
                    matrix[i, j] = results[layer][label].get('accuracy', 0)

        im = axes[1].imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1)
        axes[1].set_xticks(range(len(block_labels)))
        axes[1].set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticks(range(len(layers)))
        axes[1].set_yticklabels(layers)
        axes[1].set_title('Blocking Probes (Accuracy)')
        plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_dir / 'probe_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved heatmap to {output_dir / 'probe_heatmap.png'}")

    # Bar chart of best layer for each probe
    fig, ax = plt.subplots(figsize=(12, 6))

    labels_to_plot = dist_labels[:6] + block_labels[:4]  # Limit for readability
    best_scores = []
    best_layers = []

    for label in labels_to_plot:
        best_score = 0
        best_layer = None
        for layer in layers:
            if label in results[layer]:
                r = results[layer][label]
                score = r.get('r2', r.get('accuracy', 0))
                if score > best_score:
                    best_score = score
                    best_layer = layer
        best_scores.append(best_score)
        best_layers.append(best_layer or 'none')

    bars = ax.bar(range(len(labels_to_plot)), best_scores, color='steelblue')
    ax.set_xticks(range(len(labels_to_plot)))
    ax.set_xticklabels(labels_to_plot, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best Score (R² or Accuracy)')
    ax.set_title('Best Probing Performance by Label')
    ax.set_ylim(0, 1)

    # Add layer names on bars
    for bar, layer in zip(bars, best_layers):
        height = bar.get_height()
        ax.annotate(layer.replace('_embedding', ''),
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / 'probe_best_layers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved best layers chart to {output_dir / 'probe_best_layers.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='planning_from_baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='safety', choices=['safety', 'optimality', 'simple_reach'])
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--out_dir', default='probe_results')
    parser.add_argument('--use_fixed_env', action='store_true',
                       help='Use fixed environment (deterministic zones)')
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_name = 'PointLtl2-v0'

    print("="*60)
    print("PROBING PLANNING REPRESENTATIONS")
    print("="*60)
    print(f"Model: {args.exp}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Fixed env: {args.use_fixed_env}")

    # Create environment
    if args.use_fixed_env:
        base_env = safety_gymnasium.make('PointLtl2-v0.fixed')
    else:
        base_env = safety_gymnasium.make(env_name)

    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    propositions = base_env.get_propositions()

    # Create task sampler
    task_sampler = create_task_sampler(args.task, propositions)
    env = SequenceWrapper(base_env, task_sampler)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    env = RemoveTruncWrapper(env)

    # Load model
    print(f"\nLoading model: {env_name} / {args.exp} / seed={args.seed}")
    config = model_configs[env_name]
    model_store = ModelStore(env_name, args.exp, args.seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)
    model.eval()

    props = set(env.get_propositions())

    # Collect data
    print(f"\nCollecting probing data...")
    data = collect_probing_data(env, model, props, args.task, args.n_episodes, args.max_steps)

    print(f"\nCollected {len(data['agent_pos'])} data points")

    # Save raw data
    np.savez(output_dir / f'probe_data_{args.task}.npz', **data)
    print(f"Saved raw data to {output_dir / f'probe_data_{args.task}.npz'}")

    # Run probing analysis
    print("\n" + "="*60)
    print("PROBING ANALYSIS")
    print("="*60)

    results = run_probing_analysis(data, output_dir)

    # Plot results
    plot_probe_results(results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Find best probes for key planning features
    key_probes = [
        ('Distance to nearest goal', 'dist_green_min', 'dist_yellow_min'),
        ('Blocking detection', 'green_0_blocked', 'yellow_0_blocked'),
        ('Chained distance (optimality)', 'total_via_blue_0', 'total_via_blue_1'),
        ('Blue to green distance', 'blue_0_to_green', 'blue_1_to_green'),
    ]

    for name, *labels in key_probes:
        print(f"\n{name}:")
        for label in labels:
            best_score = 0
            best_layer = None
            for layer, layer_results in results.items():
                if label in layer_results:
                    r = layer_results[label]
                    score = r.get('r2', r.get('accuracy', 0))
                    if score > best_score:
                        best_score = score
                        best_layer = layer
            if best_layer:
                print(f"  {label}: best={best_score:.3f} in {best_layer}")

    env.close()
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
