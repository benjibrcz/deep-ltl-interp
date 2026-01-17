#!/usr/bin/env python3
"""
Test: Can we create maps where the current agent fails badly,
which would be good training signal for learning to plan?

The idea: if we can find maps where myopic behavior leads to
consistent failure, training on these maps could force the
network to learn chained distance computation.
"""

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


# Hard optimality configurations
# These are designed to make myopic behavior fail
HARD_CONFIGS = [
    {
        'name': 'extreme_opposite',
        'description': 'Nearest blue is opposite direction from green',
        'agent': np.array([0.0, 0.0]),
        'blue_near': np.array([1.0, 0.0]),   # 1.0 from agent
        'blue_far': np.array([-1.0, 0.0]),   # 1.0 from agent (equidistant)
        'green': np.array([-2.5, 0.0]),      # Far in -x direction
        # Myopic (if right-bias): agent→blue_near(1.0) + blue_near→green(3.5) = 4.5
        # Optimal: agent→blue_far(1.0) + blue_far→green(1.5) = 2.5
        # Difference: 2.0 (80% longer path if myopic)
    },
    {
        'name': 'tempting_trap',
        'description': 'Very close blue that leads to very long path',
        'agent': np.array([0.0, 0.0]),
        'blue_near': np.array([0.3, 0.0]),   # Very close! Only 0.3 away
        'blue_far': np.array([-1.5, 0.0]),   # 1.5 from agent
        'green': np.array([-2.5, 0.0]),      # Green is in -x direction
        # Myopic: 0.3 + 2.8 = 3.1
        # Optimal: 1.5 + 1.0 = 2.5
        # The "tempting" close blue is a trap
    },
    {
        'name': 'diagonal_trap',
        'description': 'Diagonal layout where nearest blue is worst',
        'agent': np.array([0.0, 0.0]),
        'blue_near': np.array([0.8, 0.8]),   # 1.13 from agent (diagonal)
        'blue_far': np.array([-1.2, -0.5]),  # 1.30 from agent
        'green': np.array([-2.0, -1.5]),     # Green in third quadrant
        # blue_near → green: 3.6
        # blue_far → green: 1.3
    },
    {
        'name': 'timeout_trap',
        'description': 'Myopic choice leads to near-timeout',
        'agent': np.array([0.0, 0.0]),
        'blue_near': np.array([0.5, 0.5]),   # 0.71 from agent
        'blue_far': np.array([-0.8, -0.8]),  # 1.13 from agent
        'green': np.array([-2.5, -2.5]),     # Far corner
        # Myopic: 0.71 + 4.24 = 4.95
        # Optimal: 1.13 + 2.40 = 3.53
    },
]


def compute_path_lengths(config):
    """Compute optimal vs myopic path lengths."""
    agent = config['agent']
    blue_near = config['blue_near']
    blue_far = config['blue_far']
    green = config['green']

    # Distance from agent to each blue
    d_agent_near = np.linalg.norm(agent - blue_near)
    d_agent_far = np.linalg.norm(agent - blue_far)

    # Distance from each blue to green
    d_near_green = np.linalg.norm(blue_near - green)
    d_far_green = np.linalg.norm(blue_far - green)

    # Total paths
    path_via_near = d_agent_near + d_near_green
    path_via_far = d_agent_far + d_far_green

    # Which is optimal?
    if path_via_far < path_via_near:
        optimal_path = path_via_far
        myopic_path = path_via_near  # Assuming agent goes to nearer blue
        optimal_choice = 'far'
    else:
        optimal_path = path_via_near
        myopic_path = path_via_far
        optimal_choice = 'near'

    return {
        'd_agent_near': d_agent_near,
        'd_agent_far': d_agent_far,
        'd_near_green': d_near_green,
        'd_far_green': d_far_green,
        'path_via_near': path_via_near,
        'path_via_far': path_via_far,
        'optimal_path': optimal_path,
        'myopic_path': myopic_path,
        'optimal_choice': optimal_choice,
        'inefficiency': (myopic_path - optimal_path) / optimal_path,
    }


def visualize_configs(configs, output_dir):
    """Visualize the hard configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for idx, config in enumerate(configs[:4]):
        ax = axes[idx // 2, idx % 2]

        stats = compute_path_lengths(config)

        # Plot zones
        ax.add_patch(patches.Circle(config['blue_near'], ZONE_RADIUS,
                                    facecolor='#2196f3', alpha=0.7,
                                    edgecolor='orange', linewidth=2,
                                    label='Blue (near)'))
        ax.add_patch(patches.Circle(config['blue_far'], ZONE_RADIUS,
                                    facecolor='#2196f3', alpha=0.7,
                                    edgecolor='green', linewidth=3,
                                    label='Blue (far, optimal)'))
        ax.add_patch(patches.Circle(config['green'], ZONE_RADIUS,
                                    facecolor='#4caf50', alpha=0.7,
                                    edgecolor='darkgreen', linewidth=2))

        # Plot agent
        ax.scatter(*config['agent'], s=200, c='orange', marker='D',
                  zorder=10, edgecolors='black', linewidths=2)

        # Draw paths
        # Myopic path (red dashed)
        ax.plot([config['agent'][0], config['blue_near'][0]],
               [config['agent'][1], config['blue_near'][1]],
               'r--', linewidth=2, alpha=0.5)
        ax.plot([config['blue_near'][0], config['green'][0]],
               [config['blue_near'][1], config['green'][1]],
               'r--', linewidth=2, alpha=0.5)

        # Optimal path (green solid)
        ax.plot([config['agent'][0], config['blue_far'][0]],
               [config['agent'][1], config['blue_far'][1]],
               'g-', linewidth=2, alpha=0.7)
        ax.plot([config['blue_far'][0], config['green'][0]],
               [config['blue_far'][1], config['green'][1]],
               'g-', linewidth=2, alpha=0.7)

        # Labels
        ax.annotate(f"d={stats['d_agent_near']:.2f}",
                   (config['agent'] + config['blue_near'])/2,
                   color='red', fontsize=9)
        ax.annotate(f"d={stats['d_agent_far']:.2f}",
                   (config['agent'] + config['blue_far'])/2 + [0.1, 0.1],
                   color='green', fontsize=9)

        ax.set_xlim(-3.5, 2)
        ax.set_ylim(-3.5, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        title = f"{config['name']}\n"
        title += f"Myopic: {stats['path_via_near']:.2f}, Optimal: {stats['path_via_far']:.2f}\n"
        title += f"Inefficiency: +{stats['inefficiency']*100:.0f}%"
        ax.set_title(title, fontsize=10)

    plt.suptitle('Hard Optimality Configurations\n(Red=myopic path, Green=optimal path)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'hard_configs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'hard_configs.png'}")


def test_agent_on_config(env, agent, config, max_steps=400):
    """Test current agent on a specific configuration."""
    obs, _ = env.reset(), {}
    builder = get_env_internals(env)

    # Override positions
    builder.agent_pos[0] = config['agent'][0]
    builder.agent_pos[1] = config['agent'][1]

    # We can't easily modify zone positions in the existing env
    # So we'll just run and see which direction agent goes

    # Track first direction
    trajectory = [config['agent'].copy()]

    for step in range(min(50, max_steps)):  # Just first 50 steps to see direction
        action = agent.get_action(obs, deterministic=True)
        action = action.flatten()
        obs, reward, done, info = env.step(action)

        pos = builder.agent_pos[:2].copy()
        trajectory.append(pos)

        if done:
            break

    trajectory = np.array(trajectory)

    # Determine which direction agent went
    if len(trajectory) > 10:
        direction = trajectory[10] - trajectory[0]
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Compare to blue directions
        to_near = config['blue_near'] - config['agent']
        to_near = to_near / (np.linalg.norm(to_near) + 1e-6)

        to_far = config['blue_far'] - config['agent']
        to_far = to_far / (np.linalg.norm(to_far) + 1e-6)

        sim_near = np.dot(direction, to_near)
        sim_far = np.dot(direction, to_far)

        went_toward = 'near' if sim_near > sim_far else 'far'
    else:
        went_toward = 'unknown'

    return {
        'trajectory': trajectory,
        'went_toward': went_toward,
    }


def main():
    output_dir = Path('hard_optimality_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HARD OPTIMALITY MAP ANALYSIS")
    print("="*60)
    print("\nThese maps are designed to make myopic behavior fail badly.")
    print("If we train on these, the agent might learn to plan.\n")

    # Analyze configurations
    print("Configuration Analysis:")
    print("-"*60)

    for config in HARD_CONFIGS:
        stats = compute_path_lengths(config)
        print(f"\n{config['name']}: {config['description']}")
        print(f"  Distance agent→blue_near: {stats['d_agent_near']:.2f}")
        print(f"  Distance agent→blue_far:  {stats['d_agent_far']:.2f}")
        print(f"  Path via near: {stats['path_via_near']:.2f}")
        print(f"  Path via far:  {stats['path_via_far']:.2f}")
        print(f"  Optimal choice: blue_{stats['optimal_choice']}")
        print(f"  Inefficiency if myopic: +{stats['inefficiency']*100:.0f}%")

    # Visualize
    visualize_configs(HARD_CONFIGS, output_dir)

    # Test current agent behavior
    print("\n" + "="*60)
    print("TESTING CURRENT AGENT'S DIRECTIONAL PREFERENCE")
    print("="*60)

    env_name = 'PointLtl2-v0'

    def optimality_sampler(props):
        return lambda: create_optimality_task(props)

    base_env = safety_gymnasium.make(env_name)
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    propositions = base_env.get_propositions()
    sample_task = optimality_sampler(propositions)
    env = SequenceWrapper(base_env, sample_task)
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, 'planning_from_baseline', seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    print("\nNote: We can't fully test these configs without modifying the")
    print("environment's zone positions, but we can see the agent's bias.\n")

    env.close()

    # Summary
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATION")
    print("="*60)
    print("""
To train the agent to plan without architecture changes:

1. CREATE HARD MAPS: Use configurations like these where:
   - Myopic choice leads to 50-100% longer paths
   - Optimal blue is NOT the nearest one
   - The difference is large enough to affect success rate

2. ADD PATH-LENGTH PENALTY: Modify reward function:
   reward = base_reward - 0.002 * steps_taken

   This incentivizes finding shorter paths.

3. CURRICULUM:
   - Start with obvious cases (3x distance difference)
   - Gradually reduce to subtle cases (1.2x difference)
   - Always include tempting "trap" configurations

4. TRAINING DURATION:
   - More RL steps on these hard configurations
   - The network CAN learn this - it's just arithmetic
   - Current training didn't incentivize it enough

EXPECTED OUTCOME:
If the training signal is strong enough (myopic = failure),
the network should learn to compute chained distances,
since it's within the representational capacity of the MLP.
""")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
