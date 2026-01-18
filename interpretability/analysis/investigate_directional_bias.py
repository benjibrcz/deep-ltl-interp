#!/usr/bin/env python3
"""
Investigate the source of directional bias in the agent.

Possible causes:
1. Initial agent orientation/heading
2. Learned bias from training distribution
3. Asymmetry in observation space
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def analyze_initial_state(env, n_resets=10):
    """Analyze agent's initial state across resets."""
    print("\n=== AGENT INITIAL STATE ANALYSIS ===")

    orientations = []
    velocities = []

    for i in range(n_resets):
        env.reset(seed=i)
        builder = get_env_internals(env)

        # Get agent state
        if hasattr(builder, 'agent_pos'):
            pos = builder.agent_pos[:2].copy()

        # Check for velocity/orientation in observation
        if hasattr(builder, 'agent_vel'):
            vel = builder.agent_vel[:2].copy()
            velocities.append(vel)

        # Check for rotation/heading
        if hasattr(builder, 'agent_rot'):
            rot = builder.agent_rot
            orientations.append(rot)

    if orientations:
        orientations = np.array(orientations)
        print(f"\nInitial orientations (radians): {orientations}")
        print(f"Mean orientation: {np.mean(orientations):.3f} rad = {np.degrees(np.mean(orientations)):.1f} deg")

    if velocities:
        velocities = np.array(velocities)
        print(f"\nInitial velocities:\n{velocities}")
        print(f"Mean velocity: {np.mean(velocities, axis=0)}")


def analyze_first_actions(env, model, propositions, n_episodes=10):
    """Analyze the agent's first few actions."""
    print("\n=== FIRST ACTION ANALYSIS ===")

    class SimpleAgent:
        def __init__(self, model, propositions):
            self.model = model
            self.propositions = propositions

        def get_action_and_value(self, obs):
            if not isinstance(obs, list):
                obs = [obs]
            preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
            with torch.no_grad():
                dist, value = self.model(preprocessed)
                action = dist.mode
                mean = dist.dist.mean
                std = dist.dist.stddev
            return action.numpy(), value.numpy(), mean.numpy(), std.numpy()

    agent = SimpleAgent(model, propositions)

    first_actions = []
    first_means = []
    first_values = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=i), {}
        action, value, mean, std = agent.get_action_and_value(obs)
        first_actions.append(action.flatten())
        first_means.append(mean.flatten())
        first_values.append(value.flatten())

    first_actions = np.array(first_actions)
    first_means = np.array(first_means)
    first_values = np.array(first_values)

    print(f"\nFirst actions (mode):\n{first_actions}")
    print(f"\nMean first action: {np.mean(first_actions, axis=0)}")
    print(f"Std first action: {np.std(first_actions, axis=0)}")

    print(f"\nFirst action means (from distribution):\n{first_means}")
    print(f"\nMean: {np.mean(first_means, axis=0)}")

    print(f"\nFirst values: {first_values.flatten()}")
    print(f"Mean value: {np.mean(first_values):.3f}")

    # Interpret action direction
    # In PointLtl environment, actions are typically [forward/backward, turn]
    avg_action = np.mean(first_actions, axis=0)
    print(f"\n=== INTERPRETATION ===")
    print(f"Average first action: {avg_action}")
    if len(avg_action) >= 2:
        print(f"  Forward component: {avg_action[0]:.3f}")
        print(f"  Turn component: {avg_action[1]:.3f}")

    return first_actions


def test_different_orientations(env, model, propositions, n_per_orientation=5):
    """Test if we can identify orientation-dependent behavior."""
    print("\n=== ORIENTATION TEST ===")

    # We can't easily change agent orientation in the fixed env
    # But we can look at the observation to understand what the agent "sees"

    obs, _ = env.reset(seed=0), {}
    builder = get_env_internals(env)

    print(f"\nObservation shape: {obs['features'].shape}")
    print(f"Observation (first 20 values): {obs['features'][:20]}")

    # Look for lidar-like patterns
    features = obs['features']
    n_features = len(features)

    # Try to identify segments
    # Typically: position (2-3), velocity (2-3), lidar (many)
    print(f"\nFeature analysis:")
    print(f"  Total features: {n_features}")

    # Check if there's a clear lidar section (usually many similar-magnitude values)
    if n_features > 20:
        potential_lidar = features[6:]  # Skip first few (likely pos/vel)
        print(f"  Potential lidar section ({len(potential_lidar)} values):")
        print(f"    Min: {np.min(potential_lidar):.3f}")
        print(f"    Max: {np.max(potential_lidar):.3f}")
        print(f"    Mean: {np.mean(potential_lidar):.3f}")

        # Check for asymmetry in lidar
        n_lidar = len(potential_lidar)
        first_half = potential_lidar[:n_lidar//2]
        second_half = potential_lidar[n_lidar//2:]
        print(f"\n  Lidar asymmetry check:")
        print(f"    First half mean: {np.mean(first_half):.3f}")
        print(f"    Second half mean: {np.mean(second_half):.3f}")
        print(f"    Difference: {np.mean(first_half) - np.mean(second_half):.3f}")


def main():
    env_name = 'PointLtl2-v0'
    exp = 'planning_from_baseline'

    print("="*60)
    print("INVESTIGATING DIRECTIONAL BIAS")
    print("="*60)

    # Create environment
    def optimality_sampler(props):
        return lambda: create_optimality_task(props)

    base_env = safety_gymnasium.make('PointLtl2-v0.fixed')
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    propositions = base_env.get_propositions()
    sample_task = optimality_sampler(propositions)
    env = SequenceWrapper(base_env, sample_task)
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)

    # Load model
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())

    # Run analyses
    analyze_initial_state(env, n_resets=10)
    analyze_first_actions(env, model, props, n_episodes=10)
    test_different_orientations(env, model, props)

    env.close()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The directional bias is likely due to:
1. The agent's initial orientation in the fixed environment
2. The action space encoding (forward/turn) combined with initial heading

This means:
- The 30% "optimal" choices in the original test were NOT from planning
- They were from cases where the optimal blue happened to be
  in the direction the agent was already facing

To properly test planning, we would need to:
1. Randomize agent initial orientation
2. Or place blues at different angles that control for orientation
3. Or use a different action space (e.g., direct velocity control)
""")


if __name__ == '__main__':
    main()
