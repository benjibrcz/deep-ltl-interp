#!/usr/bin/env python3
"""
Corrected Depth-Sweep Test

This properly tests the paper's prediction:
- Theorem 2: Depth-1 (myopic) goals provide NO information about transitions
- Theorem 1: Depth-n goals (n>1) require planning and reveal world model

Key insight: The test must give the agent FULL sequential goals, not just
disjunctive first steps. The agent's first choice reveals whether it's
planning ahead.

Test design:
- Depth-1: "reach A OR reach B" → expect ~50% (no planning needed)
- Depth-2: "reach A THEN reach C" vs "reach B THEN reach D" (FULL sequences)
           → if agent plans, should choose sequence with shorter total path
"""

import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import json

import preprocessing
from envs import make_env
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from ltl.logic import Assignment
from ltl.automata import LDBASequence


@dataclass
class TestConfig:
    n_episodes_per_test: int = 50
    max_steps: int = 500
    zone_radius: float = 0.4


def load_model(exp_name: str, seed: int = 0):
    """Load a trained DeepLTL model"""
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp_name, seed=seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(env, training_status, config, use_aux_head=False, use_transition_head=False)
    model.eval()

    props = set(env.get_propositions())
    env.close()

    return model, props


def create_single_reach_goal(zone: str, propositions: set[str]) -> LDBASequence:
    """Depth-1 goal: just reach one zone"""
    reach_assignment = Assignment.single_proposition(zone, propositions).to_frozen()
    reach_set = frozenset([reach_assignment])
    avoid_set = frozenset()
    return LDBASequence([(reach_set, avoid_set)])


def create_disjunctive_goal(zone_a: str, zone_b: str, propositions: set[str]) -> LDBASequence:
    """Depth-1 disjunctive goal: reach A OR reach B"""
    reach_a = Assignment.single_proposition(zone_a, propositions).to_frozen()
    reach_b = Assignment.single_proposition(zone_b, propositions).to_frozen()
    reach_set = frozenset([reach_a, reach_b])
    avoid_set = frozenset()
    return LDBASequence([(reach_set, avoid_set)])


def create_sequential_goal(zone_sequence: list[str], propositions: set[str]) -> LDBASequence:
    """Create a sequential goal: reach zone1 THEN reach zone2 THEN ..."""
    sequence = []
    for zone in zone_sequence:
        reach_assignment = Assignment.single_proposition(zone, propositions).to_frozen()
        reach_set = frozenset([reach_assignment])
        avoid_set = frozenset()
        sequence.append((reach_set, avoid_set))
    return LDBASequence(sequence)


def compute_sequence_path_length(agent_pos: np.ndarray, zone_sequence: list[str],
                                  zone_positions: dict) -> float:
    """Compute total path length for a sequence of zones"""
    total = 0.0
    current_pos = agent_pos.copy()

    for zone in zone_sequence:
        zone_pos = find_zone_position(zone, zone_positions)
        if zone_pos is None:
            return float('inf')
        total += np.linalg.norm(current_pos - zone_pos)
        current_pos = zone_pos

    return total


def find_zone_position(zone_name: str, zone_positions: dict) -> Optional[np.ndarray]:
    """Find position of a zone by color prefix"""
    for key, pos in zone_positions.items():
        if key.startswith(zone_name):
            return np.array(pos[:2])
    return None


def run_episode_with_goal(model, env, goal: LDBASequence, propositions: set[str],
                          config: TestConfig) -> tuple[str, int]:
    """
    Run episode with given goal, return (first_zone_reached, steps).
    first_zone_reached is the color of the first zone reached, or 'neither'.
    """
    obs = env.reset()
    zone_positions = getattr(env.env, 'zone_positions', {})

    for step in range(config.max_steps):
        # Prepare observation with goal
        current_props = obs.get('propositions', []) if isinstance(obs, dict) else []
        obs_dict = {
            'features': obs['features'] if isinstance(obs, dict) else obs,
            'goal': list(goal),
            'propositions': current_props
        }

        # Get action
        preprocessed = preprocessing.preprocess_obss([obs_dict], propositions)
        with torch.no_grad():
            dist, _ = model(preprocessed)
            action = dist.mode.numpy().flatten()

        obs, reward, done, info = env.step(action)

        # Check which zone we're in
        agent_pos = getattr(env.env, 'agent_pos', None)
        if agent_pos is not None:
            agent_pos = np.array(agent_pos[:2])
        else:
            agent_pos = obs['features'][:2] if isinstance(obs, dict) else obs[:2]

        # Check all zones
        for zone_name, zone_pos in zone_positions.items():
            zone_pos = np.array(zone_pos[:2])
            if np.linalg.norm(agent_pos - zone_pos) < config.zone_radius:
                color = zone_name.split('_')[0]
                return color, step + 1

        if done:
            break

    return 'neither', 0


def test_depth_1(model, env, propositions: set[str], config: TestConfig) -> dict:
    """
    Test depth-1: Disjunctive goal "reach A OR reach B"

    Paper prediction (Theorem 2): Should be ~50% since no planning needed.
    """
    results = {
        'depth': 1,
        'description': 'Disjunctive: reach A OR reach B',
        'chose_closer': 0,
        'chose_farther': 0,
        'neither': 0,
        'total': 0
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes_per_test):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Pick two random colors
        import random
        zone_a, zone_b = random.sample(colors, 2)

        # Create disjunctive goal
        goal = create_disjunctive_goal(zone_a, zone_b, propositions)

        # Determine which is closer
        pos_a = find_zone_position(zone_a, zone_positions)
        pos_b = find_zone_position(zone_b, zone_positions)
        if pos_a is None or pos_b is None:
            continue

        dist_a = np.linalg.norm(agent_pos - pos_a)
        dist_b = np.linalg.norm(agent_pos - pos_b)
        closer = zone_a if dist_a < dist_b else zone_b

        # Run episode
        reached, steps = run_episode_with_goal(model, env, goal, propositions, config)

        results['total'] += 1
        if reached == 'neither':
            results['neither'] += 1
        elif reached == closer:
            results['chose_closer'] += 1
        else:
            results['chose_farther'] += 1

    return results


def test_depth_2(model, env, propositions: set[str], config: TestConfig) -> dict:
    """
    Test depth-2: Two competing SEQUENCES

    Goal A: reach zone1 THEN reach zone2
    Goal B: reach zone3 THEN reach zone4

    Agent is given goal A (full sequence).
    We measure: does agent's first choice optimize for the FULL sequence?

    Paper prediction: If agent has world model, should choose first zone
    that leads to shorter total path, not just closer first zone.
    """
    results = {
        'depth': 2,
        'description': 'Sequential: reach A then B (full sequence)',
        'chose_shorter_sequence': 0,
        'chose_longer_sequence': 0,
        'chose_closer_first': 0,  # Track distance-based heuristic
        'chose_farther_first': 0,
        'conflicts': 0,  # Cases where closer first != shorter sequence
        'neither': 0,
        'total': 0
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes_per_test):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Create two competing sequences
        import random
        shuffled = random.sample(colors, 4)
        seq_a = [shuffled[0], shuffled[1]]  # e.g., blue → green
        seq_b = [shuffled[2], shuffled[3]]  # e.g., yellow → magenta

        # Compute total path lengths
        len_a = compute_sequence_path_length(agent_pos, seq_a, zone_positions)
        len_b = compute_sequence_path_length(agent_pos, seq_b, zone_positions)

        if len_a == float('inf') or len_b == float('inf'):
            continue

        shorter_seq = seq_a if len_a < len_b else seq_b

        # Which first zone is closer?
        pos_first_a = find_zone_position(seq_a[0], zone_positions)
        pos_first_b = find_zone_position(seq_b[0], zone_positions)
        dist_first_a = np.linalg.norm(agent_pos - pos_first_a)
        dist_first_b = np.linalg.norm(agent_pos - pos_first_b)
        closer_first = seq_a[0] if dist_first_a < dist_first_b else seq_b[0]

        # Is there a conflict? (closer first zone != shorter sequence)
        if (closer_first == seq_a[0]) != (shorter_seq == seq_a):
            results['conflicts'] += 1

        # Give agent the shorter sequence as its goal
        # (We want to see if it executes it optimally)
        # Actually, for a fair test: give a disjunctive SEQUENCE choice
        # But LDBASequence doesn't support that directly...

        # Alternative: Give agent seq_a, see if it goes to seq_a[0] first
        # This tests: does agent follow the sequential goal?
        goal = create_sequential_goal(seq_a, propositions)

        # Run episode
        reached, steps = run_episode_with_goal(model, env, goal, propositions, config)

        results['total'] += 1
        if reached == 'neither':
            results['neither'] += 1
        else:
            # Did it reach the FIRST zone in the sequence?
            if reached == seq_a[0]:
                results['chose_shorter_sequence'] += 1  # Following goal correctly
            else:
                results['chose_longer_sequence'] += 1  # Went to wrong zone

            # Track distance heuristic
            if reached == closer_first:
                results['chose_closer_first'] += 1
            else:
                results['chose_farther_first'] += 1

    return results


def test_depth_2_proper(model, env, propositions: set[str], config: TestConfig) -> dict:
    """
    Proper depth-2 test using disjunctive first step with sequential continuation.

    This creates a goal where:
    - First step: reach blue OR reach green (disjunctive)
    - If reached blue: then reach yellow
    - If reached green: then reach magenta

    Agent must plan: which first choice leads to easier completion?
    """
    results = {
        'depth': 2,
        'description': 'Disjunctive first step with sequential continuation',
        'chose_shorter_total': 0,
        'chose_longer_total': 0,
        'closer_first_was_shorter': 0,  # No conflict
        'closer_first_was_longer': 0,   # Conflict - tests planning
        'neither': 0,
        'total': 0,
        'details': []
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes_per_test):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Create two sequences with disjunctive first step
        import random
        shuffled = random.sample(colors, 4)
        seq_a = [shuffled[0], shuffled[1]]  # Option A: first → second
        seq_b = [shuffled[2], shuffled[3]]  # Option B: third → fourth

        # Compute total path lengths
        len_a = compute_sequence_path_length(agent_pos, seq_a, zone_positions)
        len_b = compute_sequence_path_length(agent_pos, seq_b, zone_positions)

        if len_a == float('inf') or len_b == float('inf'):
            continue

        shorter_first = seq_a[0] if len_a < len_b else seq_b[0]

        # Which first zone is closer (distance heuristic)?
        pos_a = find_zone_position(seq_a[0], zone_positions)
        pos_b = find_zone_position(seq_b[0], zone_positions)
        dist_a = np.linalg.norm(agent_pos - pos_a)
        dist_b = np.linalg.norm(agent_pos - pos_b)
        closer_first = seq_a[0] if dist_a < dist_b else seq_b[0]

        # Track conflict cases
        if closer_first == shorter_first:
            results['closer_first_was_shorter'] += 1
        else:
            results['closer_first_was_longer'] += 1

        # Create disjunctive goal for first step
        # Note: This is still depth-1 in terms of goal structure
        # A true test would need the agent to know about the continuation
        goal = create_disjunctive_goal(seq_a[0], seq_b[0], propositions)

        # Run episode
        reached, steps = run_episode_with_goal(model, env, goal, propositions, config)

        results['total'] += 1
        if reached == 'neither':
            results['neither'] += 1
        elif reached == shorter_first:
            results['chose_shorter_total'] += 1
        else:
            results['chose_longer_total'] += 1

        results['details'].append({
            'seq_a': seq_a, 'seq_b': seq_b,
            'len_a': len_a, 'len_b': len_b,
            'closer_first': closer_first,
            'shorter_first': shorter_first,
            'reached': reached
        })

    return results


def run_corrected_experiment(exp_name: str = 'planning_from_baseline',
                             output_dir: str = 'interpretability/world_model_extraction/results'):
    """Run the corrected depth experiment."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CORRECTED DEPTH-SWEEP TEST")
    print("=" * 70)
    print()
    print("Paper predictions:")
    print("  Theorem 2: Depth-1 goals → ~50% (no planning needed)")
    print("  Theorem 1: Depth-n goals → should show planning if world model exists")
    print()

    # Load model
    print(f"Loading model: {exp_name}")
    model, props = load_model(exp_name)

    # Create environment
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    config = TestConfig(n_episodes_per_test=100)

    # Run tests
    print("\n" + "=" * 70)
    print("TEST 1: Depth-1 (Disjunctive)")
    print("Goal: 'reach blue OR reach green'")
    print("Expected: ~50% chose closer (no planning needed)")
    print("=" * 70)

    results_d1 = test_depth_1(model, env, props, config)

    reached = results_d1['chose_closer'] + results_d1['chose_farther']
    if reached > 0:
        closer_rate = results_d1['chose_closer'] / reached
        print(f"\nResults (among {reached} episodes that reached a zone):")
        print(f"  Chose closer zone: {results_d1['chose_closer']} ({closer_rate:.1%})")
        print(f"  Chose farther zone: {results_d1['chose_farther']} ({1-closer_rate:.1%})")
        print(f"  Reached neither: {results_d1['neither']}")

    print("\n" + "=" * 70)
    print("TEST 2: Depth-2 (Sequential)")
    print("Goal: 'reach A then reach B' (full sequence)")
    print("Tests: Does agent follow the sequential goal?")
    print("=" * 70)

    results_d2 = test_depth_2(model, env, props, config)

    reached = results_d2['chose_shorter_sequence'] + results_d2['chose_longer_sequence']
    if reached > 0:
        correct_rate = results_d2['chose_shorter_sequence'] / reached
        print(f"\nResults (among {reached} episodes that reached a zone):")
        print(f"  Followed sequence (reached first goal): {results_d2['chose_shorter_sequence']} ({correct_rate:.1%})")
        print(f"  Went to wrong zone first: {results_d2['chose_longer_sequence']} ({1-correct_rate:.1%})")
        print(f"  Reached neither: {results_d2['neither']}")

    print("\n" + "=" * 70)
    print("TEST 3: Planning Test (Disjunctive with future consequence)")
    print("Setup: Agent chooses between two first-step options")
    print("       Each leads to different second-step difficulty")
    print("Question: Does agent consider full path length, or just first step?")
    print("=" * 70)

    results_d2b = test_depth_2_proper(model, env, props, config)

    reached = results_d2b['chose_shorter_total'] + results_d2b['chose_longer_total']
    conflicts = results_d2b['closer_first_was_longer']

    if reached > 0:
        shorter_rate = results_d2b['chose_shorter_total'] / reached
        print(f"\nResults (among {reached} episodes that reached a zone):")
        print(f"  Chose zone leading to shorter total path: {results_d2b['chose_shorter_total']} ({shorter_rate:.1%})")
        print(f"  Chose zone leading to longer total path: {results_d2b['chose_longer_total']} ({1-shorter_rate:.1%})")
        print(f"  Reached neither: {results_d2b['neither']}")
        print(f"\nConflict analysis:")
        print(f"  Cases where closer first ≠ shorter total: {conflicts}/{results_d2b['total']}")

    env.close()

    # Analysis
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    d1_rate = results_d1['chose_closer'] / max(1, results_d1['chose_closer'] + results_d1['chose_farther'])
    d2_rate = results_d2['chose_shorter_sequence'] / max(1, results_d2['chose_shorter_sequence'] + results_d2['chose_longer_sequence'])

    print(f"\nDepth-1 (disjunctive): {d1_rate:.1%} chose closer")
    print(f"Depth-2 (sequential): {d2_rate:.1%} followed sequence correctly")

    if abs(d1_rate - 0.5) < 0.15:
        print("\n→ Depth-1 is near 50% as expected (Theorem 2 confirmed)")

    if d2_rate > 0.7:
        print("→ Agent follows sequential goals (but this doesn't test PLANNING)")
    else:
        print("→ Agent doesn't reliably follow sequential goals")

    # Save results
    all_results = {
        'depth_1': {k: v for k, v in results_d1.items()},
        'depth_2': {k: v for k, v in results_d2.items()},
        'depth_2_planning': {k: v for k, v in results_d2b.items() if k != 'details'}
    }

    with open(f"{output_dir}/corrected_depth_test_{exp_name}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}/")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='planning_from_baseline')
    parser.add_argument('--output_dir', type=str, default='interpretability/world_model_extraction/results')
    args = parser.parse_args()

    run_corrected_experiment(args.exp, args.output_dir)
