#!/usr/bin/env python3
"""
Value Function Planning Test

This properly tests whether DeepLTL has learned to anticipate sequence difficulty.

Key insight: DeepLTL uses the value function to select sequences at test time.
So we can directly query V(s, seq_a) vs V(s, seq_b) to test if the agent
prefers easier sequences.

Test A: Does V(s0, easy_seq) > V(s0, hard_seq)?
Test B: Does value anticipate future difficulty within a sequence?
"""

import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
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
    n_episodes: int = 100
    max_steps: int = 500
    zone_radius: float = 0.4


def load_model(exp_name: str, seed: int = 0):
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


def create_sequential_goal(zone_sequence: list[str], propositions: set[str]) -> LDBASequence:
    sequence = []
    for zone in zone_sequence:
        reach_assignment = Assignment.single_proposition(zone, propositions).to_frozen()
        reach_set = frozenset([reach_assignment])
        avoid_set = frozenset()
        sequence.append((reach_set, avoid_set))
    return LDBASequence(sequence)


def find_zone_position(zone_name: str, zone_positions: dict) -> Optional[np.ndarray]:
    for key, pos in zone_positions.items():
        if key.startswith(zone_name):
            return np.array(pos[:2])
    return None


def compute_sequence_path_length(agent_pos: np.ndarray, zone_sequence: list[str],
                                  zone_positions: dict) -> float:
    total = 0.0
    current_pos = agent_pos.copy()
    for zone in zone_sequence:
        zone_pos = find_zone_position(zone, zone_positions)
        if zone_pos is None:
            return float('inf')
        total += np.linalg.norm(current_pos - zone_pos)
        current_pos = zone_pos
    return total


def get_value_for_sequence(model, obs, goal: LDBASequence, propositions: set[str]) -> float:
    """Query the value function for a specific sequence goal."""
    current_props = obs.get('propositions', []) if isinstance(obs, dict) else []
    obs_dict = {
        'features': obs['features'] if isinstance(obs, dict) else obs,
        'goal': list(goal),
        'propositions': current_props
    }

    preprocessed = preprocessing.preprocess_obss([obs_dict], propositions)
    with torch.no_grad():
        _, value = model(preprocessed)
        return float(value.item()) if value is not None else 0.0


def test_a_value_prefers_easier_sequence(model, env, propositions: set[str],
                                          config: TestConfig) -> dict:
    """
    Test A: Does V(s0, easy_seq) > V(s0, hard_seq)?

    For each episode:
    1. Sample two sequences: seq_a, seq_b
    2. Compute ground-truth difficulty (path length)
    3. Query V(s0, seq_a) and V(s0, seq_b)
    4. Check if value correctly predicts which is easier
    """
    results = {
        'description': 'Does value function prefer easier sequences?',
        'total_comparisons': 0,
        'value_preferred_easier': 0,
        'value_preferred_harder': 0,
        'value_equal': 0,
        'correlation_data': [],
        'value_diff_when_correct': [],
        'value_diff_when_wrong': [],
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Sample two different 2-step sequences
        import random
        shuffled = random.sample(colors, 4)
        seq_a = [shuffled[0], shuffled[1]]
        seq_b = [shuffled[2], shuffled[3]]

        # Compute ground-truth difficulty (path length)
        len_a = compute_sequence_path_length(agent_pos, seq_a, zone_positions)
        len_b = compute_sequence_path_length(agent_pos, seq_b, zone_positions)

        if len_a == float('inf') or len_b == float('inf'):
            continue

        # Create goals
        goal_a = create_sequential_goal(seq_a, propositions)
        goal_b = create_sequential_goal(seq_b, propositions)

        # Query value function
        v_a = get_value_for_sequence(model, obs, goal_a, propositions)
        v_b = get_value_for_sequence(model, obs, goal_b, propositions)

        # Compare
        easier_seq = 'a' if len_a < len_b else 'b'
        value_prefers = 'a' if v_a > v_b else ('b' if v_b > v_a else 'equal')

        results['total_comparisons'] += 1
        results['correlation_data'].append({
            'len_a': len_a, 'len_b': len_b,
            'v_a': v_a, 'v_b': v_b,
            'easier': easier_seq,
            'value_prefers': value_prefers
        })

        if value_prefers == 'equal':
            results['value_equal'] += 1
        elif value_prefers == easier_seq:
            results['value_preferred_easier'] += 1
            results['value_diff_when_correct'].append(abs(v_a - v_b))
        else:
            results['value_preferred_harder'] += 1
            results['value_diff_when_wrong'].append(abs(v_a - v_b))

    # Compute summary stats
    total = results['total_comparisons']
    if total > 0:
        results['easier_preference_rate'] = results['value_preferred_easier'] / total
        results['harder_preference_rate'] = results['value_preferred_harder'] / total

    # Compute correlation between path length difference and value difference
    if results['correlation_data']:
        len_diffs = [d['len_a'] - d['len_b'] for d in results['correlation_data']]
        val_diffs = [d['v_a'] - d['v_b'] for d in results['correlation_data']]

        # Correlation: negative correlation means shorter path → higher value (correct)
        if len(len_diffs) > 1:
            correlation = np.corrcoef(len_diffs, val_diffs)[0, 1]
            results['length_value_correlation'] = correlation

    return results


def test_b_value_anticipation(model, env, propositions: set[str],
                               config: TestConfig) -> dict:
    """
    Test B: Does value anticipate future difficulty?

    For length-2 sequences with the SAME first target, compare:
    - Initial value for easy second-step vs hard second-step

    If agent anticipates, value should differ at start.
    If reactive, value separation only appears after first goal.
    """
    results = {
        'description': 'Does value anticipate future difficulty?',
        'comparisons': [],
        'initial_value_tracks_difficulty': 0,
        'total_valid': 0,
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Pick a first target, then two different second targets
        import random
        first_target = random.choice(colors)
        remaining = [c for c in colors if c != first_target]
        second_targets = random.sample(remaining, 2)

        seq_easy_candidate = [first_target, second_targets[0]]
        seq_hard_candidate = [first_target, second_targets[1]]

        # Compute which is actually easier (from first target to second)
        first_pos = find_zone_position(first_target, zone_positions)
        if first_pos is None:
            continue

        second_pos_0 = find_zone_position(second_targets[0], zone_positions)
        second_pos_1 = find_zone_position(second_targets[1], zone_positions)

        if second_pos_0 is None or second_pos_1 is None:
            continue

        # Distance from first target to each second target
        dist_to_0 = np.linalg.norm(first_pos - second_pos_0)
        dist_to_1 = np.linalg.norm(first_pos - second_pos_1)

        if dist_to_0 < dist_to_1:
            seq_easy = seq_easy_candidate
            seq_hard = seq_hard_candidate
            dist_easy = dist_to_0
            dist_hard = dist_to_1
        else:
            seq_easy = seq_hard_candidate
            seq_hard = seq_easy_candidate
            dist_easy = dist_to_1
            dist_hard = dist_to_0

        # Create goals
        goal_easy = create_sequential_goal(seq_easy, propositions)
        goal_hard = create_sequential_goal(seq_hard, propositions)

        # Query initial values
        v_easy = get_value_for_sequence(model, obs, goal_easy, propositions)
        v_hard = get_value_for_sequence(model, obs, goal_hard, propositions)

        results['total_valid'] += 1

        comparison = {
            'first_target': first_target,
            'seq_easy': seq_easy,
            'seq_hard': seq_hard,
            'dist_easy': dist_easy,
            'dist_hard': dist_hard,
            'v_easy': v_easy,
            'v_hard': v_hard,
            'value_prefers_easy': v_easy > v_hard
        }
        results['comparisons'].append(comparison)

        if v_easy > v_hard:
            results['initial_value_tracks_difficulty'] += 1

    if results['total_valid'] > 0:
        results['anticipation_rate'] = results['initial_value_tracks_difficulty'] / results['total_valid']

    return results


def test_c_suffix_marginal_value(model, env, propositions: set[str],
                                  config: TestConfig) -> dict:
    """
    Test C: Suffix Marginal Value Test

    For a fixed start state and fixed first target A, compute:
        ΔV_C = V(s0, [A, C]) - V(s0, [A])

    Compare ΔV across "easy C" vs "hard C".

    If the value head genuinely anticipates the second step, ΔV should be
    higher for easy C than hard C, even when absolute V is dominated by
    the first step.
    """
    results = {
        'description': 'Suffix marginal value: does adding an easy vs hard second step change value differently?',
        'comparisons': [],
        'delta_v_higher_for_easy': 0,
        'delta_v_higher_for_hard': 0,
        'total_valid': 0,
    }

    colors = ['blue', 'green', 'yellow', 'magenta']

    for ep in range(config.n_episodes):
        obs = env.reset()
        zone_positions = getattr(env.env, 'zone_positions', {})
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

        # Pick a first target, then two different second targets
        import random
        first_target = random.choice(colors)
        remaining = [c for c in colors if c != first_target]
        second_targets = random.sample(remaining, 2)

        # Compute which second target is easier (closer to first target)
        first_pos = find_zone_position(first_target, zone_positions)
        if first_pos is None:
            continue

        second_pos_0 = find_zone_position(second_targets[0], zone_positions)
        second_pos_1 = find_zone_position(second_targets[1], zone_positions)

        if second_pos_0 is None or second_pos_1 is None:
            continue

        dist_to_0 = np.linalg.norm(first_pos - second_pos_0)
        dist_to_1 = np.linalg.norm(first_pos - second_pos_1)

        if dist_to_0 < dist_to_1:
            easy_second = second_targets[0]
            hard_second = second_targets[1]
        else:
            easy_second = second_targets[1]
            hard_second = second_targets[0]

        # Create goals
        goal_first_only = create_sequential_goal([first_target], propositions)
        goal_with_easy = create_sequential_goal([first_target, easy_second], propositions)
        goal_with_hard = create_sequential_goal([first_target, hard_second], propositions)

        # Query values
        v_first_only = get_value_for_sequence(model, obs, goal_first_only, propositions)
        v_with_easy = get_value_for_sequence(model, obs, goal_with_easy, propositions)
        v_with_hard = get_value_for_sequence(model, obs, goal_with_hard, propositions)

        # Compute marginal values
        delta_v_easy = v_with_easy - v_first_only
        delta_v_hard = v_with_hard - v_first_only

        results['total_valid'] += 1

        comparison = {
            'first_target': first_target,
            'easy_second': easy_second,
            'hard_second': hard_second,
            'v_first_only': v_first_only,
            'v_with_easy': v_with_easy,
            'v_with_hard': v_with_hard,
            'delta_v_easy': delta_v_easy,
            'delta_v_hard': delta_v_hard,
        }
        results['comparisons'].append(comparison)

        # If value anticipates, ΔV for easy should be higher (less negative or more positive)
        # because adding an easy second step should hurt value less than adding a hard one
        if delta_v_easy > delta_v_hard:
            results['delta_v_higher_for_easy'] += 1
        else:
            results['delta_v_higher_for_hard'] += 1

    if results['total_valid'] > 0:
        results['anticipation_rate'] = results['delta_v_higher_for_easy'] / results['total_valid']

        # Also compute mean delta_v for easy vs hard
        delta_easy = [c['delta_v_easy'] for c in results['comparisons']]
        delta_hard = [c['delta_v_hard'] for c in results['comparisons']]
        results['mean_delta_v_easy'] = np.mean(delta_easy)
        results['mean_delta_v_hard'] = np.mean(delta_hard)
        results['delta_v_difference'] = results['mean_delta_v_easy'] - results['mean_delta_v_hard']

    return results


def run_experiment(exp_name: str = 'planning_from_baseline',
                   output_dir: str = 'interpretability/world_model_extraction/results'):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("VALUE FUNCTION PLANNING TEST")
    print("=" * 70)
    print()
    print("This tests whether DeepLTL's value function anticipates sequence difficulty.")
    print("DeepLTL uses V(s, seq) to select sequences at test time, so this directly")
    print("tests whether the agent has learned something world-model-ish.")
    print()

    model, props = load_model(exp_name)
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    config = TestConfig(n_episodes=200)

    # Test A
    print("=" * 70)
    print("TEST A: Does V(s0, easy_seq) > V(s0, hard_seq)?")
    print("=" * 70)

    results_a = test_a_value_prefers_easier_sequence(model, env, props, config)

    print(f"\nResults ({results_a['total_comparisons']} comparisons):")
    print(f"  Value preferred easier sequence: {results_a['value_preferred_easier']} ({results_a.get('easier_preference_rate', 0):.1%})")
    print(f"  Value preferred harder sequence: {results_a['value_preferred_harder']} ({results_a.get('harder_preference_rate', 0):.1%})")
    print(f"  Value equal: {results_a['value_equal']}")

    if 'length_value_correlation' in results_a:
        corr = results_a['length_value_correlation']
        print(f"\n  Correlation (path_length_diff vs value_diff): {corr:.3f}")
        print(f"  (Negative = shorter path → higher value = CORRECT)")

        if corr < -0.2:
            print("  → Value function DOES track sequence difficulty")
        elif corr > 0.2:
            print("  → Value function is INVERSELY related to difficulty (unexpected)")
        else:
            print("  → Value function shows WEAK/NO relationship to difficulty")

    # Test B
    print("\n" + "=" * 70)
    print("TEST B: Does value anticipate future difficulty?")
    print("(Same first target, different second targets)")
    print("=" * 70)

    results_b = test_b_value_anticipation(model, env, props, config)

    print(f"\nResults ({results_b['total_valid']} valid comparisons):")
    print(f"  Initial value higher for easier second-step: {results_b['initial_value_tracks_difficulty']} ({results_b.get('anticipation_rate', 0):.1%})")

    if results_b.get('anticipation_rate', 0) > 0.6:
        print("  → Value function ANTICIPATES future difficulty")
    elif results_b.get('anticipation_rate', 0) > 0.4:
        print("  → Weak anticipation (near random)")
    else:
        print("  → Value function does NOT anticipate future difficulty")

    # Test C
    print("\n" + "=" * 70)
    print("TEST C: Suffix Marginal Value")
    print("ΔV = V(s, [A,C]) - V(s, [A])")
    print("Does ΔV differ for easy C vs hard C?")
    print("=" * 70)

    results_c = test_c_suffix_marginal_value(model, env, props, config)

    print(f"\nResults ({results_c['total_valid']} valid comparisons):")
    print(f"  ΔV higher for easy second-step: {results_c['delta_v_higher_for_easy']} ({results_c.get('anticipation_rate', 0):.1%})")
    print(f"  ΔV higher for hard second-step: {results_c['delta_v_higher_for_hard']}")

    if 'mean_delta_v_easy' in results_c:
        print(f"\n  Mean ΔV (easy second): {results_c['mean_delta_v_easy']:.4f}")
        print(f"  Mean ΔV (hard second): {results_c['mean_delta_v_hard']:.4f}")
        print(f"  Difference: {results_c['delta_v_difference']:.4f}")

    if results_c.get('anticipation_rate', 0) > 0.6:
        print("\n  → Value function DOES anticipate second-step difficulty")
    elif results_c.get('anticipation_rate', 0) > 0.4:
        print("\n  → Marginal value shows WEAK/NO discrimination")
    else:
        print("\n  → Marginal value is INVERTED (unexpected)")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    pref_rate = results_a.get('easier_preference_rate', 0.5)
    antic_rate_b = results_b.get('anticipation_rate', 0.5)
    antic_rate_c = results_c.get('anticipation_rate', 0.5)

    print(f"\nTest A (value prefers easier full sequence): {pref_rate:.1%}")
    print(f"Test B (value anticipates, same first target): {antic_rate_b:.1%}")
    print(f"Test C (marginal value ΔV discriminates):      {antic_rate_c:.1%}")

    if antic_rate_c > 0.6:
        print("\n→ EVIDENCE OF LOOKAHEAD: Marginal value tracks second-step difficulty")
    elif pref_rate > 0.55 and antic_rate_c < 0.55:
        print("\n→ FIRST-STEP DOMINATED: Value tracks full sequence weakly,")
        print("  but marginal value doesn't discriminate second step.")
        print("  Suggests value is mostly reflecting first-step difficulty.")
    else:
        print("\n→ WEAK/LIMITED LOOKAHEAD: Some sensitivity to sequence difficulty,")
        print("  but not clearly anticipating future steps.")

    # Save results
    save_results = {
        'test_a': {k: v for k, v in results_a.items() if k != 'correlation_data'},
        'test_b': {k: v for k, v in results_b.items() if k != 'comparisons'},
        'test_c': {k: v for k, v in results_c.items() if k != 'comparisons'},
    }

    with open(f"{output_dir}/value_planning_test_{exp_name}.json", 'w') as f:
        json.dump(save_results, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}/")

    return results_a, results_b, results_c


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='planning_from_baseline')
    args = parser.parse_args()
    run_experiment(args.exp)
