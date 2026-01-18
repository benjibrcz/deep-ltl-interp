#!/usr/bin/env python3
"""
Sequence Difficulty Test

This tests planning more directly:
- Give agent sequential goals of varying difficulty
- Measure: Does success rate / behavior depend on sequence difficulty?

If agent has world model:
  → Should succeed more on easy sequences
  → Value function should reflect anticipated difficulty
  → Behavior should adapt to sequence structure

If agent is reactive:
  → Success depends mainly on per-step navigation
  → No anticipation of future difficulty
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
    n_episodes: int = 50
    max_steps: int = 1000
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


def run_sequence(model, env, goal: LDBASequence, zone_sequence: list[str],
                 propositions: set[str], config: TestConfig) -> dict:
    """Run an episode with a sequential goal and track progress."""
    obs = env.reset()
    zone_positions = getattr(env.env, 'zone_positions', {})
    agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])

    # Compute expected path length
    expected_length = compute_sequence_path_length(agent_pos, zone_sequence, zone_positions)

    result = {
        'sequence': zone_sequence,
        'expected_length': expected_length,
        'zones_reached': [],
        'steps_per_zone': [],
        'total_steps': 0,
        'completed': False,
        'values': []  # Track value estimates
    }

    current_goal_idx = 0
    steps_since_last_zone = 0

    for step in range(config.max_steps):
        current_props = obs.get('propositions', []) if isinstance(obs, dict) else []

        # Get remaining goal
        remaining_goal = list(goal)[current_goal_idx:]
        if not remaining_goal:
            result['completed'] = True
            break

        obs_dict = {
            'features': obs['features'] if isinstance(obs, dict) else obs,
            'goal': remaining_goal,
            'propositions': current_props
        }

        preprocessed = preprocessing.preprocess_obss([obs_dict], propositions)
        with torch.no_grad():
            dist, value = model(preprocessed)
            action = dist.mode.numpy().flatten()
            result['values'].append(float(value.item()) if value is not None else None)

        obs, reward, done, info = env.step(action)
        steps_since_last_zone += 1
        result['total_steps'] = step + 1

        # Check if reached current target
        agent_pos = np.array(getattr(env.env, 'agent_pos', [0, 0])[:2])
        target_zone = zone_sequence[current_goal_idx]
        target_pos = find_zone_position(target_zone, zone_positions)

        if target_pos is not None:
            if np.linalg.norm(agent_pos - target_pos) < config.zone_radius:
                result['zones_reached'].append(target_zone)
                result['steps_per_zone'].append(steps_since_last_zone)
                current_goal_idx += 1
                steps_since_last_zone = 0

                if current_goal_idx >= len(zone_sequence):
                    result['completed'] = True
                    break

        if done:
            break

    return result


def run_experiment(exp_name: str = 'planning_from_baseline',
                   output_dir: str = 'interpretability/world_model_extraction/results'):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SEQUENCE DIFFICULTY TEST")
    print("=" * 70)
    print()
    print("Question: Does agent success depend on sequence difficulty?")
    print("If yes → agent may be planning ahead")
    print("If no  → agent is reactive, not planning")
    print()

    model, props = load_model(exp_name)
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    config = TestConfig(n_episodes=30)
    colors = ['blue', 'green', 'yellow', 'magenta']

    # Test different sequence lengths
    results_by_length = {}

    for seq_length in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"Testing sequences of length {seq_length}")
        print(f"{'='*70}")

        results = {
            'length': seq_length,
            'episodes': [],
            'completion_rate': 0,
            'avg_steps': 0,
            'avg_expected_length': 0
        }

        completed = 0
        total_steps = 0
        total_expected = 0

        for ep in range(config.n_episodes):
            obs = env.reset()

            # Generate random sequence
            import random
            seq = random.sample(colors, seq_length)
            goal = create_sequential_goal(seq, props)

            ep_result = run_sequence(model, env, goal, seq, props, config)
            results['episodes'].append({
                'sequence': ep_result['sequence'],
                'completed': ep_result['completed'],
                'zones_reached': len(ep_result['zones_reached']),
                'total_steps': ep_result['total_steps'],
                'expected_length': ep_result['expected_length']
            })

            if ep_result['completed']:
                completed += 1
            total_steps += ep_result['total_steps']
            total_expected += ep_result['expected_length']

        results['completion_rate'] = completed / config.n_episodes
        results['avg_steps'] = total_steps / config.n_episodes
        results['avg_expected_length'] = total_expected / config.n_episodes

        print(f"  Completion rate: {results['completion_rate']:.1%}")
        print(f"  Avg steps: {results['avg_steps']:.0f}")
        print(f"  Avg expected path length: {results['avg_expected_length']:.1f}")

        results_by_length[seq_length] = results

    # Analyze: does success correlate with difficulty?
    print("\n" + "=" * 70)
    print("ANALYSIS: Success vs Difficulty")
    print("=" * 70)

    # For length-2 sequences, split by difficulty
    if 2 in results_by_length:
        episodes = results_by_length[2]['episodes']
        median_length = np.median([e['expected_length'] for e in episodes])

        easy = [e for e in episodes if e['expected_length'] < median_length]
        hard = [e for e in episodes if e['expected_length'] >= median_length]

        easy_rate = sum(1 for e in easy if e['completed']) / len(easy) if easy else 0
        hard_rate = sum(1 for e in hard if e['completed']) / len(hard) if hard else 0

        print(f"\nLength-2 sequences split by difficulty:")
        print(f"  Easy (shorter path): {easy_rate:.1%} completion ({len(easy)} episodes)")
        print(f"  Hard (longer path):  {hard_rate:.1%} completion ({len(hard)} episodes)")

        if easy_rate > hard_rate + 0.1:
            print("\n→ Success correlates with path length")
            print("  This is expected even for reactive agents (shorter = easier)")
        else:
            print("\n→ Success does NOT strongly correlate with path length")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for length, res in results_by_length.items():
        print(f"  Length {length}: {res['completion_rate']:.1%} completion")

    length_1_rate = results_by_length.get(1, {}).get('completion_rate', 0)
    length_2_rate = results_by_length.get(2, {}).get('completion_rate', 0)
    length_3_rate = results_by_length.get(3, {}).get('completion_rate', 0)

    print(f"\nDrop-off with sequence length:")
    if length_1_rate > 0:
        print(f"  1→2: {length_1_rate:.1%} → {length_2_rate:.1%}")
        print(f"  2→3: {length_2_rate:.1%} → {length_3_rate:.1%}")

    # Save
    with open(f"{output_dir}/sequence_difficulty_{exp_name}.json", 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'episodes'}
                   for k, v in results_by_length.items()}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return results_by_length


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='planning_from_baseline')
    args = parser.parse_args()
    run_experiment(args.exp)
