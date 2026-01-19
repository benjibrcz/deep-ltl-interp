#!/usr/bin/env python3
"""
Empirical Difficulty Analysis

Instead of using geometric path length as a proxy for difficulty,
this script measures ACTUAL difficulty via rollouts:
- Completion rate
- Mean steps to completion

This makes the optimality and value test conclusions much stronger.
"""

import os
import sys
sys.path.insert(0, 'src')

import json
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import preprocessing
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from ltl.automata import LDBASequence
from ltl.logic import Assignment
from envs import make_env

# Direct imports for creating custom env
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from safety_gymnasium.utils.registration import make


COLORS = ['blue', 'green', 'yellow', 'magenta']
COLOR_PAIRS = [(a, b) for a in COLORS for b in COLORS if a != b]


class Agent:
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
        return action.detach().numpy().flatten(), value.item()


def get_unwrapped(env):
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def load_model(exp_name, env_name='PointLtl2-v0'):
    """Load a trained model."""
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp_name, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(temp_env, training_status, config)
    model.eval()
    props = list(temp_env.get_propositions())
    temp_env.close()

    return model, props


def create_optvar_env(int_color, goal_color, layout_seed, propositions):
    """Create optvar environment."""
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    env = make('PointLtl2-v0.optvar', config=config)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)

    int_reach = frozenset([Assignment.single_proposition(int_color, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_color, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=500)
    env = RemoveTruncWrapper(env)

    return env


def create_opteq_env(int_color, goal_color, layout_seed, propositions):
    """Create equidistant environment."""
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    env = make('PointLtl2-v0.opteq', config=config)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)

    int_reach = frozenset([Assignment.single_proposition(int_color, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_color, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=500)
    env = RemoveTruncWrapper(env)

    return env


def measure_empirical_difficulty(env, agent, int_color, goal_color, n_rollouts=15, max_steps=500):
    """
    Measure empirical difficulty for each intermediate zone option.

    Returns dict with:
    - completion_rate per intermediate
    - mean_steps per intermediate
    - which intermediate is empirically easier
    """
    unwrapped = get_unwrapped(env)

    # Reset to get zone positions
    env.reset()
    zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}
    agent_start = unwrapped.agent_pos[:2].copy()

    # Find intermediate zones
    int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
    goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]

    if len(int_zones) < 2:
        return None

    # For each intermediate, measure empirical difficulty
    results = {}

    for int_name, int_pos in int_zones:
        completions = 0
        total_steps = []

        for _ in range(n_rollouts):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            steps = 0
            reached_int = False
            reached_goal = False

            for step in range(max_steps):
                action, _ = agent.get_action(obs, deterministic=True)
                if len(action) == 1:
                    action = action[0]

                obs, reward, done, info = env.step(action)
                steps += 1

                # Check if we reached this specific intermediate first
                if not reached_int:
                    pos = unwrapped.agent_pos[:2]
                    if np.linalg.norm(pos - int_pos) < 0.4:
                        reached_int = True
                    # Check if we reached the OTHER intermediate first (wrong choice)
                    for other_name, other_pos in int_zones:
                        if other_name != int_name:
                            if np.linalg.norm(pos - other_pos) < 0.4:
                                # Went to wrong intermediate - count as failure for this option
                                break
                    else:
                        continue
                    break

                if done:
                    reached_goal = info.get('success', False)
                    break

            if reached_int and reached_goal:
                completions += 1
                total_steps.append(steps)
            elif reached_int:
                # Reached intermediate but not goal
                total_steps.append(steps)

        results[int_name] = {
            'completion_rate': completions / n_rollouts,
            'mean_steps': np.mean(total_steps) if total_steps else max_steps,
            'std_steps': np.std(total_steps) if len(total_steps) > 1 else 0,
            'n_completions': completions,
            'position': int_pos.tolist(),
        }

    # Determine which is empirically easier (higher completion rate, or if tied, fewer steps)
    int_names = list(results.keys())
    if len(int_names) >= 2:
        r0, r1 = results[int_names[0]], results[int_names[1]]

        # Primary: completion rate. Secondary: mean steps
        if r0['completion_rate'] > r1['completion_rate']:
            empirically_easier = int_names[0]
        elif r1['completion_rate'] > r0['completion_rate']:
            empirically_easier = int_names[1]
        elif r0['mean_steps'] < r1['mean_steps']:
            empirically_easier = int_names[0]
        else:
            empirically_easier = int_names[1]

        results['empirically_easier'] = empirically_easier
        results['difficulty_diff'] = abs(r0['completion_rate'] - r1['completion_rate'])
        results['steps_diff'] = abs(r0['mean_steps'] - r1['mean_steps'])

    # Also compute geometric difficulty for comparison
    for int_name, int_pos in int_zones:
        d_agent = np.linalg.norm(int_pos - agent_start)
        d_goal = min(np.linalg.norm(gpos - int_pos) for _, gpos in goal_zones)
        results[int_name]['geometric_total'] = d_agent + d_goal
        results[int_name]['d_agent'] = d_agent
        results[int_name]['d_goal'] = d_goal

    # Geometric easier
    geom_totals = [(name, results[name]['geometric_total']) for name in int_names]
    results['geometrically_easier'] = min(geom_totals, key=lambda x: x[1])[0]

    return results


def run_optimality_episode(env, agent, int_color, max_steps=500):
    """Run episode and see which intermediate the agent chooses."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    unwrapped = get_unwrapped(env)
    zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

    int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]

    chosen_int = None
    for step in range(max_steps):
        action, value = agent.get_action(obs, deterministic=True)
        if len(action) == 1:
            action = action[0]

        obs, reward, done, info = env.step(action)

        if chosen_int is None:
            pos = unwrapped.agent_pos[:2]
            for name, int_pos in int_zones:
                if np.linalg.norm(pos - int_pos) < 0.4:
                    chosen_int = name
                    break

        if done:
            break

    return {
        'chosen_int': chosen_int,
        'success': info.get('success', False),
        'steps': step + 1,
    }


def run_empirical_optimality_test(agent_obj, propositions, n_configs=50, n_difficulty_rollouts=15, seed=42):
    """
    Main optimality test using empirical difficulty labels.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []

    print(f"\nRunning empirical optimality test ({n_configs} configurations)...")
    print(f"Each config: {n_difficulty_rollouts} rollouts to measure difficulty\n")

    for i in tqdm(range(n_configs)):
        int_color, goal_color = random.choice(COLOR_PAIRS)
        layout_seed = seed + i * 7

        try:
            # Create environment
            env = create_optvar_env(int_color, goal_color, layout_seed, propositions)

            # Measure empirical difficulty
            difficulty = measure_empirical_difficulty(
                env, agent_obj, int_color, goal_color,
                n_rollouts=n_difficulty_rollouts
            )

            if difficulty is None:
                env.close()
                continue

            # Now run the actual test episode
            episode = run_optimality_episode(env, agent_obj, int_color)
            env.close()

            # Determine if agent chose empirically easier option
            if episode['chosen_int'] is not None:
                chose_empirically_easier = (episode['chosen_int'] == difficulty['empirically_easier'])
                chose_geometrically_easier = (episode['chosen_int'] == difficulty['geometrically_easier'])
            else:
                chose_empirically_easier = None
                chose_geometrically_easier = None

            results.append({
                'config_id': i,
                'int_color': int_color,
                'goal_color': goal_color,
                'layout_seed': layout_seed,
                'difficulty': difficulty,
                'episode': episode,
                'chose_empirically_easier': chose_empirically_easier,
                'chose_geometrically_easier': chose_geometrically_easier,
                'empirical_geo_agree': difficulty['empirically_easier'] == difficulty['geometrically_easier'],
            })

        except Exception as e:
            print(f"\nError in config {i}: {e}")
            continue

    return results


def run_empirical_equidistant_test(agent_obj, propositions, n_configs=50, n_difficulty_rollouts=15, seed=42):
    """
    Equidistant test using empirical difficulty labels.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []

    print(f"\nRunning empirical equidistant test ({n_configs} configurations)...")

    for i in tqdm(range(n_configs)):
        int_color, goal_color = random.choice(COLOR_PAIRS)
        layout_seed = seed + i * 7

        try:
            env = create_opteq_env(int_color, goal_color, layout_seed, propositions)

            difficulty = measure_empirical_difficulty(
                env, agent_obj, int_color, goal_color,
                n_rollouts=n_difficulty_rollouts
            )

            if difficulty is None:
                env.close()
                continue

            episode = run_optimality_episode(env, agent_obj, int_color)
            env.close()

            if episode['chosen_int'] is not None:
                chose_empirically_easier = (episode['chosen_int'] == difficulty['empirically_easier'])
            else:
                chose_empirically_easier = None

            results.append({
                'config_id': i,
                'int_color': int_color,
                'goal_color': goal_color,
                'layout_seed': layout_seed,
                'difficulty': difficulty,
                'episode': episode,
                'chose_empirically_easier': chose_empirically_easier,
            })

        except Exception as e:
            print(f"\nError in config {i}: {e}")
            continue

    return results


def analyze_results(results, test_name):
    """Analyze and print results with confidence intervals."""
    print(f"\n{'='*70}")
    print(f"{test_name} RESULTS")
    print(f"{'='*70}")

    # Filter to valid results
    valid = [r for r in results if r.get('chose_empirically_easier') is not None]

    if not valid:
        print("No valid results!")
        return {}

    n = len(valid)

    # Empirical difficulty
    chose_easier = sum(1 for r in valid if r['chose_empirically_easier'])
    p_easier = chose_easier / n

    # 95% CI using Wilson score interval
    z = 1.96
    denominator = 1 + z**2/n
    centre = (p_easier + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p_easier*(1-p_easier) + z**2/(4*n))/n) / denominator
    ci_low, ci_high = centre - margin, centre + margin

    print(f"\nUsing EMPIRICAL difficulty labels:")
    print(f"  Chose empirically easier: {chose_easier}/{n} ({100*p_easier:.1f}%)")
    print(f"  95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print(f"  CI includes 50%: {'YES' if ci_low <= 0.5 <= ci_high else 'NO'}")

    # Geometric difficulty (for comparison)
    if 'chose_geometrically_easier' in valid[0]:
        chose_geo = sum(1 for r in valid if r.get('chose_geometrically_easier'))
        p_geo = chose_geo / n
        print(f"\nUsing GEOMETRIC difficulty labels (for comparison):")
        print(f"  Chose geometrically easier: {chose_geo}/{n} ({100*p_geo:.1f}%)")

    # Agreement between empirical and geometric
    if 'empirical_geo_agree' in valid[0]:
        agree = sum(1 for r in valid if r.get('empirical_geo_agree'))
        print(f"\nEmpirical vs Geometric agreement: {agree}/{n} ({100*agree/n:.1f}%)")

    # Success rate
    successes = sum(1 for r in valid if r['episode']['success'])
    print(f"\nTask success rate: {successes}/{n} ({100*successes/n:.1f}%)")

    # Difficulty difference distribution
    diffs = [r['difficulty'].get('difficulty_diff', 0) for r in valid]
    print(f"\nEmpirical difficulty difference (completion rate):")
    print(f"  Mean: {np.mean(diffs):.3f}, Std: {np.std(diffs):.3f}")
    print(f"  Min: {np.min(diffs):.3f}, Max: {np.max(diffs):.3f}")

    return {
        'n': n,
        'chose_easier': chose_easier,
        'p_easier': p_easier,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'success_rate': successes / n,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='fresh_baseline')
    parser.add_argument('--n-configs', type=int, default=50, help='Number of map configurations')
    parser.add_argument('--n-difficulty-rollouts', type=int, default=15, help='Rollouts per intermediate to measure difficulty')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='interpretability/results/empirical_difficulty')
    args = parser.parse_args()

    print("="*70)
    print("EMPIRICAL DIFFICULTY ANALYSIS")
    print("="*70)
    print(f"Model: {args.exp}")
    print(f"Configurations: {args.n_configs}")
    print(f"Difficulty rollouts per intermediate: {args.n_difficulty_rollouts}")
    print(f"Total rollouts estimate: ~{args.n_configs * args.n_difficulty_rollouts * 2 * 2} (optvar + opteq)")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model, propositions = load_model(args.exp)
    agent = Agent(model, set(propositions))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run optvar test
    optvar_results = run_empirical_optimality_test(
        agent, propositions,
        n_configs=args.n_configs,
        n_difficulty_rollouts=args.n_difficulty_rollouts,
        seed=args.seed
    )
    optvar_analysis = analyze_results(optvar_results, "OPTVAR (varied maps)")

    # Run equidistant test
    opteq_results = run_empirical_equidistant_test(
        agent, propositions,
        n_configs=args.n_configs,
        n_difficulty_rollouts=args.n_difficulty_rollouts,
        seed=args.seed + 1000
    )
    opteq_analysis = analyze_results(opteq_results, "OPTEQ (equidistant)")

    # Save results
    output = {
        'args': vars(args),
        'timestamp': timestamp,
        'optvar': {
            'results': optvar_results,
            'analysis': optvar_analysis,
        },
        'opteq': {
            'results': opteq_results,
            'analysis': opteq_analysis,
        },
    }

    output_file = f"{args.output_dir}/empirical_difficulty_{args.exp}_{timestamp}.json"

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_numpy(output), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOPTVAR (varied maps):")
    if optvar_analysis:
        print(f"  Chose empirically easier: {optvar_analysis['p_easier']*100:.1f}% (N={optvar_analysis['n']})")
        print(f"  95% CI: [{optvar_analysis['ci_low']*100:.1f}%, {optvar_analysis['ci_high']*100:.1f}%]")

    print(f"\nOPTEQ (equidistant):")
    if opteq_analysis:
        print(f"  Chose empirically easier: {opteq_analysis['p_easier']*100:.1f}% (N={opteq_analysis['n']})")
        print(f"  95% CI: [{opteq_analysis['ci_low']*100:.1f}%, {opteq_analysis['ci_high']*100:.1f}%]")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
