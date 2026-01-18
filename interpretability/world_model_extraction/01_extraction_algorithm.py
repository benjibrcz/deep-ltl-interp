#!/usr/bin/env python3
"""
World Model Extraction from DeepLTL Policy

Implements Algorithm 2 from "General Agents Need World Models" (Richens et al. 2025)
to extract transition probabilities from the goal-conditioned policy.

The key insight: By querying the agent with either-or goals of the form
ψ_a ∨ ψ_b (reach zone A n times vs reach zone B k times), we can infer
what the agent believes about transition probabilities.

For DeepLTL, we adapt this to:
- States = discrete zone positions (discretized from continuous space)
- Actions = direction of movement (discretized or continuous)
- Transitions = P(reach zone X | start near zone Y, move toward Z)

Reference: Algorithm 2, Appendix C of the paper
"""

import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from typing import Callable, Optional
from dataclasses import dataclass
from collections import defaultdict

import preprocessing
from envs import make_env
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from ltl.logic import Assignment
from ltl.automata import LDBASequence


@dataclass
class ExtractionConfig:
    """Configuration for world model extraction"""
    n_trials: int = 10  # Number of repetitions in sequential goal (depth parameter)
    n_episodes_per_query: int = 5  # Episodes to estimate action choice probability
    zone_radius: float = 0.4  # Matches actual zone size in env
    max_steps: int = 500  # Max steps per episode (increased from 300)


def load_model(exp_name: str, seed: int = 0):
    """Load a trained DeepLTL model"""
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp_name, seed=seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    # Check if model has aux/transition heads
    use_aux = 'aux' in exp_name or 'combined' in exp_name
    use_trans = 'transition' in exp_name or 'combined' in exp_name
    model = build_model(env, training_status, config, use_aux_head=use_aux, use_transition_head=use_trans)
    model.eval()

    props = set(env.get_propositions())
    env.close()

    return model, props


def create_sequential_reach_goal(
    target_zone: str,
    n_repetitions: int,
    propositions: set[str]
) -> LDBASequence:
    """
    Create a goal: reach target_zone n times in sequence

    This corresponds to ψ_a in Algorithm 2:
    "Take action a, then reach zone repeatedly"
    """
    reach_assignment = Assignment.single_proposition(target_zone, propositions).to_frozen()
    reach_set = frozenset([reach_assignment])
    avoid_set = frozenset()

    # Create sequence of n reach goals
    sequence = [(reach_set, avoid_set) for _ in range(n_repetitions)]

    return LDBASequence(sequence)


def create_disjunctive_sequential_goal(
    zone_a: str,
    zone_b: str,
    n_for_a: int,
    n_for_b: int,
    propositions: set[str]
) -> LDBASequence:
    """
    Create a composite goal: (reach A n times) OR (reach B m times)

    This is the ψ_a,b(n,m) goal from Algorithm 2.
    Agent must choose which branch to pursue.
    """
    # First step: disjunctive choice between starting toward A or B
    reach_a = Assignment.single_proposition(zone_a, propositions).to_frozen()
    reach_b = Assignment.single_proposition(zone_b, propositions).to_frozen()

    # The disjunction in the first step determines which branch
    initial_reach = frozenset([reach_a, reach_b])
    avoid_set = frozenset()

    # Start with disjunctive choice
    sequence = [(initial_reach, avoid_set)]

    # Note: In practice, we'll need to observe which zone the agent goes to first
    # to determine which branch it chose

    return LDBASequence(sequence)


def create_either_or_goal_pair(
    zone_a: str,
    zone_b: str,
    n_trials: int,
    k_threshold: int,
    propositions: set[str]
) -> tuple[LDBASequence, LDBASequence]:
    """
    Create the either-or goal pair from Algorithm 2:

    ψ_a(k,n): Reach zone_a, goal is satisfied if outcome is zone_a at most k times
    ψ_b(k,n): Reach zone_b, goal is satisfied if outcome is zone_a more than k times

    For DeepLTL, we adapt this to:
    ψ_a: Sequential reach of zone_a (n times)
    ψ_b: Sequential reach of zone_b (k times)

    The agent's choice between pursuing ψ_a vs ψ_b reveals its belief about
    the relative difficulty/probability of reaching each zone.
    """
    goal_a = create_sequential_reach_goal(zone_a, n_trials, propositions)
    goal_b = create_sequential_reach_goal(zone_b, k_threshold, propositions)

    return goal_a, goal_b


class WorldModelExtractor:
    """
    Extract approximate transition probabilities from a goal-conditioned policy
    using Algorithm 2 from the paper.
    """

    def __init__(self, model, propositions: set[str], config: ExtractionConfig):
        self.model = model
        self.propositions = propositions
        self.config = config
        self.device = 'cpu'

        # Extracted model: P_hat(reach zone_j | start near zone_i)
        self.transition_estimates = {}

    def query_policy_preference(
        self,
        env,
        goal_a: LDBASequence,
        goal_b: LDBASequence,
        initial_state: Optional[np.ndarray] = None
    ) -> tuple[float, float]:
        """
        Query which goal the policy prefers by running episodes.

        Returns: (prob_choose_a, prob_choose_b)
        """
        choices = {'a': 0, 'b': 0, 'neither': 0}

        for _ in range(self.config.n_episodes_per_query):
            # Run episode with disjunctive goal and see which zone agent reaches first
            choice = self._run_choice_episode(env, goal_a, goal_b, initial_state)
            choices[choice] += 1

        total = sum(choices.values())
        prob_a = choices['a'] / total if total > 0 else 0.5
        prob_b = choices['b'] / total if total > 0 else 0.5

        return prob_a, prob_b

    def _run_choice_episode(
        self,
        env,
        goal_a: LDBASequence,
        goal_b: LDBASequence,
        initial_state: Optional[np.ndarray] = None
    ) -> str:
        """
        Run an episode and determine which goal the agent chooses to pursue.

        We infer the choice by observing which zone the agent reaches first.
        """
        # Create a disjunctive goal that allows either choice
        # The agent's behavior reveals its preference

        obs = env.reset()

        # Get zone positions from underlying env (RemoveTruncWrapper strips info)
        zone_positions = getattr(env.env, 'zone_positions', {})

        # Find the zones corresponding to goal_a and goal_b
        # goal_a and goal_b are LDBASequence objects
        # Extract the target zone from each
        zone_a_name = self._extract_target_zone(goal_a)
        zone_b_name = self._extract_target_zone(goal_b)

        if zone_a_name is None or zone_b_name is None:
            return 'neither'

        # Run episode with disjunctive goal
        disjunctive_goal = self._create_disjunctive_goal(goal_a, goal_b)

        for step in range(self.config.max_steps):
            # Update observation with current goal
            # Note: list(disjunctive_goal) returns list of (reach, avoid) tuples, as expected by preprocessing
            # Also need 'propositions' for epsilon checking
            current_props = obs.get('propositions', []) if isinstance(obs, dict) else []
            obs_dict = {
                'features': obs if isinstance(obs, np.ndarray) else obs['features'],
                'goal': list(disjunctive_goal),
                'propositions': current_props
            }

            # Get action from policy
            preprocessed = preprocessing.preprocess_obss([obs_dict], self.propositions)
            with torch.no_grad():
                dist, _ = self.model(preprocessed)
                action = dist.mode.numpy().flatten()

            # RemoveTruncWrapper returns (obs, reward, done, info) - no separate truncated
            obs, reward, done, info = env.step(action)

            # Get agent position from underlying env or features
            agent_pos = getattr(env.env, 'agent_pos', None)
            if agent_pos is not None:
                agent_pos = np.array(agent_pos[:2])
            else:
                agent_pos = obs['features'][:2] if isinstance(obs, dict) else obs[:2]

            # Check if reached zone_a
            zone_a_pos = self._find_zone_position(zone_a_name, zone_positions)
            zone_b_pos = self._find_zone_position(zone_b_name, zone_positions)

            if zone_a_pos is not None:
                dist_a = np.linalg.norm(agent_pos - zone_a_pos)
                if dist_a < self.config.zone_radius:
                    return 'a'

            if zone_b_pos is not None:
                dist_b = np.linalg.norm(agent_pos - zone_b_pos)
                if dist_b < self.config.zone_radius:
                    return 'b'

            if done:
                break

        return 'neither'

    def _extract_target_zone(self, goal: LDBASequence) -> Optional[str]:
        """Extract the primary target zone from an LDBASequence goal"""
        if not goal.reach_avoid:
            return None

        reach_set, _ = goal.reach_avoid[0]
        if not reach_set or reach_set == LDBASequence.EPSILON:
            return None

        # Get first assignment from reach set
        for assignment in reach_set:
            # FrozenAssignment has get_true_propositions() method
            true_props = assignment.get_true_propositions()
            if true_props:
                return next(iter(true_props))
        return None

    def _create_disjunctive_goal(self, goal_a: LDBASequence, goal_b: LDBASequence) -> LDBASequence:
        """Create a disjunctive goal allowing either goal_a or goal_b"""
        zone_a = self._extract_target_zone(goal_a)
        zone_b = self._extract_target_zone(goal_b)

        if zone_a is None or zone_b is None:
            return goal_a  # Fallback

        reach_a = Assignment.single_proposition(zone_a, self.propositions).to_frozen()
        reach_b = Assignment.single_proposition(zone_b, self.propositions).to_frozen()

        # Disjunction: agent can satisfy goal by reaching either zone
        reach_set = frozenset([reach_a, reach_b])
        avoid_set = frozenset()

        return LDBASequence([(reach_set, avoid_set)])

    def _find_zone_position(self, zone_name: str, zone_positions: dict) -> Optional[np.ndarray]:
        """Find position of a zone by name prefix"""
        for key, pos in zone_positions.items():
            if key.startswith(zone_name):
                return np.array(pos) if not isinstance(pos, np.ndarray) else pos
        return None

    def extract_transition_estimate(
        self,
        env,
        zone_a: str,
        zone_b: str,
        n_max: int = 10
    ) -> float:
        """
        Implement Algorithm 2 to estimate P(reach zone_a | choice between a and b)

        By varying k from 1 to n and observing when the agent switches preference,
        we can estimate the relative probabilities.

        Returns: Estimated P(zone_a is easier/closer than zone_b)
        """
        # Sweep k from 1 to n_max and find switching point
        k_star = n_max

        for k in range(1, n_max + 1):
            # Create goals: reach zone_a n_max times vs reach zone_b k times
            goal_a, goal_b = create_either_or_goal_pair(
                zone_a, zone_b, n_max, k, self.propositions
            )

            prob_a, prob_b = self.query_policy_preference(env, goal_a, goal_b)

            # If agent switches to preferring goal_a, record k_star
            if prob_a > prob_b:
                k_star = k
                break

        # Estimate: P_hat ≈ (k_star - 0.5) / n_max
        # This comes from the binomial inversion in Algorithm 2
        p_estimate = (k_star - 0.5) / n_max

        return p_estimate


def run_extraction_experiment(
    exp_name: str = 'planning_from_baseline',
    n_trials: int = 10,
    output_dir: str = 'interpretability/world_model_extraction/results'
):
    """
    Run the world model extraction experiment.

    This implements the core test from the paper:
    1. Query the agent with either-or goals at various depth
    2. Extract approximate transition probabilities
    3. Compare to ground truth environment dynamics
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("WORLD MODEL EXTRACTION EXPERIMENT")
    print("Based on Algorithm 2 from 'General Agents Need World Models'")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {exp_name}")
    model, props = load_model(exp_name)

    # Create environment
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, temp_sampler, sequence=True)

    # Create extractor
    config = ExtractionConfig(n_trials=n_trials)
    extractor = WorldModelExtractor(model, props, config)

    # Get available zones
    obs = env.reset()
    zone_positions = getattr(env.env, 'zone_positions', {})

    print(f"\nAvailable zones: {list(zone_positions.keys())}")

    # Extract transition estimates for all zone pairs
    colors = sorted(set(p.split('_')[0] for p in zone_positions.keys()))
    print(f"Colors: {colors}")

    results = {}

    for i, color_a in enumerate(colors):
        for color_b in colors[i+1:]:
            print(f"\n  Extracting P({color_a} vs {color_b})...")

            try:
                p_estimate = extractor.extract_transition_estimate(
                    env, color_a, color_b, n_max=n_trials
                )
                results[(color_a, color_b)] = p_estimate
                print(f"    P({color_a} preferred) = {p_estimate:.3f}")
            except Exception as e:
                print(f"    Error: {e}")
                results[(color_a, color_b)] = None

    env.close()

    # Save results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    for (zone_a, zone_b), p_est in results.items():
        if p_est is not None:
            print(f"  P({zone_a} > {zone_b}) = {p_est:.3f}")

    # Save to file
    import json
    with open(f"{output_dir}/extraction_results_{exp_name}.json", 'w') as f:
        json.dump({
            'exp_name': exp_name,
            'n_trials': n_trials,
            'results': {f"{a}_{b}": p for (a, b), p in results.items() if p is not None}
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='planning_from_baseline')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='interpretability/world_model_extraction/results')
    args = parser.parse_args()

    run_extraction_experiment(args.exp, args.n_trials, args.output_dir)
