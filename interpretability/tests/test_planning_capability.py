"""
Test the planning capabilities of the trained model.

Tests:
1. Optimality: For (F A | F B), does the agent choose the closer goal?
2. Safety: For (F A | F B) & G !C, does the agent avoid the blocked path?

Uses the sequence wrapper directly (not LTL formulas).
"""

import random
import numpy as np
import torch
from tqdm import tqdm

import preprocessing
from envs import make_env, get_env_attr
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from ltl.automata import LDBASequence
from ltl.logic import Assignment
from sequence.samplers import CurriculumSampler
from sequence.samplers.sequence_samplers import (
    all_disjunctive_reach_tasks,
    all_reach_global_avoid_tasks,
    all_disjunctive_reach_global_avoid_tasks,
)


def create_disjunctive_reach_sampler(propositions):
    """Create sampler for (F A | F B) tasks."""
    tasks = all_disjunctive_reach_tasks(1, num_disjuncts=2)(propositions)
    def sampler():
        return random.choice(tasks)
    return sampler


def create_global_avoid_sampler(propositions):
    """Create sampler for F A & G !B tasks."""
    tasks = all_reach_global_avoid_tasks(1)(propositions)
    def sampler():
        return random.choice(tasks)
    return sampler


def create_combined_sampler(propositions):
    """Create sampler for (F A | F B) & G !C tasks."""
    tasks = all_disjunctive_reach_global_avoid_tasks(num_disjuncts=2)(propositions)
    def sampler():
        return random.choice(tasks)
    return sampler


class SimpleAgent:
    """Simple agent that just uses the model with sequence observations."""
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


def test_formula_type(env_name, exp, seed, sampler_fn, formula_name, num_episodes=100):
    """
    Test agent on a specific formula type.
    """
    print("\n" + "="*60)
    print(f"TEST: {formula_name}")
    print("="*60)

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # First make a temp env to get propositions
    from sequence.samplers import curricula
    temp_sampler = CurriculumSampler.partial(curricula['PointLtl2-v0'])
    temp_env = make_env(env_name, temp_sampler, sequence=True)
    propositions = list(temp_env.get_propositions())
    temp_env.close()

    # Create the actual sampler
    sampler = sampler_fn(propositions)

    # Now create env with our test sampler
    def partial_sampler(props):
        return sampler
    env = make_env(env_name, partial_sampler, sequence=True)

    # Load model
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    env.reset(seed=seed)

    successes = 0
    violations = 0
    timeouts = 0
    total_steps = []

    for i in tqdm(range(num_episodes), desc=formula_name):
        obs, info = env.reset(), {}
        done = False
        num_steps = 0

        # Print first few tasks to understand what we're testing
        if i < 3:
            goal = obs.get('initial_goal', obs.get('goal', 'unknown'))
            print(f"\n  Episode {i}: goal = {goal}")

        while not done:
            action = agent.get_action(obs, deterministic=True)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1

            if done:
                if 'success' in info:
                    successes += 1
                    total_steps.append(num_steps)
                elif 'violation' in info:
                    violations += 1
                else:
                    timeouts += 1

    env.close()

    success_rate = successes / num_episodes
    violation_rate = violations / num_episodes
    timeout_rate = timeouts / num_episodes
    avg_steps = np.mean(total_steps) if total_steps else 0

    print(f"\nSuccess rate: {success_rate:.2%}")
    print(f"Violation rate: {violation_rate:.2%}")
    print(f"Timeout rate: {timeout_rate:.2%}")
    print(f"Avg steps (on success): {avg_steps:.1f}")

    return success_rate, violation_rate, timeout_rate


def main():
    import sys
    sys.path.insert(0, 'src')

    env_name = 'PointLtl2-v0'
    exp = 'planning_from_baseline'
    seed = 42
    num_episodes = 50

    print("="*60)
    print("PLANNING CAPABILITY TESTS")
    print(f"Model: {exp}")
    print(f"Environment: {env_name}")
    print(f"Episodes per test: {num_episodes}")
    print("="*60)

    results = {}

    # Test 1: Disjunctive reach (F A | F B)
    results['disjunctive'] = test_formula_type(
        env_name, exp, seed,
        create_disjunctive_reach_sampler,
        "Disjunctive Reach (F A | F B)",
        num_episodes
    )

    # Test 2: Global safety (F A & G !B)
    results['safety'] = test_formula_type(
        env_name, exp, seed,
        create_global_avoid_sampler,
        "Global Safety (F A & G !B)",
        num_episodes
    )

    # Test 3: Combined (F A | F B) & G !C
    results['combined'] = test_formula_type(
        env_name, exp, seed,
        create_combined_sampler,
        "Combined Planning (F A | F B) & G !C",
        num_episodes
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Disjunctive reach (F A | F B):     {results['disjunctive'][0]:.2%} success, {results['disjunctive'][1]:.2%} violations")
    print(f"Global safety (F A & G !B):        {results['safety'][0]:.2%} success, {results['safety'][1]:.2%} violations")
    print(f"Combined (F A | F B) & G !C:       {results['combined'][0]:.2%} success, {results['combined'][1]:.2%} violations")


if __name__ == '__main__':
    main()
