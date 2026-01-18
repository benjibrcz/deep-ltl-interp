"""
Visualize planning trajectories with rendering.
"""

import random
import numpy as np
import torch
import time

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


def format_goal(goal):
    """Format goal sequence for display."""
    if not goal:
        return "DONE"

    reach, avoid = goal[0]

    # Extract zone names from frozensets
    reach_names = []
    avoid_names = []

    for assignment in reach:
        # assignment is a frozenset of (prop, value) tuples
        for prop, val in assignment:
            if val:
                reach_names.append(prop)

    for assignment in avoid:
        for prop, val in assignment:
            if val:
                avoid_names.append(prop)

    reach_str = " | ".join(reach_names) if reach_names else "any"
    avoid_str = ", ".join(avoid_names) if avoid_names else "none"

    return f"Reach: {reach_str}, Avoid: {avoid_str}"


def visualize_episodes(env_name, exp, seed, sampler_fn, formula_name, num_episodes=3):
    """
    Visualize agent on a specific formula type.
    """
    print("\n" + "="*60)
    print(f"VISUALIZING: {formula_name}")
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

    # Now create env with our test sampler WITH RENDERING
    def partial_sampler(props):
        return sampler

    # Import safety gymnasium to get render mode
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper
    from gymnasium.wrappers import FlattenObservation, TimeLimit
    from envs.seq_wrapper import SequenceWrapper
    from envs.remove_trunc_wrapper import RemoveTruncWrapper

    # Create env with render_mode='human'
    base_env = safety_gymnasium.make(env_name, render_mode='human')
    base_env = SafetyGymWrapper(base_env)
    base_env = FlattenObservation(base_env)
    env = SequenceWrapper(base_env, sampler)
    env = TimeLimit(env, max_episode_steps=1000)
    env = RemoveTruncWrapper(env)

    # Load model
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    agent = SimpleAgent(model, props)

    env.reset(seed=seed)

    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        obs, info = env.reset(), {}
        done = False
        num_steps = 0

        goal = obs.get('initial_goal', obs.get('goal', 'unknown'))
        print(f"Goal: {format_goal(goal)}")

        while not done:
            action = agent.get_action(obs, deterministic=True)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1

            # Small delay for visualization
            time.sleep(0.02)

            if done:
                if 'success' in info:
                    print(f"SUCCESS in {num_steps} steps!")
                elif 'violation' in info:
                    print(f"VIOLATION at step {num_steps}")
                else:
                    print(f"TIMEOUT after {num_steps} steps")

        # Pause between episodes
        print("Starting next episode in 2 seconds...")
        time.sleep(2)

    env.close()


def main():
    import sys
    import argparse

    sys.path.insert(0, 'src')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='combined',
                        choices=['disjunctive', 'safety', 'combined', 'all'],
                        help='Which test to visualize')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    env_name = 'PointLtl2-v0'
    exp = 'planning_from_baseline'

    print("="*60)
    print("PLANNING TRAJECTORY VISUALIZATION")
    print(f"Model: {exp}")
    print(f"Environment: {env_name}")
    print("="*60)

    tests = {
        'disjunctive': (create_disjunctive_reach_sampler, "Disjunctive Reach (F A | F B)"),
        'safety': (create_global_avoid_sampler, "Global Safety (F A & G !B)"),
        'combined': (create_combined_sampler, "Combined Planning (F A | F B) & G !C"),
    }

    if args.test == 'all':
        for name, (sampler_fn, desc) in tests.items():
            visualize_episodes(env_name, exp, args.seed, sampler_fn, desc, args.episodes)
    else:
        sampler_fn, desc = tests[args.test]
        visualize_episodes(env_name, exp, args.seed, sampler_fn, desc, args.episodes)


if __name__ == '__main__':
    main()
