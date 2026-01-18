"""
Save planning trajectories as images using the zones visualization.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import preprocessing
from envs import make_env, get_env_attr
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler
from sequence.samplers.sequence_samplers import (
    all_disjunctive_reach_tasks,
    all_reach_global_avoid_tasks,
    all_disjunctive_reach_global_avoid_tasks,
)
from visualize.zones import draw_trajectories, draw_zones, draw_path, draw_diamond, setup_axis, FancyAxes


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


def get_zone_positions(env):
    """Extract zone positions from the environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env

    # zone_positions is on the Builder level
    if hasattr(unwrapped, 'zone_positions'):
        return {k: v.copy() for k, v in unwrapped.zone_positions.items()}
    return {}


def get_agent_position(env):
    """Get current agent position."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env

    if hasattr(unwrapped, 'agent_pos'):
        return unwrapped.agent_pos[:2].copy()
    return None


def format_goal(goal):
    """Format goal sequence for display."""
    if not goal:
        return "DONE"

    reach, avoid = goal[0]
    reach_names = []
    avoid_names = []

    for assignment in reach:
        for prop, val in assignment:
            if val:
                reach_names.append(prop)

    for assignment in avoid:
        for prop, val in assignment:
            if val:
                avoid_names.append(prop)

    reach_str = " | ".join(sorted(reach_names)) if reach_names else "any"
    avoid_str = ", ".join(sorted(avoid_names)) if avoid_names else "none"

    return f"Reach: {reach_str}, Avoid: {avoid_str}"


def collect_trajectory(env, agent, max_steps=1000):
    """Run one episode and collect the trajectory."""
    obs, info = env.reset(), {}
    done = False
    num_steps = 0

    # Get zone positions
    zone_positions = get_zone_positions(env)

    # Collect trajectory points
    trajectory = []
    start_pos = get_agent_position(env)
    if start_pos is not None:
        trajectory.append(start_pos.copy())

    goal = obs.get('initial_goal', obs.get('goal', None))
    result = None

    while not done and num_steps < max_steps:
        action = agent.get_action(obs, deterministic=True)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]
        obs, reward, done, info = env.step(action)
        num_steps += 1

        # Record position
        pos = get_agent_position(env)
        if pos is not None:
            trajectory.append(pos.copy())

        if done:
            if 'success' in info:
                result = 'success'
            elif 'violation' in info:
                result = 'violation'
            else:
                result = 'timeout'

    if not done:
        result = 'timeout'

    return {
        'zone_positions': zone_positions,
        'trajectory': trajectory,
        'goal': goal,
        'result': result,
        'steps': num_steps
    }


def save_trajectory_grid(trajectories, filename, title=None):
    """Save a grid of trajectories as an image."""
    n = len(trajectories)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    for i, traj in enumerate(trajectories):
        ax = fig.add_subplot(rows, cols, i + 1, axes_class=FancyAxes, edgecolor='gray', linewidth=.5)
        setup_axis(ax)

        # Draw zones
        draw_zones(ax, traj['zone_positions'])

        # Draw start position (orange diamond)
        if len(traj['trajectory']) > 0:
            draw_diamond(ax, traj['trajectory'][0], color='orange')

        # Draw trajectory path
        path_color = 'green' if traj['result'] == 'success' else 'red'
        draw_path(ax, traj['trajectory'], color=path_color, linewidth=3)

        # Add title with goal info
        goal_str = format_goal(traj['goal'])
        result_str = traj['result'].upper()
        ax.set_title(f"{goal_str}\n{result_str} ({traj['steps']} steps)", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def run_and_save(env_name, exp, seed, sampler_fn, formula_name, num_episodes, output_file):
    """Run episodes and save trajectory visualization."""
    print(f"\nCollecting trajectories for: {formula_name}")

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Get propositions
    from sequence.samplers import curricula
    temp_sampler = CurriculumSampler.partial(curricula['PointLtl2-v0'])
    temp_env = make_env(env_name, temp_sampler, sequence=True)
    propositions = list(temp_env.get_propositions())
    temp_env.close()

    # Create sampler
    sampler = sampler_fn(propositions)
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

    # Collect trajectories
    trajectories = []
    successes = 0

    for i in range(num_episodes):
        traj = collect_trajectory(env, agent)
        trajectories.append(traj)
        if traj['result'] == 'success':
            successes += 1
        print(f"  Episode {i+1}: {traj['result']} in {traj['steps']} steps")

    env.close()

    # Save visualization
    title = f"{formula_name}\nSuccess Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)"
    save_trajectory_grid(trajectories, output_file, title)

    return successes / num_episodes


def main():
    import sys
    sys.path.insert(0, 'src')

    env_name = 'PointLtl2-v0'
    exp = 'planning_from_baseline'
    seed = 123
    num_episodes = 8

    print("="*60)
    print("SAVING PLANNING TRAJECTORY VISUALIZATIONS")
    print(f"Model: {exp}")
    print(f"Environment: {env_name}")
    print("="*60)

    # Test 1: Disjunctive reach
    run_and_save(
        env_name, exp, seed,
        create_disjunctive_reach_sampler,
        "Disjunctive Reach: (F A | F B)",
        num_episodes,
        "trajectories_disjunctive.png"
    )

    # Test 2: Global safety
    run_and_save(
        env_name, exp, seed + 1,
        create_global_avoid_sampler,
        "Global Safety: F A & G !B",
        num_episodes,
        "trajectories_safety.png"
    )

    # Test 3: Combined planning
    run_and_save(
        env_name, exp, seed + 2,
        create_combined_sampler,
        "Combined Planning: (F A | F B) & G !C",
        num_episodes,
        "trajectories_combined.png"
    )

    print("\n" + "="*60)
    print("All trajectory visualizations saved!")
    print("  - trajectories_disjunctive.png")
    print("  - trajectories_safety.png")
    print("  - trajectories_combined.png")
    print("="*60)


if __name__ == '__main__':
    main()
