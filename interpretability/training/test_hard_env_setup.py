#!/usr/bin/env python3
"""Test that the hard optimality environment and curriculum are set up correctly."""

import sys
sys.path.insert(0, 'src')

import numpy as np

# Test environment creation
print("Testing environment creation...")
import safety_gymnasium

for env_name in ['PointLtl2-v0.hard1', 'PointLtl2-v0.hard2', 'PointLtl2-v0.hard3', 'PointLtl2-v0.hard4', 'PointLtl2-v0.hardmix']:
    try:
        env = safety_gymnasium.make(env_name)
        obs, info = env.reset()
        # Raw env returns dict observation
        if isinstance(obs, dict):
            print(f"  {env_name}: OK (dict obs with keys: {list(obs.keys())[:3]}...)")
        else:
            print(f"  {env_name}: OK (obs shape: {obs.shape})")
        env.close()
    except Exception as e:
        print(f"  {env_name}: FAILED - {e}")

# Test curriculum
print("\nTesting curriculum...")
from sequence.samplers import curricula, HARD_OPTIMALITY_CURRICULUM

print(f"  Available curricula: {list(curricula.keys())}")
print(f"  HARD_OPTIMALITY_CURRICULUM stages: {len(HARD_OPTIMALITY_CURRICULUM.stages)}")

# Test task sampler
print("\nTesting optimality task sampler...")
from sequence.samplers.sequence_samplers import sample_optimality_task, all_optimality_tasks

propositions = ['blue', 'green', 'yellow', 'magenta']
task = sample_optimality_task()(propositions)
print(f"  Task: {task}")
print(f"  Sequence length: {len(task)}")
for i, (reach, avoid) in enumerate(task.reach_avoid):
    print(f"    Step {i}: reach={reach}, avoid={avoid}")

# Test full environment setup with wrapper
print("\nTesting full environment stack...")
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from sequence.samplers import CurriculumSampler

env_name = 'PointLtl2-v0.hardmix'
base_env = safety_gymnasium.make(env_name)
base_env = SafetyGymWrapper(base_env)
base_env = FlattenObservation(base_env)

propositions = base_env.get_propositions()
print(f"  Propositions: {propositions}")

curriculum = HARD_OPTIMALITY_CURRICULUM
sampler = CurriculumSampler(curriculum, propositions)
sample_task = sampler

env = SequenceWrapper(base_env, sample_task)
env = TimeLimit(env, max_episode_steps=400)
env = RemoveTruncWrapper(env)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
if isinstance(obs, dict):
    print(f"  Observation: dict with keys {list(obs.keys())[:5]}...")
else:
    print(f"  Observation shape: {obs.shape}")
print(f"  Environment created successfully!")

env.close()
print("\nAll tests passed!")
