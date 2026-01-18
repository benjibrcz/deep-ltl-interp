#!/usr/bin/env python
"""
Training script for hard optimality maps.

This trains the agent on challenging zone configurations where myopic behavior
(choosing the closest intermediate goal) leads to much longer total paths.

The goal is to force the agent to learn chained distance computation:
d(agent→blue) + d(blue→green) for optimal intermediate goal selection.

Usage:
    # Train from scratch on hard optimality maps
    python run_hard_optimality.py --name hard_opt_scratch --seed 0 --device cuda

    # Fine-tune from a pre-trained model
    python run_hard_optimality.py --name hard_opt_finetune --seed 0 --device cuda --from_exp planning_from_baseline
"""
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from utils import kill_all_wandb_processes


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True
    # Fine-tuning options
    from_exp: str | None = None  # Load weights from this experiment
    from_seed: int = 0  # Seed of the model to load


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'PointLtl2-v0.hardmix',  # Hard optimality environment
            '--steps_per_process', '4096',
            '--batch_size', '2048',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '5_000_000',  # Shorter since we're fine-tuning
            '--model_config', 'PointLtl2-v0',  # Same model architecture
            '--curriculum', 'PointLtl2-v0.hardmix',  # Hard optimality curriculum
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]

        # Add fine-tuning options
        if args.from_exp:
            command.extend(['--from_exp', args.from_exp])
            command.extend(['--from_seed', str(args.from_seed)])

        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        print(f"Running: {' '.join(command)}")
        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 2 --device cpu --name test_hard_opt --seed 1 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        sys.exit(0)
