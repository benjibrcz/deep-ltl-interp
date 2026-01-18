#!/usr/bin/env python3
"""
Run training with transition prediction loss.

This script trains the agent with an additional supervised loss that forces
the network to predict next-state env features given current state and action.

The hypothesis is that this auxiliary task will force the agent to develop
internal representations that encode the transition/dynamics model, which is
useful for planning (understanding "what happens if I go here").

Usage:
    # From repo root:
    PYTHONPATH=src python interpretability/training/run_transition_loss_training.py

    # With different transition_loss_coef:
    PYTHONPATH=src python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.05
"""

import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run transition loss training')
    parser.add_argument('--transition_loss_coef', type=float, default=0.1,
                        help='Coefficient for transition loss (default: 0.1)')
    parser.add_argument('--from_exp', type=str, default='planning_from_baseline',
                        help='Source experiment to fine-tune from')
    parser.add_argument('--from_seed', type=int, default=0,
                        help='Seed of source model')
    parser.add_argument('--num_steps', type=int, default=2_000_000,
                        help='Number of training steps')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: transition_loss_<coef>)')
    args = parser.parse_args()

    exp_name = args.name or f"transition_loss_{args.transition_loss_coef:.2f}".replace('.', '')

    cmd = [
        'python', 'src/train/train_ppo.py',
        '--env', 'PointLtl2-v0',
        '--model_config', 'PointLtl2-v0',
        '--curriculum', 'PointLtl2-v0',
        '--name', exp_name,
        '--from_exp', args.from_exp,
        '--from_seed', str(args.from_seed),
        '--seed', '0',
        '--device', 'cpu',
        '--num_procs', '8',
        '--num_steps', str(args.num_steps),
        '--transition_loss_coef', str(args.transition_loss_coef),
        '--save',
        '--log_csv',
    ]

    print(f"Starting training with transition_loss_coef={args.transition_loss_coef}")
    print(f"Experiment name: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
