#!/usr/bin/env python
"""
Training with step penalty to incentivize path efficiency.

The step penalty makes shorter paths more rewarding, which should
force the agent to learn optimal intermediate goal selection.

Example:
    python run_step_penalty_training.py --name step_penalty_001 --step_penalty 0.001 --seed 0

Recommended step_penalty values:
    - 0.001: Mild penalty (~0.1 total for 100-step episode)
    - 0.002: Moderate penalty (~0.2 total for 100-step episode)
    - 0.005: Strong penalty (~0.5 total for 100-step episode)

The penalty should be small enough that completing the task is still
the primary objective, but large enough to differentiate optimal from
myopic paths.
"""
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing


@dataclass
class Args:
    name: str
    seed: int | list[int]
    step_penalty: float = 0.002  # Default moderate penalty
    device: str = 'cpu'
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True
    # Fine-tuning options
    from_exp: str | None = None
    from_seed: int = 0


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'PointLtl2-v0.hardmix',
            '--steps_per_process', '4096',
            '--batch_size', '2048',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '5_000_000',
            '--model_config', 'PointLtl2-v0',
            '--curriculum', 'PointLtl2-v0.hardmix',
            '--step_penalty', str(args.step_penalty),
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]

        if args.from_exp:
            command.extend(['--from_exp', args.from_exp])
            command.extend(['--from_seed', str(args.from_seed)])

        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        print(f"Running with step_penalty={args.step_penalty}: {' '.join(command)}")
        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv += '--num_procs 2 --device cpu --name test_step_penalty --seed 1 --step_penalty 0.002 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        sys.exit(0)
