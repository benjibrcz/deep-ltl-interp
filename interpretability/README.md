# DeepLTL Interpretability Research

This folder contains experiments investigating whether the DeepLTL agent learns genuine planning capabilities or just behavioral heuristics.

## Key Finding

The agent exhibits **local planning** (reactive pattern matching) but fails at **global planning** (requires simulating future states). Training on harder environments improves robustness but doesn't induce planning representations.

See [REPORT.md](REPORT.md) for the full write-up.

## Folder Structure

```
interpretability/
├── REPORT.md                    # Main findings report
├── PLANNING_INCENTIVES.md       # Approaches to incentivize planning
│
├── experiments/                 # Behavioral experiments
│   ├── safety_test.py          # (F A | F B) & G !C - 80% success
│   ├── optimality_test.py      # F blue THEN F green - 10% optimal
│   ├── equidistant_test.py     # Control for directional bias
│   ├── equidistant_swapped_test.py
│   ├── planning_test_battery.py # Local vs global planning tests
│   └── test_hard_optimality_maps.py
│
├── probing/                     # Representation analysis
│   ├── probe_planning_representations.py  # Linear probes
│   ├── probe_analysis_summary.py
│   └── auxiliary_distance_prediction.py   # Aux loss approach
│
├── analysis/                    # Deeper analysis scripts
│   ├── investigate_world_model.py
│   ├── investigate_directional_bias.py
│   └── planning_taxonomy.py
│
├── visualization/               # Figure generation
│   ├── generate_clean_figures.py
│   ├── generate_combined_examples.py
│   ├── generate_findings_figure.py
│   ├── visualize_planning.py
│   └── save_planning_trajectories.py
│
├── training/                    # Training scripts
│   ├── run_hard_optimality.py       # Train on hard maps
│   ├── run_step_penalty_training.py # Train with step penalty
│   └── test_hard_env_setup.py
│
├── tests/                       # Test scripts
│   └── test_planning_capability.py
│
└── results/                     # Generated results
    ├── safety/                  # Safety test results
    ├── optimality/              # Optimality test results
    ├── equidistant/             # Control test results
    ├── probing/                 # Probe results
    ├── planning_battery/        # Local/global battery results
    ├── hard_optimality/         # Hard map analysis
    ├── world_model/             # World model analysis
    └── figures/                 # Generated figures
```

## Quick Start

```bash
# Run safety planning test
PYTHONPATH=src python interpretability/experiments/safety_test.py --exp planning_from_baseline

# Run optimality planning test
PYTHONPATH=src python interpretability/experiments/optimality_test.py --exp planning_from_baseline

# Run probing analysis
PYTHONPATH=src python interpretability/probing/probe_planning_representations.py --exp planning_from_baseline

# Run local vs global planning battery
PYTHONPATH=src python interpretability/experiments/planning_test_battery.py --exp planning_from_baseline
```

## Key Results

| Test | Metric | Baseline | After Aux Loss |
|------|--------|----------|----------------|
| Safety (local) | Correct choice | 80% | - |
| Optimality (global) | Optimal choice | 20% | **40%** |
| Task success | Completion rate | 50% | **90%** (combined) |
| Chained distance probe | R² | 0.315 | **0.405** |

## Completed Experiments

1. **Baseline Analysis**: Local vs global planning, directional bias control
2. **Hard Environment Training**: Improved robustness but not planning
3. **Step Penalty**: Made agent faster but not smarter
4. **Auxiliary Loss (chained distance)**: **2x improvement** in planning (20% → 40%)
5. **Transition Loss**: Doesn't improve planning directly
6. **Combined (aux + trans)**: Best overall (40% optimal, 90% success)
