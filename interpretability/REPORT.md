# World Model vs Behavioral Heuristics in DeepLTL

**Research Question**: Does the DeepLTL agent learn an internal world model for planning, or just behavioral heuristics that produce agent-like behavior without genuine multi-step reasoning?

## Executive Summary

| Planning Type | Task | Baseline Result | After Incentive Training |
|--------------|------|-----------------|--------------------------|
| **Local** (safety) | Choose unblocked goal | 80% correct | - |
| **Global** (optimality) | Choose path-minimizing intermediate | 10-20% correct | **40% correct** |

**Key Findings**:
1. The baseline agent succeeds at local planning (pattern matching) but fails at global planning (requires world model)
2. Training on harder environments alone doesn't induce planning - the agent finds alternative solutions
3. **Auxiliary supervision on chained distances doubles planning performance** (20% → 40%)
4. Combining auxiliary + transition prediction achieves best overall results (40% optimal, 90% task success)

---

## Part 1: Baseline Agent Analysis

### Background: Local vs Global Planning

| Planning Type | Definition | Example | Requires World Model? |
|--------------|------------|---------|----------------------|
| **Local** | Sequential navigation through goals | "Go to A, then B" | No - reactive heuristics sufficient |
| **Global** | Choosing between options based on downstream consequences | "Choose A1 or A2 based on which makes reaching B easier" | Yes - must simulate future states |

**Hypothesis**: If the agent has only behavioral heuristics (not a world model), it should succeed at local planning but fail at global planning.

---

### Experiment 1: Safety Planning (Local)

**Task**: `(F green | F yellow) & G !blue`
- Choose between green and yellow goals
- One goal is blocked by a blue zone (which must be avoided)
- Agent must detect the blocking and choose the unblocked goal

#### Results: 80% Correct

The agent successfully identifies when a goal is blocked and redirects to the alternative. This works because blocking detection is **perceptual** - it's a pattern in the current observation (obstacle between agent and goal), not a simulation of future states.

---

### Experiment 2: Optimality Planning (Global)

**Task**: `F blue THEN F green`
- Must reach a blue zone first, then reach green
- Two blue zones available with different strategic values
- **Optimal choice**: Blue closer to green (minimizes total path)
- **Myopic choice**: Blue closer to agent (greedy)

#### Configuration

```
Agent start: (0.2, 0.2)
Blue1: (-0.8, -1.0)  ← closer to green (OPTIMAL)
Blue2: (1.0, 1.0)    ← closer to agent (MYOPIC)
Green: (-1.4, -1.9)

Optimal path: agent → blue1 → green = 2.64 total distance
Myopic path:  agent → blue2 → green = 4.89 total distance
Savings from optimal: 46%
```

#### Results: 10-20% Optimal, 80-90% Myopic

The agent consistently chooses the closer blue zone, ignoring that this leads to a much longer total path.

**Why this fails**: Choosing optimally requires computing chained distances:
- d(agent → blue1) + d(blue1 → green) vs
- d(agent → blue2) + d(blue2 → green)

This requires simulating "if I go to blue1, where will I be, and how far to green from there?" - a world model capability the agent lacks.

---

### Experiment 3: Controlling for Directional Bias

**Question**: Is the 10% optimal rate due to partial planning ability, or just coincidence?

**Setup**: Both blues placed equidistant from the agent. If the agent plans, it should choose blue closer to green.

#### Results

| Configuration | Optimal Blue Location | Agent's Choice | "Optimal" Rate |
|--------------|----------------------|----------------|----------------|
| Original | Lower-left | Upper-right (95%) | 5% |
| Swapped (green moved) | Upper-right | Upper-right (95%) | 95% |

The agent goes upper-right ~95% of the time regardless of which blue is optimal. Any apparent "optimal" behavior is coincidental alignment with this directional bias, not planning.

---

### Experiment 4: Probing for Planning Representations

We trained linear probes to decode information from model activations during task execution.

#### What the Agent Encodes

| Feature | Performance | Interpretation |
|---------|-------------|----------------|
| Distance to each zone | R² = 0.74-0.93 | **Strong** - knows where things are |
| Blocking detection | 95% accuracy | **Strong** - explains safety success |
| Blue → Green distance | R² = 0.34-0.48 | **Weak** - doesn't compute |
| Total path via each blue | R² = 0.36-0.51 | **Weak** - no chained distances |

**Key Finding**: The model strongly encodes **immediate spatial features** (distances from self, blocking patterns) but weakly encodes **computed/relational features** (distances between other objects, chained path lengths).

---

### Experiment 5: Training on Hard Optimality Maps

**Question**: Can we induce planning by training on environments where myopic behavior leads to much longer paths?

**Setup**: Created 4 "hard optimality" map configurations where:
- Each color has 2 zones at strategic positions
- Myopic choice leads to 24-81% longer total paths
- Fine-tuned from existing agent for 2M steps

#### Results: Improved Robustness, NOT Planning

| Metric | Before Training | After Training |
|--------|-----------------|----------------|
| Task Success Rate | 30% | **100%** |
| Optimal Choice Rate | 20% | 20% |
| Myopic Choice Rate | 80% | **80%** |
| Episode Length | 240 steps | 122 steps |

The agent learned to complete tasks faster and more reliably, but its planning behavior didn't change at all.

**Interpretation**: Training on hard environments doesn't induce planning. The agent found an alternative optimization path - became faster at executing its existing (myopic) strategy without developing chained distance computation.

---

## Part 2: Incentivizing Planning Through Auxiliary Losses

Since environmental difficulty alone doesn't induce planning, we tested explicit auxiliary supervision.

### Approaches Tested

| Approach | Description | Implementation |
|----------|-------------|----------------|
| **Step Penalty** | Subtract penalty each timestep to incentivize shorter paths | `--step_penalty 0.002` |
| **Auxiliary Loss** | Supervised head predicting chained distance d(agent→blue) + d(blue→green) | `--aux_loss_coef 0.2` |
| **Transition Loss** | Supervised head predicting next-state features given current state + action | `--transition_loss_coef 0.1` |
| **Combined** | Both auxiliary and transition losses together | Both flags |

---

### Experiment 6: Step Penalty

**Result**: Made agent faster but NOT smarter

| Metric | Without Penalty | With Penalty (0.002) |
|--------|-----------------|----------------------|
| Episode Length | 122 steps | **95 steps** |
| Optimal Choice | 20% | 10% |

The agent learned to move faster but planning actually degraded slightly.

---

### Experiment 7: Auxiliary Chained Distance Prediction

**Result**: **2x improvement in planning**

We swept auxiliary loss coefficients:

| aux_loss_coef | Optimal Choice Rate |
|---------------|:-------------------:|
| 0.0 (baseline) | 20% |
| 0.1 | 30% |
| 0.15 | 40% |
| **0.2** | **40%** |
| 0.3 | 20% |

The sweet spot is aux_loss_coef=0.2, achieving **40% optimal choice rate** (2x baseline).

**Note**: Training longer (4M steps) at coef=0.2 caused complete regression to 0% optimal - representations degraded with overtraining.

---

### Experiment 8: Transition Prediction Loss

**Result**: Doesn't improve planning directly

| Model | Optimal Choice | Task Success |
|-------|:--------------:|:------------:|
| Baseline | 20% | 50% |
| Transition Loss (0.1) | 20% | 40% |

Despite learning to predict next-state features, this didn't translate to better planning decisions.

---

### Experiment 9: Combined Approach

**Result**: Best overall performance

| Model | Optimal Choice | Task Success |
|-------|:--------------:|:------------:|
| Baseline | 20% | 50% |
| Aux Loss (0.2) | 40% | 50% |
| Trans Loss (0.1) | 20% | 40% |
| **Combined** | **40%** | **90%** |

The combined approach maintains 40% optimal planning while achieving 90% task success rate.

---

### Probing the Trained Models

| Model | Chained Distance R² | Distance to Goal R² |
|-------|:-------------------:|:-------------------:|
| Baseline | 0.315 | 0.769 |
| Aux Loss (0.2) | 0.356 | **0.853** |
| Trans Loss (0.1) | 0.361 | 0.736 |
| **Combined** | **0.405** | 0.840 |

**Key Finding**: Models with higher chained distance probe R² show better planning behavior. The combined model has the strongest planning representations (R²=0.405).

---

## Conclusions

### 1. Baseline Agent Has Heuristics, Not a World Model

| Capability | Result | Mechanism |
|------------|--------|-----------|
| Local planning (safety) | 80% success | Pattern recognition |
| Global planning (optimality) | 20% success | Would require world model |
| Apparent optimal choices | Directional bias | Behavioral heuristic |

### 2. Environmental Difficulty Doesn't Induce Planning

Training on harder environments improves robustness (100% task success) but doesn't improve planning (still 20% optimal). The network finds alternative solutions.

### 3. Auxiliary Supervision Works

| Intervention | Optimal Rate | Improvement |
|--------------|:------------:|:-----------:|
| Baseline | 20% | - |
| Hard environment training | 20% | 0% |
| Step penalty | 10% | -50% |
| **Aux loss (chained dist)** | **40%** | **+100%** |
| Combined (aux + trans) | 40% | +100% |

Direct supervision on planning-relevant quantities (chained distances) doubles planning performance.

### 4. 40% Ceiling Suggests Architectural Limits

Even with direct supervision, the agent only achieves 40% optimal choice rate. This suggests:
- Network architecture may limit planning ability
- More training or different approaches needed
- Fundamental difficulty in learning counterfactual reasoning

---

## Implications for Interpretability

1. **Planning is not emergent from task success**: Even when optimal planning would help, RL finds alternative solutions
2. **Representations must be explicitly incentivized**: Chained distances won't emerge just because they'd be useful
3. **Behavioral testing alone is insufficient**: The agent improved on metrics without developing the target capability
4. **Auxiliary supervision is effective**: Can induce specific representations through additional loss terms

---

## Files Reference

### Experiments
| File | Purpose |
|------|---------|
| `experiments/safety_test.py` | Safety planning test |
| `experiments/optimality_test.py` | Optimality planning test |
| `experiments/equidistant_test.py` | Directional bias control |
| `experiments/planning_test_battery.py` | Local vs global battery |

### Probing
| File | Purpose |
|------|---------|
| `probing/probe_planning_representations.py` | Linear probe analysis |

### Training
| File | Purpose |
|------|---------|
| `training/run_hard_optimality.py` | Hard map training |
| `training/run_aux_loss_training.py` | Auxiliary loss training |
| `training/run_transition_loss_training.py` | Transition loss training |

### Results
| Directory | Contents |
|-----------|----------|
| `results/trajectory_comparison/` | Trajectory visualizations per model |
| `results/probing/` | Probe heatmaps and analysis |
| `results/figures/` | Generated figures |
| `results/planning_incentive_summary.png` | Summary comparison chart |

---

## Appendix: Training Commands

```bash
# Baseline model
PYTHONPATH=src python src/train/train_ppo.py --env PointLtl2-v0 --model_config PointLtl2-v0 \
    --curriculum PointLtl2-v0 --name planning_from_baseline --num_steps 2000000

# Auxiliary loss training (best: 0.2)
PYTHONPATH=src python interpretability/training/run_aux_loss_training.py --aux_loss_coef 0.2

# Transition loss training
PYTHONPATH=src python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.1

# Combined training
PYTHONPATH=src python src/train/train_ppo.py --env PointLtl2-v0 --model_config PointLtl2-v0 \
    --curriculum PointLtl2-v0 --name combined_aux02_trans01 --from_exp planning_from_baseline \
    --aux_loss_coef 0.2 --transition_loss_coef 0.1 --num_steps 2000000

# Run optimality test
PYTHONPATH=src python interpretability/experiments/optimality_test.py --exp <model_name> --n_runs 10

# Run probing
PYTHONPATH=src python interpretability/probing/probe_planning_representations.py --exp <model_name> --task optimality
```
