# World Model vs Behavioral Heuristics in DeepLTL

**Research Question**: Does the DeepLTL agent learn an internal world model for planning, or just behavioral heuristics that produce agent-like behavior without genuine multi-step reasoning?

## Executive Summary

| Planning Type | Task | Baseline Result | After Incentive Training |
|--------------|------|-----------------|--------------------------|
| **Local** (safety) | Choose unblocked goal | 80% correct | - |
| **Global** (optimality) | Choose path-minimizing intermediate | ~50% (random) | ~50% (random) |
| **Equidistant** (no myopic cue) | Choose when both intermediates same distance | 54% optimal | 53% optimal |

**Key Findings**:
1. The baseline agent succeeds at local planning (pattern matching) but fails at global planning (requires world model)
2. **On varied maps with proper zone separation, both agents show ~50% optimal choice** - essentially random between optimal and myopic
3. **When both intermediates are equidistant from the agent, choice is ~50/50** - confirming the model relies on "closest intermediate" heuristic, not path planning
4. Training on harder environments alone doesn't induce planning - the agent finds alternative solutions
5. Neither baseline nor auxiliary-trained models show evidence of computing multi-step paths

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

#### Results: ~50% Optimal, ~50% Myopic (Updated Jan 2025)

Testing on varied maps with proper zone separation (min 1.0 distance between zones):

| Model | Optimal | Myopic | Success |
|-------|---------|--------|---------|
| fresh_baseline | 50% | 50% | 93% |
| combined_aux02_trans01 | 49% | 51% | 75% |

The agent's choices appear essentially **random** between optimal and myopic when tested across many varied configurations.

**Why this fails**: Choosing optimally requires computing chained distances:
- d(agent → blue1) + d(blue1 → green) vs
- d(agent → blue2) + d(blue2 → green)

This requires simulating "if I go to blue1, where will I be, and how far to green from there?" - a world model capability the agent lacks.

---

### Experiment 3: Equidistant Optimality Test (Updated Jan 2025)

**Question**: When the "myopic" cue (closer to agent) is removed, can the agent choose based on full path length?

**Setup**: Custom `PointLtl2-v0.opteq` environment where both intermediate zones are placed at the **exact same distance** from the agent (within 0.05 tolerance). The only way to choose optimally is to consider the full path to the goal.

#### Results: Random Choice (~50/50)

| Model | Optimal | Suboptimal | Success |
|-------|---------|------------|---------|
| fresh_baseline | 54% | 46% | 93% |
| combined_aux02_trans01 | 53% | 45% | 78% |

When the "closest intermediate" heuristic cannot differentiate between options, **both agents choose essentially at random**.

**Key Insight**: This confirms the model does NOT compute multi-step paths. It relies entirely on the myopic heuristic of "go to the nearest intermediate zone." When that cue is absent, it guesses.

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
| Global planning (optimality) | ~50% (random) | Would require world model |
| Equidistant test | ~50% (random) | No planning, just guessing |

### 2. The ~50% "Optimal" Rate is Random, Not Planning

On varied maps with proper zone separation:
- Both baseline and auxiliary-trained agents show ~50% optimal choice
- When intermediates are equidistant (no myopic cue), choice is still ~50/50
- This is consistent with random selection, not partial planning ability

### 3. Environmental Difficulty Doesn't Induce Planning

Training on harder environments improves robustness but doesn't improve planning. The network finds alternative solutions.

### 4. Auxiliary Supervision Does NOT Induce Planning

| Model | Optimal Rate (Varied Maps) | Optimal Rate (Equidistant) |
|-------|:--------------------------:|:--------------------------:|
| fresh_baseline | 50% | 54% |
| combined_aux02_trans01 | 49% | 53% |

Despite auxiliary supervision on chained distances, the trained model shows no improvement in actual planning behavior.

### 5. Probing Shows Weak Planning Representations

| Feature | R² Score |
|---------|----------|
| d_agent_to_int (observable) | 0.43-0.54 |
| d_int_to_goal (requires computation) | **0.08-0.18** |

The model does not encode the chained distances needed for optimal planning.

---

## Implications for Interpretability

1. **Planning is not emergent from task success**: Even when optimal planning would help, RL finds alternative solutions (myopic heuristics)
2. **Behavioral testing must control for confounds**: The ~50% "optimal" rate initially looked like partial planning, but equidistant testing reveals it's random
3. **Auxiliary supervision improves probe R² but not behavior**: Higher representation quality doesn't guarantee better decision-making
4. **The myopic heuristic is robust**: When both intermediates are equidistant, the agent doesn't fall back to planning - it just guesses

---

## Files Reference

### Experiments
| File | Purpose |
|------|---------|
| `analysis/optimality_test_clean.py` | Optimality test with varied maps/colors |
| `analysis/optimality_test_equidistant.py` | Equidistant optimality test |
| `analysis/preview_optvar_maps.py` | Preview optvar map layouts |
| `analysis/preview_opteq_maps.py` | Preview equidistant map layouts |

### Custom Environments
| File | Purpose |
|------|---------|
| `src/.../ltl_optimality_varied.py` | PointLtl2-v0.optvar environment |
| `src/.../ltl_optimality_equidistant.py` | PointLtl2-v0.opteq environment |

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
