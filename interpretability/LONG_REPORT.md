# World Model vs Behavioral Heuristics in DeepLTL: Full Report

**Research Question**: Does the DeepLTL agent learn an internal world model for planning, or just behavioral heuristics that produce goal-directed behavior without genuine multi-step reasoning?

---

## Executive Summary

| Test Category | Key Result | N | 95% CI |
|--------------|------------|---|--------|
| Safety (local planning) | 80% correct | 50 | - |
| Optimality - geometric | ~50% optimal | 100 | [40%, 60%] |
| Optimality - empirical difficulty | 68-76% easier | 50 | [54%, 86%] |
| Equidistant (no distance cue) | 54-57% optimal | 50 | [40%, 70%] |
| **Orientation bias** | **74% forward** | 95 | p < 0.0001 |
| **Controlled orientation** | **58% optimal** | 84 | [48%, 68%] (not sig.) |
| Value anticipation (controlled) | ~50% | 200 | - |
| Probing: d_int_to_goal | R² = 0.08-0.18 | - | - |

**Conclusion**: The agent succeeds through reactive heuristics, not planning. It uses a **forward motion heuristic** (74% preference for initial heading direction) and perceptual cues (closest zone, blocking). When orientation bias is controlled, optimal choice drops to 58% (not significant). The agent cannot reason about multi-step consequences.

---

# Part 1: Behavioral Testing

## 1.1 Background: Local vs Global Planning

| Planning Type | Definition | Example | Requires World Model? |
|--------------|------------|---------|----------------------|
| **Local** | Sequential navigation through goals | "Go to A, then B" | No - reactive heuristics sufficient |
| **Global** | Choosing between options based on downstream consequences | "Choose A1 or A2 based on which makes reaching B easier" | Yes - must simulate future states |

**Hypothesis**: If the agent has only behavioral heuristics (not a world model), it should succeed at local planning but fail at global planning.

---

## 1.2 Safety Planning (Local)

**Task**: `(F green | F yellow) & G !blue`
- Choose between green and yellow goals
- One goal is blocked by a blue zone (which must be avoided)
- Agent must detect the blocking and choose the unblocked goal

### Results: 80% Correct

The agent successfully identifies when a goal is blocked and redirects to the alternative. This works because blocking detection is **perceptual** - it's a pattern in the current observation (obstacle between agent and goal), not a simulation of future states.

| Success | Failure |
|:-------:|:-------:|
| Agent avoids blue, chooses yellow | Agent wanders, reaches neither |
| ![Success](results/safety/run_2.png) | ![Failure](results/safety/run_4.png) |

---

## 1.3 Optimality Planning (Global)

**Task**: `F blue THEN F green`
- Must reach a blue zone first, then reach green
- Two blue zones available with different strategic values
- **Optimal choice**: Blue closer to green (minimizes total path)
- **Myopic choice**: Blue closer to agent (greedy)

### Results with Geometric Labels: ~50% Optimal

| Model | Optimal | Myopic | Success |
|-------|---------|--------|---------|
| fresh_baseline | 50% | 50% | 93% |
| combined_aux02_trans01 | 49% | 51% | 75% |

Using geometric path length as the difficulty metric, the agent's choices appear essentially **random**.

![Optimality test trajectories](results/optimality_analysis/optimality_test_fresh_baseline.png)

*Green trajectories = optimal choice, orange = myopic choice*

### Results with Empirical Difficulty Labels

We measured actual difficulty via rollouts (completion rate, mean steps) instead of geometric distance:

| Model | Chose Empirically Easier | 95% CI | CI includes 50%? |
|-------|--------------------------|--------|------------------|
| fresh_baseline | 68.0% (N=50) | [54.2%, 79.2%] | **NO** |
| combined_aux02_trans01 | 76.0% (N=50) | [62.6%, 85.7%] | **NO** |

**Key finding**: Empirical and geometric difficulty agree only 58% of the time. The agent appears to use a heuristic that correlates better with empirical difficulty than geometric distance.

**Why this fails at true planning**: Choosing optimally requires computing chained distances:
- d(agent → blue1) + d(blue1 → green) vs
- d(agent → blue2) + d(blue2 → green)

This requires simulating "if I go to blue1, where will I be, and how far to green from there?" - a world model capability.

---

## 1.4 Equidistant Optimality Test

**Question**: When the "myopic" cue (closer to agent) is removed, can the agent choose based on full path length?

**Setup**: Custom `PointLtl2-v0.opteq` environment where both intermediate zones are placed at the **same distance** from the agent (within 0.05 tolerance). The only way to choose optimally is to consider the full path to the goal.

### Results: Random Choice (~50/50)

**Geometric labels:**

| Model | Optimal | Suboptimal | Success |
|-------|---------|------------|---------|
| fresh_baseline | 54% | 46% | 93% |
| combined_aux02_trans01 | 53% | 45% | 78% |

**Empirical difficulty labels:**

| Model | Chose Easier | 95% CI | CI includes 50%? |
|-------|-------------|--------|------------------|
| fresh_baseline | 62.0% (N=50) | [48.2%, 74.1%] | YES |
| combined_aux02_trans01 | 57.1% (N=49) | [43.3%, 70.0%] | YES |

![Equidistant test trajectories](results/optimality_analysis/optimality_test_equidistant_fresh_baseline.png)

**Key Insight**: When the "closest intermediate" heuristic cannot differentiate between options, the agent's performance drops to ~50-60% with confidence intervals including chance. This confirms the model does NOT compute multi-step paths.

---

## 1.5 What Determines "Empirical Difficulty"?

We investigated why the agent appears to choose "empirically easier" options at rates above chance (68-76% in OPTVAR).

### Finding: Spatial Position Bias Confounds the Results

| Model | OPTEQ Choice | Spatial Bias |
|-------|--------------|--------------|
| fresh_baseline | 54% "optimal" | **LEFT** (66% chose x<0) |
| combined_aux02_trans01 | 47% "optimal" | **RIGHT** (61% chose x>0) |

**Critical Analysis:** After controlling for spatial position:
- Goal direction (d_goal) has NO unique effect on choice (p = 0.49)
- Spatial position (pos_x) IS significant (p = 0.0002)

### Layout Generation Confound

In the OPTEQ environment:
- **70%** of toward-goal zones are on the LEFT
- **34%** of away-from-goal zones are on the LEFT
- Correlation: d_goal ~ pos_x = **0.41***

This means fresh_baseline's LEFT bias aligns with goal direction, creating a spurious "optimal" choice rate.

### Completion Rate Has No Geometric Correlates (OPTVAR)

| Feature | Correlation with Completion Rate | p-value |
|---------|----------------------------------|---------|
| d_agent | r = -0.10 | 0.33 |
| d_goal | r = -0.04 | 0.66 |
| geometric_total | r = -0.07 | 0.52 |

In OPTVAR, completion rate (which defines "empirically easier") has **NO correlation with ANY geometric feature**. The 68% "chose easier" rate reflects a shared heuristic between agent behavior and the completion measurement - both driven by the same spatial factors.

### Interpretation

The "above-chance" empirical difficulty findings are confounded by:
1. Arbitrary spatial biases in each model (LEFT vs RIGHT preference)
2. Non-random placement of optimal zones in the layout generation

**Neither geometric nor empirical difficulty metrics support planning.** The agent uses spatial heuristics.

---

## 1.6 Orientation Bias Analysis

**Question**: Is the "spatial bias" actually an orientation bias (forward motion preference)?

### Hypothesis

The agent doesn't have a LEFT/RIGHT preference in absolute coordinates. Instead, it has a **forward motion preference** - it goes in the direction it's initially facing.

### Orientation Analysis Results

| Metric | fresh_baseline | Significance |
|--------|---------------|--------------|
| Forward preference | **73.7%** | p < 0.0001 |
| LEFT/RIGHT bias | 56.8% | Much weaker |
| Chose forward when only one zone forward | **79.5%** | N=39 |

**Key Finding**: The apparent "LEFT bias" (66%) is explained by initial heading direction. The agent prefers forward motion, and heading happens to correlate with zone position.

### Controlled Orientation Test

**Setup**: Set agent orientation to face the midpoint between the two intermediate zones. This way, both zones are equally "forward" and any preference must come from actual planning.

**Results**:

| Model | Optimal Choice | 95% CI | p-value | Spatial Bias |
|-------|---------------|--------|---------|--------------|
| fresh_baseline | **58.3%** | [48.3%, 67.7%] | 0.125 | 50%/50% L/R |

**Critical Findings**:
- Optimal choice rate is **NOT statistically significant** (CI includes 50%)
- Spatial bias is **completely eliminated** (50%/50% LEFT/RIGHT)
- When forward bias is controlled, the agent chooses essentially at random

### Implications

1. The "spatial bias" finding was actually measuring **orientation bias**
2. The agent uses a simple forward motion heuristic, not goal-directed planning
3. When this heuristic is neutralized, no evidence of planning remains

---

# Part 2: Probing Analysis

## 2.1 What Does the Model Encode?

We trained linear probes to decode information from model activations (GRU hidden states) during task execution.

### Strongly Encoded Features

| Feature | R² | Interpretation |
|---------|-----|----------------|
| Distance to each zone | 0.74-0.93 | Agent knows where things are |
| Blocking detection | 95% accuracy | Explains safety success |
| Agent position | 0.85+ | Basic spatial awareness |

### Weakly Encoded Features

| Feature | R² | Interpretation |
|---------|-----|----------------|
| d_int_to_goal (intermediate → goal) | 0.08-0.18 | **Critical gap** - doesn't compute |
| Total path via intermediate | 0.37-0.54 | No chained distances |
| Optimality gap | 0.15-0.25 | Can't compare paths |

### Key Finding

The model strongly encodes **immediate spatial features** (distances from self, blocking patterns) but weakly encodes **computed/relational features** (distances between other objects, chained path lengths).

This explains the behavioral results: the agent can use perceptual information (blocking, closest zone) but cannot compute multi-step path costs.

---

# Part 3: Value Function Analysis

Based on the theoretical framework from "General Agents Need World Models" (Richens et al., arXiv:2506.01622).

## 3.1 Theoretical Background

**Theorem 2**: Depth-1 (myopic) goals are compatible with many transition models. An agent can achieve myopic optimality without encoding a unique world model.

**Theorem 1**: Depth-n goals (n>1) DO constrain the world model. To optimally choose between sequences, the agent must reason about reachability.

## 3.2 Agent Verification

| Agent | Curriculum Success | Random 2-Step Success |
|-------|-------------------|----------------------|
| fresh_baseline | **100%** (50/50) | **90%** (18/20) |

The agent reliably completes sequential tasks, confirming the test infrastructure works.

## 3.3 Value Function Tests

### Test A: Does V prefer easier full sequences?

| Model | V(easy) > V(hard) | Correlation |
|-------|-------------------|-------------|
| fresh_baseline | 57.5% | r = -0.252 |
| planning_from_baseline | 61% | r = -0.27 |
| combined_aux02_trans01 | 58% | r = -0.22 |

Weak but detectable preference for easier sequences.

### Test B: Same first target, different second targets

| Model | V higher for easier second |
|-------|---------------------------|
| fresh_baseline | **52.0%** |
| combined_aux02_trans01 | **50.5%** |

Essentially random - no anticipation when first step is controlled.

### Test C: Suffix Marginal Value (Key Discriminator)

Compute ΔV = V(s, [A,C]) - V(s, [A]) and compare for easy C vs hard C.

| Model | ΔV higher for easy | Mean ΔV (easy) | Mean ΔV (hard) | Difference |
|-------|-------------------|----------------|----------------|------------|
| fresh_baseline | 49.5% | -0.1850 | -0.1875 | **0.0026** |
| combined_aux02_trans01 | 50.5% | - | - | **0.0008** |

### Interpretation

The value function is **first-step dominated**:
- Test A's 57-61% is explained by correlation: easier first step → easier total sequence
- When first step is controlled (Tests B & C), value shows **no discrimination** of second-step difficulty
- Marginal value of adding second step is identical regardless of difficulty (ΔV diff ≈ 0)

---

# Part 4: Training Interventions

## 4.1 Approaches Tested

| Approach | Description | Implementation |
|----------|-------------|----------------|
| Hard environments | Train on maps where myopic = longer path | 2M steps fine-tuning |
| Step penalty | Subtract penalty each timestep | `--step_penalty 0.002` |
| Auxiliary loss | Predict chained distances | `--aux_loss_coef 0.2` |
| Transition loss | Predict next-state features | `--transition_loss_coef 0.1` |
| Combined | Aux + transition together | Both flags |

## 4.2 Results

### Hard Environment Training

| Metric | Before | After |
|--------|--------|-------|
| Task Success | 30% | **100%** |
| Optimal Choice | ~50% | ~50% |
| Episode Length | 240 steps | 122 steps |

Agent became faster and more reliable, but planning behavior unchanged.

### Auxiliary Loss Sweep

| aux_loss_coef | Probe R² | Optimal Choice |
|---------------|----------|:--------------:|
| 0.0 (baseline) | 0.315 | ~50% |
| 0.1 | 0.340 | ~50% |
| 0.2 | 0.356 | ~50% |
| 0.3 | 0.320 | ~50% |

### Combined Approach

| Model | Probe R² | Optimal Choice | Task Success |
|-------|----------|:--------------:|:------------:|
| Baseline | 0.315 | ~50% | 93% |
| Combined | **0.405** | ~50% | 75% |

## 4.3 Key Conclusion (Auxiliary Losses)

**Improving probe R² does not improve planning behavior.**

All approaches improved the model's ability to encode chained distances (as measured by linear probes), but none improved actual decision-making on optimality tasks.

---

## 4.4 Curriculum & Discount Interventions

Based on insights from the DeepLTL author, we tested whether curriculum and discount changes could induce planning:

### Author's Hypotheses

1. **High discount (0.998)** makes return differences between optimal/suboptimal paths minimal
2. **1-step curriculum start** biases the agent toward "nearest zone" heuristics
3. **Lower discount + 2-step emphasis** might force the agent to learn planning

### Experiments

| Experiment | Discount | Curriculum | Description |
|------------|----------|------------|-------------|
| fresh_baseline | 0.998 | baseline | Original (1-step start) |
| twostep_lowdiscount | 0.95 | 2-step only | Start with 2-step sequences |
| opt_d095_mixed | 0.95 | mixed | 75% 2-step + 25% 1-step |
| opt_d099_mixed | 0.99 | mixed | Higher discount for stability |

### Training Results

| Experiment | Task Success | Plateau Stability |
|------------|--------------|-------------------|
| fresh_baseline | **91%** | Stable (±2%) |
| twostep_lowdiscount | **38%** | Declining |
| opt_d095_mixed | **64%** | Noisy (±8%) |
| opt_d099_mixed | **85%** | Stable (±4%) |

The pure 2-step curriculum with low discount (0.95) was too aggressive - the agent couldn't learn effectively. The mixed curriculum with d=0.99 achieved near-baseline performance.

### Optimality Test Results (Controlled Orientation)

| Model | Optimal Choice | 95% CI | p-value |
|-------|---------------|--------|---------|
| fresh_baseline | 58.3% | [48.3%, 67.7%] | 0.125 |
| **opt_d099_mixed** | **52.0%** | [42.3%, 61.5%] | **0.764** |

### Key Finding

**Curriculum and discount interventions do not improve planning.**

The opt_d099_mixed model achieves 85% task success with the 2-step-heavy curriculum, but shows **52% optimal choice** - indistinguishable from random (p=0.764). The agent finds heuristic solutions regardless of how it's trained.

This suggests:
1. The lack of planning is not due to insufficient training signal
2. The agent consistently discovers non-planning strategies
3. Planning may require architectural changes, not just training modifications

---

# Part 5: Conclusions

## 5.1 Summary of Evidence

| Evidence Type | Finding | Supports Planning? |
|--------------|---------|-------------------|
| Safety task (80%) | Perceptual pattern matching | No |
| Optimality task (~50%) | Random with geometric labels | No |
| Equidistant test (~50-60%) | Loses preference without distance cue | No |
| Orientation bias (74%) | Forward motion heuristic, not L/R | No |
| Controlled orientation (58%) | Random when forward bias removed (p=0.125) | No |
| Probing (R² 0.08-0.18) | Missing chained distance representations | No |
| Value function (~50% controlled) | First-step dominated | No |
| Auxiliary loss interventions | Probe R² up but behavior unchanged | No |
| Curriculum/discount (opt_d099_mixed) | 52% optimal (p=0.764), still random | No |

## 5.2 The Heuristic Hierarchy

The agent uses a hierarchy of reactive heuristics:

1. **Forward motion** - Strong preference (74%) to go in the direction initially facing
2. **Closest zone** - When one intermediate is clearly closer, prefer it
3. **Random** - When above heuristics don't differentiate

Note: The previously identified "spatial position bias" (LEFT/RIGHT preference) was actually **orientation bias** in disguise. When we control for initial heading direction, the spatial bias disappears.

These heuristics explain the results:
- OPTVAR ~50%: Closest-zone heuristic produces random-looking results because agent starts closer to non-optimal
- OPTVAR 68% empirical: Forward motion bias correlates with layout generation patterns
- OPTEQ ~54%: Forward motion bias aligns with zone positions due to heading distribution
- OPTEQ controlled orientation 58%: When forward bias removed, essentially random (p=0.125)

## 5.3 Implications

1. **Planning is not emergent from task success**: Even when optimal planning would help, RL finds alternative solutions (reactive heuristics)

2. **Behavioral testing must control for confounds**: The ~68% "empirically easier" rate looked significant but is explained by orientation bias. After controlling for initial heading (facing midpoint), the agent chooses at random (58%, not significant).

3. **"Spatial bias" was orientation bias**: The apparent LEFT/RIGHT preference was actually a **forward motion preference** (74%). When we control for heading direction, spatial bias disappears entirely (50%/50%).

4. **Auxiliary supervision improves representations but not behavior**: Higher probe R² doesn't translate to better decisions

5. **The forward motion heuristic is robust**: The agent strongly prefers to go in the direction it's initially facing. This is likely learned from the training distribution where forward motion is often rewarded.

6. **Curriculum and discount changes don't help**: Even with 2-step-heavy curriculum (75% 2-step + 25% 1-step) and lower discount (0.99), the agent achieves 85% task success but still shows random optimal choice (52%, p=0.764). The lack of planning appears fundamental to RL in this domain.

---

# Appendix A: File Reference

## Analysis Scripts
| File | Purpose |
|------|---------|
| `analysis/optimality_test_clean.py` | Optimality test with varied maps |
| `analysis/optimality_test_equidistant.py` | Equidistant optimality test |
| `analysis/empirical_difficulty_analysis.py` | Empirical difficulty measurement |
| `analysis/analyze_empirical_difficulty.py` | Spatial bias and confound analysis |
| `analysis/analyze_orientation_bias.py` | Orientation (forward) bias analysis |
| `analysis/optimality_test_controlled_orientation.py` | Optimality test with controlled orientation |
| `analysis/preview_optvar_maps.py` | Preview map layouts |

## Custom Environments
| File | Purpose |
|------|---------|
| `src/.../ltl_optimality_varied.py` | PointLtl2-v0.optvar |
| `src/.../ltl_optimality_equidistant.py` | PointLtl2-v0.opteq |

## Probing
| File | Purpose |
|------|---------|
| `probing/probe_planning_representations.py` | Linear probe analysis |
| `probing/probe_optvar_planning.py` | Probing on optvar environment |

## Value Function Analysis
| File | Purpose |
|------|---------|
| `world_model_extraction/05_value_function_planning_test.py` | Tests A, B, C |
| `world_model_extraction/FINDINGS_REPORT.md` | Full value analysis |

## Results
| Directory | Contents |
|-----------|----------|
| `results/safety/` | Safety task trajectories |
| `results/optimality_analysis/` | Optimality test results |
| `results/empirical_difficulty/` | Empirical difficulty and orientation bias analysis |
| `results/empirical_difficulty/orientation_bias_*.csv` | Orientation bias raw data |
| `results/empirical_difficulty/controlled_orientation_*.csv` | Controlled orientation test data |

## Training Scripts
| File | Purpose |
|------|---------|
| `run_optimality_sweep.py` | Sweep discount factors and curricula |
| `src/sequence/samplers/curriculum.py` | Curriculum definitions (ZONES_TWOSTEP, ZONES_MIXED) |

## Trained Models (Curriculum Experiments)
| Directory | Description |
|-----------|-------------|
| `experiments/ppo/PointLtl2-v0/fresh_baseline/` | Baseline (d=0.998, 1-step start) |
| `experiments/ppo/PointLtl2-v0/extended_baseline/` | Extended training (30M steps) |
| `experiments/ppo/PointLtl2-v0/twostep_lowdiscount/` | 2-step only, d=0.95 |
| `experiments/ppo/PointLtl2-v0/opt_d095_mixed/` | Mixed curriculum, d=0.95 |
| `experiments/ppo/PointLtl2-v0/opt_d099_mixed/` | Mixed curriculum, d=0.99 |

---

# Appendix B: Methodology Notes

## Empirical vs Geometric Difficulty

**Geometric**: Total path length d(agent→int) + d(int→goal)

**Empirical**: Measured via 15 rollouts per intermediate
- Completion rate
- Mean steps to completion

Agreement between labels: only 58%

## Confidence Intervals

All CIs are 95% Wilson score intervals, appropriate for proportions.

## Sample Sizes

- Optimality tests: N=50-100 configurations
- Value function tests: N=200 episodes
- Probing: Training on ~10k datapoints

---

# References

- Richens et al. (2025). "General Agents Need World Models." arXiv:2506.01622
- DeepLTL paper (original training methodology)
