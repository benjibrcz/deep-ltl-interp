# What Determines Empirical Difficulty?

## Key Finding: Spatial Position Bias Confounds "Optimal" Choice

The ~58-68% "chose empirically easier" rate in the optimality tests is NOT evidence of planning. It's driven by:
1. **Arbitrary spatial biases** in each model
2. **Confounds in the layout generation** where optimal zones tend to cluster in certain regions

## Evidence

### OPTEQ (Equidistant Test)

When both intermediates are equidistant from the agent, the only way to choose optimally is to consider full path to goal.

| Model | Chose Optimal | Spatial Bias | Direction |
|-------|--------------|--------------|-----------|
| fresh_baseline | 54% | **LEFT** (66%) | r = -0.36*** |
| combined_aux02_trans01 | 47% | **RIGHT** (61%) | r = +0.21* |

**Critical Finding:** After controlling for spatial position (pos_x):
- d_goal (goal direction) has NO unique effect on choice (p = 0.49 and p = 0.56)
- pos_x has STRONG effect (p = 0.0002 and p = 0.04)

### Layout Confound

In the OPTEQ environment generation:
- **70% of toward-goal zones** are on the LEFT
- **34% of away-from-goal zones** are on the LEFT
- Correlation: d_goal ~ pos_x: **r = 0.41***

This means:
- fresh_baseline's LEFT bias aligns with goal direction → appears "optimal"
- combined's RIGHT bias opposes goal direction → appears random

### OPTVAR (Varied Distances)

In OPTVAR, there's no such confound:
- Toward-goal zones are 50% LEFT, 50% RIGHT
- Agent's choice appears random (~46-50%)

## What Features Actually Correlate with Empirical Difficulty?

### Completion Rate
| Feature | OPTVAR (r) | OPTEQ (r) |
|---------|-----------|-----------|
| d_agent | -0.10 | -0.02 |
| d_goal | -0.04 | **-0.47***|
| geometric_total | -0.07 | **-0.46***|
| pos_x | +0.07 | **-0.33***|

In OPTVAR, completion rate has NO correlation with geometry (all p > 0.3).
In OPTEQ, d_goal dominates.

### Mean Steps (for completed episodes)
| Feature | OPTVAR (r) | OPTEQ (r) |
|---------|-----------|-----------|
| d_agent | **-0.60***| +0.19 |
| d_goal | **+0.89***| **+0.84***|

Steps correlate strongly with geometry, but completion rate (which determines "empirically easier") does not in OPTVAR.

## Interpretation

1. **Empirical difficulty is NOT well-predicted by geometry in OPTVAR**
   - Completion rate appears essentially random
   - The 68% "chose empirically easier" reflects a shared heuristic between the agent's choice and the completion measurement (both influenced by the same spatial/behavioral factors)

2. **Empirical difficulty IS predicted by geometry in OPTEQ**
   - But the agent doesn't use this information
   - Instead, it uses arbitrary spatial biases

3. **The agent has no lookahead**
   - After controlling for spatial position, goal direction has ZERO effect on choice
   - The ~54% "optimal" rate in OPTEQ is entirely explained by spatial bias confounds

## Conclusion

Neither geometric nor empirical difficulty metrics support the hypothesis that the agent plans. The apparent "above-chance" performance is driven by:
1. Arbitrary spatial biases in the trained models
2. Confounds between spatial position and optimal zone placement in the environments

**The agent uses spatial heuristics, not planning.**
