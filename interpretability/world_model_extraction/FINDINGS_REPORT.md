# World Model Extraction Experiment: Findings Report

## Overview

This report documents experiments testing whether the DeepLTL agent has learned an internal world model, based on the theoretical framework from "General Agents Need World Models" (Richens et al., arXiv:2506.01622).

## Theoretical Background

The paper establishes:

**Theorem 2**: Depth-1 (myopic) goals provide NO information about transitions. An agent can succeed at "reach A or B" without any world model - just go to whichever seems easier.

**Theorem 1**: Depth-n goals (n>1) DO require a world model. To choose between sequences "A→C" vs "B→D", the agent must reason: "If I go to A first, can I then reach C?"

**Key Prediction**:
- Depth-1: ~50% random choice (no planning signal)
- Depth-2+: If agent has world model, choices should optimize for full sequence

## Corrected Experiments

### Experiment 1: Depth-1 Disjunctive Test

**Goal**: "reach blue OR reach green"

**Question**: Does agent prefer the closer zone?

**Results** (100 episodes, all reached a zone):

| Metric | Result |
|--------|--------|
| Chose closer zone | 46% |
| Chose farther zone | 54% |

**Interpretation**: ~50% split confirms Theorem 2. At depth-1, the agent shows no planning - it essentially picks randomly between the two options. This is exactly what the paper predicts: myopic goals provide no information about transitions.

### Experiment 2: Sequential Goal Execution

**Goal**: "reach A THEN reach B" (full sequential goal)

**Question**: Can the agent follow sequential goals?

**Results** (100 episodes):

| Metric | Result |
|--------|--------|
| Reached correct first zone | 74% |
| Went to wrong zone first | 26% |

**Interpretation**: The agent CAN follow sequential goals 74% of the time. The goal-conditioning is working. But this doesn't test planning - it just tests whether the agent responds to the current goal.

### Experiment 3: Sequence Completion vs Length

**Goal**: Sequential goals of varying lengths

**Results**:

| Sequence Length | Completion Rate |
|-----------------|-----------------|
| 1 zone | 46.7% |
| 2 zones | 13.3% |
| 3 zones | 13.3% |

**Interpretation**: Completion drops sharply from 1→2 zones, then plateaus. The agent struggles with multi-step sequences, but this could be due to navigation difficulty rather than lack of planning.

### Experiment 4: Does First Choice Optimize Full Path?

**Setup**:
- Two possible first zones (A or B)
- Each leads to a different second zone
- Total path lengths differ

**Question**: Does the agent's first choice consider the FULL sequence?

**Result**: 41% chose the zone leading to shorter total path (below random 50%)

**Interpretation**: The agent does NOT optimize for full path length. But there's a methodological issue: the agent is given "reach A OR B" (depth-1), so it doesn't actually know about the second step. This test doesn't properly probe planning.

## Key Insight: The Fundamental Challenge

The paper's extraction algorithm requires querying the agent with disjunctive SEQUENTIAL goals:

> "You can complete (A→C) OR (B→D). Which do you choose?"

But our implementation only gives disjunctive goals at the FIRST step:

> "Reach A or B" (agent doesn't know about C or D)

This means we can't directly test whether the agent considers future consequences because **we never tell it about the future**.

## What We CAN Conclude

### Confirmed: Theorem 2 (Myopic Goals)

Depth-1 choices are ~50% random, exactly as predicted. No planning signal.

### Confirmed: Agent Follows Sequential Goals

74% of the time, the agent goes to the first zone in its assigned sequence.

### Not Tested: Depth-Dependent Extraction

We cannot properly test whether depth>1 goals reveal more about the world model because our goal format doesn't support disjunctive sequences.

### Consistent with Previous Findings

The results align with prior interpretability work:
- Controlled choice experiments: 55% vs 45% (essentially random)
- No value anticipation before obstacles
- Success through reactive heuristics, not planning

## Proper Test Design (For Future Work)

To truly test the paper's claims, we would need:

1. **Disjunctive Sequential Goals**: Present agent with explicit choice between full sequences
   - Option A: blue → yellow → magenta
   - Option B: green → blue → yellow
   - Observe: which sequence does agent commit to?

2. **Vary Path Difficulty**: Create scenarios where:
   - Closer first zone leads to HARDER second step
   - Farther first zone leads to EASIER second step
   - Test: does agent choose based on full path?

3. **Value Function Analysis**: Track value estimates throughout
   - Does value anticipate future difficulty?
   - Does value differ between easy and hard sequences?

## Value Function Planning Tests (Most Informative)

These tests directly query the value function, which is how DeepLTL selects sequences.

### Test A: Does V prefer easier full sequences?

| Metric | Result |
|--------|--------|
| V(easy_seq) > V(hard_seq) | **61%** |
| Correlation (length vs value) | **r = -0.27** |

Weak but statistically detectable preference for easier sequences.

### Test B: Same first target, different second targets

| Metric | Result |
|--------|--------|
| V higher for easier second | **52%** |

Essentially random - no anticipation when first step is controlled.

### Test C: Suffix Marginal Value (Key Discriminator)

Compute ΔV = V(s, [A,C]) - V(s, [A]) and compare for easy C vs hard C.

| Metric | Result |
|--------|--------|
| ΔV higher for easy second | **52%** |
| Mean ΔV (easy) | -0.2177 |
| Mean ΔV (hard) | -0.2184 |
| **Difference** | **0.0006** |

The marginal value of adding the second step is virtually identical regardless of whether it's easy or hard. This confirms the value function is **first-step dominated**.

### Interpretation

The 61% in Test A is explained by correlation: when the first step is easier, the total sequence tends to be easier too. But when we control for the first step (Tests B & C), the value function shows **no discrimination** of second-step difficulty.

This is evidence of **weak/limited lookahead**: the value function reflects mostly near-term difficulty, not future consequences.

## Comparison: Baseline vs Planning-Incentivised Agent

We compared the baseline agent (`planning_from_baseline`) against an agent trained with auxiliary planning heads (`combined_aux02_trans01`) that includes both:
- **Auxiliary loss**: Predicts next propositions
- **Transition loss**: Predicts state transitions

### Results

| Test | Baseline | Combined (aux + trans) |
|------|----------|------------------------|
| A: V prefers easier sequence | 61%, r=-0.27 | 58%, r=-0.22 |
| B: V anticipates (controlled) | 52% | 50.5% |
| C: Suffix ΔV discriminates | 52%, diff=0.0006 | 50.5%, diff=0.0008 |

### Key Finding

**The planning-incentivised training did not improve lookahead capability.**

Both agents show:
- Weak global correlation with full-sequence difficulty (~58-61%)
- Essentially random (~50%) when first step is controlled
- Near-zero marginal value difference (~0.0006-0.0008)

This suggests the lack of lookahead is a **fundamental property** of how the value function is learned in this architecture, not something that auxiliary planning losses can fix. The auxiliary heads may help with other aspects (e.g., representation learning, sample efficiency), but they do not induce genuine multi-step anticipation in the value function.

## Summary

| Test | Baseline | Combined | Interpretation |
|------|----------|----------|----------------|
| Depth-1 disjunctive | 46% closer | - | ~Random, confirms Theorem 2 |
| Sequential execution | 74% correct | - | Agent follows goals |
| V prefers easier sequence | 61%, r=-0.27 | 58%, r=-0.22 | Weak correlation |
| V anticipates (controlled) | 52% | 50.5% | Random - no anticipation |
| Marginal ΔV discriminates | 52%, diff=0.0006 | 50.5%, diff=0.0008 | First-step dominated |

**Conclusion**: The value function has weak global sensitivity to full-sequence difficulty (~58-61%), but shows no reliable anticipation of the second step when the first target is controlled (~50-52%). The marginal value test confirms the value is first-step dominated (ΔV diff ≈ 0.0006-0.0008).

Critically, **planning-incentivised training (aux + transition losses) does not improve lookahead**. Both baseline and combined agents show identical patterns of first-step dominance.

This suggests the lack of lookahead is a fundamental architectural property, not a training signal problem. The value function learns to reflect near-term difficulty regardless of auxiliary objectives.

This is consistent with **reactive/myopic behavior** rather than genuine world model-based planning.

---

## Files

- `01_extraction_algorithm.py` - Algorithm 2 implementation (original)
- `02_depth_sweep_test.py` - Original depth-sweep test (flawed methodology)
- `03_corrected_depth_test.py` - Corrected experiments
- `04_sequence_difficulty_test.py` - Sequence length/difficulty analysis
- `05_value_function_planning_test.py` - Value function Tests A, B, C
- `results/` - JSON outputs for both baseline and combined agents

## References

- Richens et al. (2025). "General Agents Need World Models." arXiv:2506.01622
- Previous interpretability report: `interpretability/REPORT.md`
