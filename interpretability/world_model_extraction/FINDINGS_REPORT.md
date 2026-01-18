# World Model Extraction Experiment: Findings Report

## Overview

This report documents experiments testing whether the DeepLTL agent has learned an internal world model, based on the theoretical framework from "General Agents Need World Models" (Richens et al., arXiv:2506.01622).

## Theoretical Background

The paper establishes:

**Theorem 2**: Depth-1 (myopic) goals are compatible with many transition models. An agent can achieve myopic optimality without encoding a unique world model - success at "reach A or B" does not certify what the agent believes about transitions.

**Theorem 1**: Depth-n goals (n>1) DO constrain the world model. To optimally choose between sequences "A→C" vs "B→D", the agent must reason about reachability: "If I go to A first, can I then reach C?"

**Key Implication**:
- Depth-1: Competence does not identify transitions (many models compatible)
- Depth-2+: If agent has world model, choices should reflect it

## Agent Verification

Before presenting results, we verify the agent achieves expected training performance:

| Agent | Curriculum Success | Random 2-Step Success |
|-------|-------------------|----------------------|
| `fresh_baseline` | **100%** (50/50) | **90%** (18/20) |

This confirms the test infrastructure is working correctly and results below reflect genuine agent behavior, not test bugs.

## Corrected Experiments

All experiments below use `fresh_baseline` with verified 90-100% task completion.

### Experiment 1: Depth-1 Disjunctive Test

**Goal**: "reach blue OR reach green"

**Question**: Does agent prefer the closer zone?

**Results** (99 episodes that reached a zone):

| Metric | Result |
|--------|--------|
| Chose closer zone | 42.4% |
| Chose farther zone | 57.6% |

**Interpretation**: In our symmetric setup (zones at varying distances, no obstacles), the agent's choices are near chance. This is consistent with Theorem 2: depth-1 disjunctions do not require a world model for competence, so observing ~50% here does not tell us whether the agent has one. The near-chance result reflects our symmetric setup, not a direct prediction of the theorem.

### Experiment 2: Sequential Goal Execution

**Goal**: "reach A THEN reach B" (full sequential goal)

**Question**: Can the agent follow sequential goals?

**Results** (99 episodes that reached a zone):

| Metric | Result |
|--------|--------|
| Reached correct first zone | 84.8% |
| Went to wrong zone first | 15.2% |

**Interpretation**: The agent CAN follow sequential goals 84.8% of the time. The goal-conditioning is working. But this doesn't test planning - it just tests whether the agent responds to the current goal.

### Experiment 3: Sequence Completion on Random Maps

**Goal**: Random 2-step sequences on random environment configurations

**Results** (20 episodes):

| Metric | Result |
|--------|--------|
| Completed (2/2) | 90% (18/20) |
| Partial (1/2) | 5% (1/20) |
| Failed (0/2) | 5% (1/20) |

**Interpretation**: The agent successfully completes 90% of random 2-step sequences, matching expected training performance. This confirms the agent has learned effective goal-following behavior.

### Experiment 4: Does First Choice Optimize Full Path?

**Setup**:
- Two possible first zones (A or B)
- Each leads to different second zone
- Total path lengths differ

**Question**: Does the agent's first choice consider the FULL sequence?

**Result**: 53% chose the zone leading to shorter total path (near chance)

**Interpretation**: The agent does NOT optimize for full path length. But there's a methodological issue: the agent is given "reach A OR B" (depth-1), so it doesn't actually know about the second step. This test doesn't properly probe planning.

## Key Insight: The Fundamental Challenge

The paper's extraction algorithm requires querying the agent with disjunctive SEQUENTIAL goals:

> "You can complete (A→C) OR (B→D). Which do you choose?"

But our implementation only gives disjunctive goals at the FIRST step:

> "Reach A or B" (agent doesn't know about C or D)

This means we can't directly test whether the agent considers future consequences because **we never tell it about the future**.

## What We CAN Conclude

### Consistent with Theorem 2 (Myopic Goals)

Depth-1 choices are near chance in our symmetric setup. This is consistent with Theorem 2's claim that myopic competence does not identify transitions - but the ~50% is a property of our setup, not a direct prediction of the theorem.

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

All tests use `fresh_baseline` (verified 100% curriculum success, 90% random 2-step success).

### Test A: Does V prefer easier full sequences?

| Metric | Result |
|--------|--------|
| V(easy_seq) > V(hard_seq) | **57.5%** |
| Correlation (length vs value) | **r = -0.252** |
| Episodes tested | 200 |

Weak but statistically detectable preference for easier sequences.

### Test B: Same first target, different second targets

| Metric | Result |
|--------|--------|
| V higher for easier second | **52.0%** |
| Episodes tested | 200 |

Essentially random - no anticipation when first step is controlled.

### Test C: Suffix Marginal Value (Key Discriminator)

Compute ΔV = V(s, [A,C]) - V(s, [A]) and compare for easy C vs hard C.

| Metric | Result |
|--------|--------|
| ΔV higher for easy second | **49.5%** |
| Mean ΔV (easy) | -0.1850 |
| Mean ΔV (hard) | -0.1875 |
| **Difference** | **0.0026** |
| Episodes tested | 200 |

The marginal value of adding the second step is virtually identical regardless of whether it's easy or hard. This confirms the value function is **first-step dominated**.

### Interpretation

The 57.5% in Test A is explained by correlation: when the first step is easier, the total sequence tends to be easier too. But when we control for the first step (Tests B & C), the value function shows **no discrimination** of second-step difficulty.

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

**Verified agent performance:** `fresh_baseline` achieves 100% curriculum success (50/50) and 90% random 2-step success (18/20).

| Test | fresh_baseline | planning_from_baseline | combined_aux02_trans01 | Interpretation |
|------|----------------|------------------------|------------------------|----------------|
| Depth-1 disjunctive | 42.4% closer | 46% closer | - | Near chance (consistent with Thm 2) |
| Sequential execution | 84.8% correct | 74% correct | - | Agent follows goals |
| 2-step completion (random) | **90%** | - | - | High success confirms test validity |
| V prefers easier sequence | 57.5%, r=-0.25 | 61%, r=-0.27 | 58%, r=-0.22 | Weak correlation |
| V anticipates (controlled) | 52% | 52% | 50.5% | Random - no anticipation |
| Marginal ΔV discriminates | 49.5%, diff=0.003 | 52%, diff=0.0006 | 50.5%, diff=0.0008 | First-step dominated |

**Conclusion**: The value function has weak global sensitivity to full-sequence difficulty (~57-61%), but shows no reliable anticipation of the second step when the first target is controlled (~50-52%). The marginal value test confirms the value is first-step dominated (ΔV diff ≈ 0.0006-0.003).

Critically, **planning-incentivised training (aux + transition losses) does not improve lookahead**. All agents show identical patterns of first-step dominance.

This suggests the lack of lookahead is a fundamental architectural property, not a training signal problem. The value function learns to reflect near-term difficulty regardless of auxiliary objectives.

This is consistent with **reactive/myopic behavior** rather than genuine world model-based planning.

---

## Files

- `01_extraction_algorithm.py` - Algorithm 2 implementation (original)
- `02_depth_sweep_test.py` - Original depth-sweep test (flawed methodology)
- `03_corrected_depth_test.py` - Corrected experiments (Experiments 1-4)
- `04_sequence_difficulty_test.py` - Sequence length/difficulty analysis
- `05_value_function_planning_test.py` - Value function Tests A, B, C
- `06_visualize_sequences.py` - Trajectory visualization
- `07_random_sequence_test.py` - Random 2-step sequence execution test
- `08_test_with_curriculum.py` - Curriculum sampler verification
- `results/` - JSON outputs and trajectory visualizations

## References

- Richens et al. (2025). "General Agents Need World Models." arXiv:2506.01622
- Previous interpretability report: `interpretability/REPORT.md`
