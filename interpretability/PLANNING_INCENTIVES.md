# Approaches to Incentivize Learned Planning

The training on hard optimality maps showed that environmental difficulty alone doesn't induce planning - the agent found alternative solutions (faster myopic execution) rather than developing chained distance computation.

This document outlines approaches to explicitly incentivize planning representations.

---

## 1. Step Penalty (Reward Shaping)

**Implemented**: `--step_penalty` argument in training

**Mechanism**: Subtract a small penalty from reward each step.
```
reward = task_reward - step_penalty
```

**Why it might work**:
- Makes total episode return depend on path length
- Myopic path (longer) gets lower return than optimal path (shorter)
- Creates direct gradient toward path-efficient behavior

**Considerations**:
- Penalty must be small enough that task completion is still primary objective
- Too large: agent may give up on difficult tasks
- Too small: may not differentiate optimal from myopic

**Recommended values**: 0.001 - 0.005

---

## 2. Efficiency Bonus (End-of-Episode Reward)

**Status**: Not implemented

**Mechanism**: At episode end, give bonus based on path efficiency.
```python
optimal_path = compute_optimal_path(start, intermediate, goal)
actual_path = sum(step_distances)
efficiency_bonus = bonus_scale * (optimal_path / actual_path)
```

**Why it might work**:
- Directly rewards choosing shorter total paths
- Requires computing optimal path (oracle)
- Single reward signal at end may be harder to learn from

**Considerations**:
- Requires path length oracle
- Credit assignment is harder (single delayed reward)
- But clearer signal about what matters

---

## 3. Auxiliary Prediction Loss (Representation Learning)

**Implemented**: `--aux_loss_coef` argument in training

**Mechanism**: Add auxiliary head that predicts chained distances.
```python
# During training:
predicted_total = model.aux_head(embedding)
actual_total = d(agent, intermediate) + d(intermediate, goal)
aux_loss = MSE(predicted_total, actual_total)

total_loss = policy_loss + value_loss + aux_loss_coef * aux_loss
```

**Why it might work**:
- Directly forces network to compute chained distances
- Creates gradient that shapes internal representations
- Doesn't change the RL objective, just adds representation constraint

**Usage**:
```bash
python src/train/train_ppo.py ... --aux_loss_coef 0.1
# or use the training script:
python interpretability/training/run_aux_loss_training.py --aux_loss_coef 0.1
```

**Recommended values**: 0.05 - 0.2

---

## 4. Transition Prediction Loss (Dynamics Learning)

**Implemented**: `--transition_loss_coef` argument in training

**Mechanism**: Add auxiliary head that predicts next-state env features given current state and action.
```python
# During training:
current_env_emb = model.compute_env_embedding(obs)
predicted_next = model.transition_head(current_env_emb, action)
actual_next = next_obs_features  # stored during collection
transition_loss = MSE(predicted_next, actual_next)  # only on valid transitions

total_loss = policy_loss + value_loss + transition_loss_coef * transition_loss
```

**Why it might work**:
- Forces network to learn dynamics/transition function
- Understanding "what happens if I go here" is crucial for planning
- More general than chained distance - teaches full state prediction
- May help agent understand spatial relationships and movement

**Usage**:
```bash
python src/train/train_ppo.py ... --transition_loss_coef 0.1
# or use the training script:
python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.1
```

**Recommended values**: 0.05 - 0.2

---

## 5. Contrastive Choice Learning (Not Implemented)

**Status**: Not implemented

**Mechanism**: Present pairs of intermediate choices, train to predict which leads to shorter total path.
```python
# Given two intermediates:
choice_logit = model.compare_intermediates(obs, inter1, inter2)
label = 1 if total_via_inter1 < total_via_inter2 else 0
contrastive_loss = BCE(choice_logit, label)
```

**Why it might work**:
- Binary classification is easier than regression
- Directly targets the decision the agent needs to make
- Could use hindsight data from completed episodes

---

## 6. Hindsight Relabeling (Not Implemented)

**Status**: Not implemented

**Mechanism**: After episode, compute what optimal choice would have been. Create synthetic training data that emphasizes optimal behavior.
```python
# After myopic episode:
if chose_worse_intermediate:
    create_synthetic_experience(
        state=start_state,
        action=action_toward_optimal_intermediate,
        reward=higher_reward
    )
```

**Why it might work**:
- Provides counterfactual supervision
- Can leverage completed episodes for additional training signal

---

## 7. Value Decomposition (Not Implemented)

**Status**: Not implemented

**Mechanism**: Decompose value into immediate and future components.
```python
V(s) = V_immediate(s) + V_future(s)
# Force V_future to represent downstream value
```

**Why it might work**:
- Makes future value explicit in representation
- Could help with temporal credit assignment

---

## 8. Information Bottleneck (Not Implemented)

**Status**: Not implemented (experimental)

**Mechanism**: Remove direct distance information from observation, forcing network to compute from raw lidar.
```python
# Instead of:
obs = [position, lidar, zone_distances]
# Use:
obs = [position, lidar]  # No zone distances
```

**Why it might work**:
- Forces distance computation rather than lookup
- Network must learn spatial reasoning

**Risk**: May just make task harder without improving planning

---

## Recommended Experiments

### Experiment 1: Step Penalty Sweep
```bash
# Try different penalty values
python run_step_penalty_training.py --step_penalty 0.001 --name penalty_001 --from_exp planning_from_baseline
python run_step_penalty_training.py --step_penalty 0.002 --name penalty_002 --from_exp planning_from_baseline
python run_step_penalty_training.py --step_penalty 0.005 --name penalty_005 --from_exp planning_from_baseline
```

### Experiment 2: Auxiliary Prediction Loss
```bash
# Train with auxiliary prediction loss
python interpretability/training/run_aux_loss_training.py --aux_loss_coef 0.1 --from_exp planning_from_baseline
# Try different coefficients
python interpretability/training/run_aux_loss_training.py --aux_loss_coef 0.05 --name aux_005 --from_exp planning_from_baseline
python interpretability/training/run_aux_loss_training.py --aux_loss_coef 0.2 --name aux_020 --from_exp planning_from_baseline
```

### Experiment 3: Transition Prediction Loss
```bash
# Train with transition prediction loss
python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.1 --from_exp planning_from_baseline
# Try different coefficients
python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.05 --name trans_005 --from_exp planning_from_baseline
python interpretability/training/run_transition_loss_training.py --transition_loss_coef 0.2 --name trans_020 --from_exp planning_from_baseline
```

### Experiment 4: Combined Approaches
Use step penalty + auxiliary prediction or transition prediction together.

---

## Success Criteria

1. **Behavioral**: Optimal choice rate on global planning tests should increase from ~50% (random) to >70%
2. **Representational**: Chained distance probe R² should increase from ~0.1-0.2 to >0.5
3. **Generalization**: Improved planning should transfer to novel zone configurations

---

## Experimental Results

### Training on Hard Optimality Maps

**Question**: Can we induce planning by training on environments where myopic behavior leads to much longer paths?

**Setup**: Created 4 "hard optimality" map configurations where:
- Each color has 2 zones at strategic positions
- Myopic choice leads to 24-81% longer total paths
- Fine-tuned from existing agent for 2M steps

**Results**: Improved Robustness, NOT Planning

| Metric | Before Training | After Training |
|--------|-----------------|----------------|
| Task Success Rate | 30% | **100%** |
| Optimal Choice Rate | ~50% | ~50% |
| Episode Length | 240 steps | 122 steps |

The agent learned to complete tasks faster and more reliably, but its planning behavior didn't change.

**Interpretation**: Training on hard environments doesn't induce planning. The agent found an alternative optimization path - became faster at executing its existing (myopic) strategy without developing chained distance computation.

---

### Step Penalty Results

**Result**: Made agent faster but NOT smarter

| Metric | Without Penalty | With Penalty (0.002) |
|--------|-----------------|----------------------|
| Episode Length | 122 steps | **95 steps** |
| Optimal Choice | ~50% | ~50% |

The agent learned to move faster but planning didn't improve.

---

### Auxiliary Chained Distance Prediction Results

We swept auxiliary loss coefficients:

| aux_loss_coef | Probe R² | Optimal Choice Rate |
|---------------|----------|:-------------------:|
| 0.0 (baseline) | 0.315 | ~50% |
| 0.1 | 0.340 | ~50% |
| 0.15 | 0.350 | ~50% |
| **0.2** | **0.356** | ~50% |
| 0.3 | 0.320 | ~50% |

The auxiliary loss improved probe R² scores but did NOT improve actual planning behavior on varied maps.

---

### Transition Prediction Loss Results

| Model | Probe R² | Optimal Choice | Task Success |
|-------|----------|:--------------:|:------------:|
| Baseline | 0.315 | ~50% | 93% |
| Transition Loss (0.1) | 0.361 | ~50% | 78% |

Despite learning to predict next-state features, this didn't translate to better planning decisions.

---

### Combined Approach Results

| Model | Chained Distance R² | Optimal Choice | Task Success |
|-------|:-------------------:|:--------------:|:------------:|
| Baseline | 0.315 | ~50% | 93% |
| Aux Loss (0.2) | 0.356 | ~50% | - |
| Trans Loss (0.1) | 0.361 | ~50% | 78% |
| **Combined** | **0.405** | ~50% | 75% |

The combined model has the strongest planning representations (R²=0.405), but this does NOT translate to improved planning behavior on varied maps or equidistant tests.

---

## Key Conclusion

**Improving probe R² does not improve planning behavior.** All approaches improved the model's ability to encode chained distances (as measured by linear probes), but none improved actual decision-making on optimality tasks.

This suggests:
1. The model may encode the information but not use it for decisions
2. Planning requires more than just having the right representations
3. The policy network may need architectural changes to leverage planning information
