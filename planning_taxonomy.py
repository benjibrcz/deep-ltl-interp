#!/usr/bin/env python3
"""
Taxonomy of Planning Capabilities for LTL Agents

This analyzes different types of planning/transition functions that could
be needed in various scenarios, and assesses whether the current architecture
could support them.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PlanningCapability:
    name: str
    description: str
    computation_type: str  # 'perceptual', 'arithmetic', 'sequential', 'recursive'
    example_task: str
    required_representation: str
    architecture_sufficient: bool  # Can current feedforward arch learn this?
    training_signal_exists: bool   # Does current training incentivize this?
    how_to_test: str
    how_to_train: str


PLANNING_CAPABILITIES = [
    # ========== SPATIAL PLANNING ==========
    PlanningCapability(
        name="Blocking Detection",
        description="Detect if path to goal is obstructed",
        computation_type="perceptual",
        example_task="(F green | F yellow) & G !blue - choose unblocked goal",
        required_representation="Spatial pattern: obstacle between agent and goal",
        architecture_sufficient=True,
        training_signal_exists=True,
        how_to_test="Probe for 'goal_X_blocked' label",
        how_to_train="Current training works (80% success)",
    ),

    PlanningCapability(
        name="Direct Distance",
        description="Know distance to each zone",
        computation_type="perceptual",
        example_task="F green - go to green",
        required_representation="Lidar readings → distance estimate",
        architecture_sufficient=True,
        training_signal_exists=True,
        how_to_test="Probe for dist_to_X",
        how_to_train="Emerges naturally from navigation reward",
    ),

    PlanningCapability(
        name="Chained Distance (2-step)",
        description="Compute d(me→A) + d(A→B) for intermediate goal selection",
        computation_type="arithmetic",
        example_task="F blue & F green - choose blue that minimizes total path",
        required_representation="Compose two distance computations",
        architecture_sufficient=True,  # MLP can compute this
        training_signal_exists=False,  # Current reward doesn't distinguish
        how_to_test="Probe for total_via_blue_i",
        how_to_train="Hard maps where myopic fails + step penalty",
    ),

    PlanningCapability(
        name="Chained Distance (N-step)",
        description="Compute total path through N waypoints",
        computation_type="arithmetic",
        example_task="F a & F b & F c & F d - optimal ordering",
        required_representation="Sum of N distance computations + comparison",
        architecture_sufficient=True,  # Still arithmetic, but harder
        training_signal_exists=False,
        how_to_test="Tasks with 3+ sequential subgoals",
        how_to_train="Curriculum with increasing sequence length",
    ),

    PlanningCapability(
        name="Path Planning (Obstacle Avoidance)",
        description="Plan path around obstacles to reach goal",
        computation_type="sequential",
        example_task="F green & G !blue (blue blocks direct path)",
        required_representation="Waypoint generation, local navigation",
        architecture_sufficient=True,  # Can learn reactive obstacle avoidance
        training_signal_exists=True,
        how_to_test="Mazes with obstacles",
        how_to_train="Current training handles simple cases",
    ),

    # ========== TEMPORAL PLANNING ==========
    PlanningCapability(
        name="Subgoal Ordering",
        description="Determine optimal order to visit multiple goals",
        computation_type="combinatorial",
        example_task="F a & F b & F c - which order minimizes path?",
        required_representation="Compare N! orderings (or heuristic)",
        architecture_sufficient=False,  # Combinatorial explosion
        training_signal_exists=False,
        how_to_test="Tasks with 4+ unordered subgoals",
        how_to_train="Would need search or learned heuristic",
    ),

    PlanningCapability(
        name="Deadline Awareness",
        description="Know if goal is reachable before timeout",
        computation_type="arithmetic",
        example_task="Reach goal in T steps - is it possible?",
        required_representation="distance / speed vs remaining time",
        architecture_sufficient=True,
        training_signal_exists=False,  # No time pressure signal
        how_to_test="Probe for 'reachable_in_time' given deadline",
        how_to_train="Vary episode lengths, tight timeouts",
    ),

    PlanningCapability(
        name="Future State Anticipation",
        description="Predict where agent will be after N steps",
        computation_type="sequential",
        example_task="Any navigation - implicit in planning",
        required_representation="Transition model: s' = f(s, a)",
        architecture_sufficient=True,  # For short horizons
        training_signal_exists=False,  # No auxiliary prediction loss
        how_to_test="Probe for next_position prediction",
        how_to_train="Auxiliary prediction objective",
    ),

    # ========== LOGICAL PLANNING ==========
    PlanningCapability(
        name="Disjunctive Goal Selection",
        description="Choose best option from (A | B | C)",
        computation_type="comparison",
        example_task="(F green | F yellow | F blue) - pick nearest/easiest",
        required_representation="Compare distances/difficulties",
        architecture_sufficient=True,
        training_signal_exists=True,  # Reward for any goal works
        how_to_test="Check if agent picks nearest disjunct",
        how_to_train="Current training + distance-aware reward",
    ),

    PlanningCapability(
        name="Constraint Satisfaction",
        description="Find path satisfying multiple constraints",
        computation_type="logical",
        example_task="F green & G !blue & G !yellow - navigate constraints",
        required_representation="Feasibility checking",
        architecture_sufficient=True,  # Pattern matching on valid regions
        training_signal_exists=True,
        how_to_test="Complex constraint combinations",
        how_to_train="Curriculum on constraint complexity",
    ),

    PlanningCapability(
        name="Until Semantics",
        description="Maintain condition A until B happens",
        computation_type="sequential",
        example_task="(safe U goal) - stay safe until reaching goal",
        required_representation="Track progress + maintain invariant",
        architecture_sufficient=True,  # Via LTL net + reactive policy
        training_signal_exists=True,
        how_to_test="Until formulas in training",
        how_to_train="Include 'until' in curriculum",
    ),

    # ========== COUNTERFACTUAL PLANNING ==========
    PlanningCapability(
        name="Alternative Comparison",
        description="Compare outcomes of different choices",
        computation_type="recursive",
        example_task="Which blue zone leads to shorter total path?",
        required_representation="Simulate both options, compare",
        architecture_sufficient=False,  # No branching simulation
        training_signal_exists=False,
        how_to_test="Optimality test with alternatives",
        how_to_train="Would need world model + search",
    ),

    PlanningCapability(
        name="Regret Computation",
        description="Know how much worse current choice is vs optimal",
        computation_type="recursive",
        example_task="After choosing blue_1, know blue_2 was better",
        required_representation="Track counterfactual outcomes",
        architecture_sufficient=False,
        training_signal_exists=False,
        how_to_test="Probe for regret signal",
        how_to_train="Hindsight relabeling",
    ),

    # ========== PROBABILISTIC PLANNING ==========
    PlanningCapability(
        name="Risk Assessment",
        description="Assess probability of failure on different paths",
        computation_type="probabilistic",
        example_task="Path A is shorter but riskier, path B is safer",
        required_representation="Uncertainty over outcomes",
        architecture_sufficient=True,  # Can learn risk patterns
        training_signal_exists=False,  # Deterministic environments
        how_to_test="Stochastic environments with risk/reward tradeoff",
        how_to_train="Stochastic dynamics, risk-sensitive reward",
    ),

    PlanningCapability(
        name="Information Gathering",
        description="Take actions to reduce uncertainty",
        computation_type="probabilistic",
        example_task="Explore to find goal location",
        required_representation="Belief state, information value",
        architecture_sufficient=False,  # No explicit belief tracking
        training_signal_exists=False,
        how_to_test="Partially observable environments",
        how_to_train="POMDP formulation, memory architecture",
    ),

    # ========== META PLANNING ==========
    PlanningCapability(
        name="Plan Recognition",
        description="Recognize what plan is currently being executed",
        computation_type="pattern",
        example_task="Given trajectory, infer the goal",
        required_representation="Goal inference from behavior",
        architecture_sufficient=True,
        training_signal_exists=False,
        how_to_test="Give partial trajectory, predict goal",
        how_to_train="Inverse RL / goal inference auxiliary task",
    ),

    PlanningCapability(
        name="Plan Repair",
        description="Adapt plan when unexpected events occur",
        computation_type="recursive",
        example_task="Original path blocked, find alternative",
        required_representation="Replan from current state",
        architecture_sufficient=True,  # Reactive replanning works
        training_signal_exists=True,  # Dynamic environments
        how_to_test="Change environment mid-episode",
        how_to_train="Non-stationary environments",
    ),
]


def analyze_capabilities():
    """Analyze and visualize the planning capabilities."""

    print("="*80)
    print("TAXONOMY OF PLANNING CAPABILITIES")
    print("="*80)

    # Group by computation type
    by_type = {}
    for cap in PLANNING_CAPABILITIES:
        if cap.computation_type not in by_type:
            by_type[cap.computation_type] = []
        by_type[cap.computation_type].append(cap)

    # Summary table
    print("\n" + "="*80)
    print("CAPABILITY MATRIX")
    print("="*80)
    print(f"\n{'Capability':<30} {'Computation':<15} {'Arch OK?':<10} {'Training?':<10}")
    print("-"*65)

    for cap in PLANNING_CAPABILITIES:
        arch = "Yes" if cap.architecture_sufficient else "NO"
        train = "Yes" if cap.training_signal_exists else "NO"
        print(f"{cap.name:<30} {cap.computation_type:<15} {arch:<10} {train:<10}")

    # Categorize
    can_do = [c for c in PLANNING_CAPABILITIES if c.architecture_sufficient and c.training_signal_exists]
    could_learn = [c for c in PLANNING_CAPABILITIES if c.architecture_sufficient and not c.training_signal_exists]
    cannot_do = [c for c in PLANNING_CAPABILITIES if not c.architecture_sufficient]

    print("\n" + "="*80)
    print("CATEGORIZATION")
    print("="*80)

    print(f"\n✓ CAN DO (arch sufficient + training signal exists): {len(can_do)}")
    for c in can_do:
        print(f"  - {c.name}")

    print(f"\n◐ COULD LEARN (arch sufficient, needs training signal): {len(could_learn)}")
    for c in could_learn:
        print(f"  - {c.name}")
        print(f"    How to train: {c.how_to_train}")

    print(f"\n✗ CANNOT DO (architecture limitations): {len(cannot_do)}")
    for c in cannot_do:
        print(f"  - {c.name}")
        print(f"    Why: {c.computation_type} requires capabilities beyond feedforward")

    return can_do, could_learn, cannot_do


def create_visualization(can_do, could_learn, cannot_do):
    """Create visual summary."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart of categories
    ax = axes[0]
    sizes = [len(can_do), len(could_learn), len(cannot_do)]
    labels = [f'Can Do\n({len(can_do)})', f'Could Learn\n({len(could_learn)})', f'Cannot Do\n({len(cannot_do)})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_title('Planning Capabilities by Feasibility', fontsize=12, fontweight='bold')

    # Bar chart by computation type
    ax = axes[1]

    comp_types = list(set(c.computation_type for c in PLANNING_CAPABILITIES))

    can_counts = [sum(1 for c in can_do if c.computation_type == t) for t in comp_types]
    could_counts = [sum(1 for c in could_learn if c.computation_type == t) for t in comp_types]
    cannot_counts = [sum(1 for c in cannot_do if c.computation_type == t) for t in comp_types]

    x = np.arange(len(comp_types))
    width = 0.25

    ax.bar(x - width, can_counts, width, label='Can Do', color='#2ecc71')
    ax.bar(x, could_counts, width, label='Could Learn', color='#f39c12')
    ax.bar(x + width, cannot_counts, width, label='Cannot Do', color='#e74c3c')

    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_types, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Capabilities by Computation Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('planning_taxonomy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: planning_taxonomy.png")


def print_test_recommendations():
    """Print specific tests we could run."""

    print("\n" + "="*80)
    print("RECOMMENDED PROBING EXPERIMENTS")
    print("="*80)

    experiments = [
        {
            'name': '1. Chained Distance (already done)',
            'status': 'COMPLETED',
            'result': 'R²=0.36-0.51 (weak)',
            'interpretation': 'Agent does not compute optimal paths through intermediates',
        },
        {
            'name': '2. N-step Chained Distance',
            'capability': 'Chained Distance (N-step)',
            'task': 'F a & F b & F c & F d (4 sequential goals)',
            'probe': 'Can we decode optimal total path length?',
            'prediction': 'Probably weaker than 2-step case',
        },
        {
            'name': '3. Subgoal Ordering',
            'capability': 'Subgoal Ordering',
            'task': 'F a & F b & F c (unordered, agent chooses)',
            'probe': 'Does agent choose optimal ordering?',
            'prediction': 'Likely myopic (nearest first)',
        },
        {
            'name': '4. Deadline Awareness',
            'capability': 'Deadline Awareness',
            'task': 'Reach goal with varying time limits',
            'probe': 'Does value/behavior change with time pressure?',
            'prediction': 'Probably not - no time in observation',
        },
        {
            'name': '5. Disjunctive Optimality',
            'capability': 'Disjunctive Goal Selection',
            'task': '(F a | F b | F c) with varying distances',
            'probe': 'Does agent consistently pick nearest?',
            'prediction': 'Should work (simple comparison)',
        },
        {
            'name': '6. Risk-Reward Tradeoff',
            'capability': 'Risk Assessment',
            'task': 'Short risky path vs long safe path',
            'probe': 'Does agent show risk sensitivity?',
            'prediction': 'Probably not in deterministic env',
        },
        {
            'name': '7. Future State Prediction',
            'capability': 'Future State Anticipation',
            'task': 'Any navigation',
            'probe': 'Can we decode position at t+5, t+10 from current embedding?',
            'prediction': 'Weak - no explicit forward model',
        },
    ]

    for exp in experiments:
        print(f"\n{exp['name']}")
        if 'status' in exp:
            print(f"  Status: {exp['status']}")
            print(f"  Result: {exp['result']}")
            print(f"  Interpretation: {exp['interpretation']}")
        else:
            print(f"  Capability: {exp['capability']}")
            print(f"  Task: {exp['task']}")
            print(f"  Probe: {exp['probe']}")
            print(f"  Prediction: {exp['prediction']}")


def print_architecture_analysis():
    """Analyze what the architecture can and cannot do."""

    print("\n" + "="*80)
    print("ARCHITECTURE ANALYSIS")
    print("="*80)

    print("""
CURRENT ARCHITECTURE:
    obs (80d) → env_net (MLP) → env_emb (64d)
    task → ltl_net (GRU) → ltl_emb (32d)
    [env_emb, ltl_emb] → actor (MLP) → action
                       → critic (MLP) → value

WHAT IT CAN COMPUTE:
    ✓ Any function of current observation (universal approximation)
    ✓ Arithmetic: distances, sums, comparisons
    ✓ Pattern matching: blocking configurations
    ✓ Task encoding: what goals to pursue

WHAT IT CANNOT COMPUTE:
    ✗ Recursive simulation: "if I do A, then B, then C..."
    ✗ Variable-length computation: depends on problem complexity
    ✗ Explicit branching: compare N hypothetical futures
    ✗ Memory of past states: only current observation

THE KEY LIMITATION:
    The architecture computes a FIXED function of current observation.
    It cannot "think longer" for harder problems.

    Planning requires: simulate → evaluate → compare → choose
    Current arch does: observe → compute → act

    The GRU in ltl_net processes the TASK, not the STATE TRAJECTORY.
    There's no recurrence over states at decision time.

WHAT COULD HELP (without full architecture change):
    1. Auxiliary losses that force planning-relevant representations
    2. Training signal that punishes non-planning behavior
    3. Observation augmentation (e.g., include goal positions explicitly)
    4. Deeper networks (more compute per forward pass)

WHAT WOULD REQUIRE ARCHITECTURE CHANGE:
    1. Multi-step lookahead (needs world model + search)
    2. Variable computation time (needs adaptive compute)
    3. Counterfactual reasoning (needs branching simulation)
    4. Long-horizon planning (needs hierarchical structure)
""")


def main():
    can_do, could_learn, cannot_do = analyze_capabilities()
    create_visualization(can_do, could_learn, cannot_do)
    print_test_recommendations()
    print_architecture_analysis()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
Of {len(PLANNING_CAPABILITIES)} planning capabilities analyzed:

  ✓ {len(can_do)} the agent CAN DO (has capacity + training signal)
  ◐ {len(could_learn)} the agent COULD LEARN (has capacity, needs training)
  ✗ {len(cannot_do)} the agent CANNOT DO (architecture limitation)

The "could learn" category is the interesting one - these are capabilities
that are within the network's representational capacity but haven't been
learned because training didn't require them.

For the optimality planning we tested:
  - Architecture: SUFFICIENT (just arithmetic)
  - Training signal: MISSING (myopic behavior was good enough)
  - Solution: Harder maps + step penalty

For capabilities like subgoal ordering or counterfactual comparison:
  - Architecture: INSUFFICIENT (needs search/simulation)
  - Would require: World model, MCTS, or similar
""")


if __name__ == '__main__':
    main()
