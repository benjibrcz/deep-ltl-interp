# Deep-LTL Project Memory

## Project Overview
This is a research project studying **deep reinforcement learning agents trained on Linear Temporal Logic (LTL) tasks**. The focus is on interpretability research to understand how these agents represent and process temporal goals.

## Key Directories

### Source Code
- `src/` - Main source code
  - `src/envs/zones/safety-gymnasium/` - Modified Safety Gymnasium with LTL tasks
  - `src/model/` - Neural network models (LTLNet)
  - `src/sequence/samplers/` - Task/curriculum samplers

### Interpretability Research
- `interpretability/zone_env/` - Zone environment analysis
  - `working_scripts/` - Analysis scripts (probing, logging, etc.)
  - `results/` - Experiment results
  - `datasets/` - Collected data
- `interpretability/letter_world/` - Letter world analysis
- `interpretability/analysis/` - Cross-environment analysis scripts

### Experiments
- `experiments/ppo/PointLtl2-v0/` - Trained models
  - `fresh_baseline/` - Standard trained agent (15M steps, discount=0.998) - **91% success, 58% optimal**
  - `combined_aux02_trans01/` - Agent with auxiliary losses
  - `aux_loss_*` - Various auxiliary loss experiments
  - `extended_baseline/` - Extended training (30M steps, discount=0.998) - **95% success, 59% optimal**
  - `twostep_lowdiscount/` - 2-step curriculum, d=0.95 - **38% success** (too aggressive)
  - `opt_d095_mixed/` - Mixed curriculum (75% 2-step), d=0.95 - **64% success** (unstable)
  - `opt_d099_mixed/` - Mixed curriculum, d=0.99 - **85% success, 52% optimal** (no planning)

## Environment Setup

### Python Environment
```bash
# Use the project's virtual environment
/Users/benji.berczi/Documents/deep-ltl-fresh/.venv/bin/python <script.py>
```

### Key Environments
- `PointLtl2-v0` - Standard zone environment (2 zones of each color: blue, green, yellow, magenta)
- `PointLtl2-v0.fixed` - Fixed layout for reproducible testing
- `PointLtl2-v0.optvar` - **Custom optimality test environment**:
  - 2 zones of intermediate color
  - 1 zone of goal color
  - 2 distractor zones
  - Generates layouts where optimal ≠ myopic by construction
- `PointLtl2-v0.opteq` - **Equidistant optimality test environment**:
  - Same structure as optvar
  - Both intermediate zones at SAME distance from agent
  - Only way to choose optimally is to consider full path to goal

### Creating Custom Optvar/Opteq Environments
```python
from safety_gymnasium.utils.registration import make

config = {
    'agent_name': 'Point',  # Required when overriding config
    'intermediate_color': 'blue',  # or 'green', 'yellow', 'magenta'
    'goal_color': 'green',
    'layout_seed': 42,  # Vary this for different layouts
}
# For standard optvar (myopic closer than optimal):
env = make('PointLtl2-v0.optvar', config=config)

# For equidistant (both intermediates same distance from agent):
env = make('PointLtl2-v0.opteq', config=config)
```

## Key Findings

### Optimality Analysis (Jan 2025)
Tested whether agents choose optimal vs myopic paths on F(intermediate) THEN F(goal) tasks using the optvar environment with proper zone separation (min 1.0 distance between zones):
- **fresh_baseline**: ~50% optimal, ~50% myopic, 93% success
- **combined_aux02_trans01**: ~49% optimal, ~51% myopic, 75% success
- **Both agents are approximately random** between optimal and myopic choices
- This contradicts paper claims about optimal planning behavior

### Probing for Planning Representations (Jan 2025)
Probed GRU hidden states for planning-related features on optvar environment:

| Feature | Description | R² Score |
|---------|-------------|----------|
| d_agent_to_int | Distance from agent to each intermediate zone | 0.43-0.54 |
| d_int_to_goal | Distance from intermediate zone to goal | **0.08-0.18** |
| total_via_int | Total path length (agent→int→goal) | 0.37-0.54 |
| optimality_gap | Difference between myopic and optimal totals | 0.15-0.25 |

**Key insight**: The model does NOT encode chained distances (d_int_to_goal has R² ~0.1-0.2). This explains myopic behavior - the model only "sees" immediate agent-to-zone distances, not multi-step path costs. It cannot compute optimal paths because it lacks the necessary planning-relevant information.

### Equidistant Optimality Test (Jan 2025)
When both intermediate zones are at the SAME distance from the agent (removing the "myopic" cue), the agent must consider full path length to choose correctly:
- **fresh_baseline**: 54% optimal, 46% suboptimal (93% success)
- **combined_aux02_trans01**: 53% optimal, 45% suboptimal (78% success)
- **Both agents are essentially RANDOM** (~50/50) when the distance cue is removed
- This confirms the model does NOT compute multi-step paths; it relies entirely on the "closest intermediate" heuristic

### Spatial Bias Analysis (Jan 2025)
The ~68% "chose empirically easier" rate is NOT evidence of planning - it's confounded by spatial bias:
- **fresh_baseline**: LEFT bias (66% chose x<0)
- **combined_aux02_trans01**: RIGHT bias (61% chose x>0)
- After controlling for spatial position, goal direction has NO effect (p=0.49)
- Different models learn different arbitrary biases - neither is planning
- See `interpretability/results/empirical_difficulty/ANALYSIS_SUMMARY.md`

### Orientation Bias Analysis (Jan 2025)
**KEY FINDING**: The "spatial bias" is actually an **orientation bias** - the agent prefers forward motion.
- **Forward preference: 73.7%** (p < 0.0001) - agent goes in direction it's initially facing
- When only one zone is forward, agent chose it **79.5%** of the time
- The LEFT/RIGHT bias (56.8%) is much weaker and explained by heading direction
- See `interpretability/analysis/analyze_orientation_bias.py`

### Controlled Orientation Test (Jan 2025)
When controlling for orientation bias (agent faces midpoint between zones):
- **Optimal choice rate: 58.3%** (95% CI: [48.3%, 67.7%]) - NOT significant (p=0.125)
- **Spatial bias eliminated**: LEFT 50% / RIGHT 50%
- **Conclusion**: Without forward bias, agent shows NO planning - essentially random
- See `interpretability/analysis/optimality_test_controlled_orientation.py`

### DeepLTL Author Insights (Jan 2025)
The author confirmed ~50% optimality rate and identified likely causes:
1. **High discount (0.998)**: Return differences between optimal/suboptimal are minimal
2. **Curriculum bias**: Starting with 1-step reach biases toward "nearest zone" heuristic
3. **Agent orientation**: Forward motion preference may affect zone choice

### Curriculum & Discount Intervention Results (Jan 2025)
Tested whether curriculum and discount changes could induce planning (based on author's suggestions):

| Experiment | Discount | Curriculum | Task Success | Optimal Choice |
|------------|----------|------------|--------------|----------------|
| fresh_baseline | 0.998 | 1-step start | 91% | 58% (p=0.125) |
| extended_baseline | 0.998 | 1-step start (30M) | 95% | 59% (p=0.093) |
| twostep_lowdiscount | 0.95 | 2-step only | 38% | - (too low) |
| opt_d095_mixed | 0.95 | 75% 2-step + 25% 1-step | 64% | - |
| **opt_d099_mixed** | **0.99** | **mixed** | **85%** | **52% (p=0.764)** |

**KEY FINDING**: Curriculum and discount interventions do NOT improve planning.
- The opt_d099_mixed model achieves 85% task success but shows **52% optimal choice** - indistinguishable from random (p=0.764)
- Extended training (30M steps) doesn't help either
- The agent finds heuristic solutions regardless of training curriculum
- See `training_curves.png` and `training_curves_mixed.png` for learning curves

### Paper Evaluation Results
fresh_baseline performance on paper specifications (φ6-φ11):
- φ6: 97% success
- φ7: 96.6% success
- φ8: 98.2% success
- φ9: 91.6% success
- φ10: 96.4% success
- φ11: 99% success

## Important Files

### Analysis Scripts
- `interpretability/analysis/optimality_test_clean.py` - Optimality analysis with varied maps/colors
- `interpretability/analysis/optimality_test_equidistant.py` - Equidistant optimality test
- `interpretability/analysis/analyze_empirical_difficulty.py` - Spatial bias and confound analysis
- `interpretability/analysis/analyze_orientation_bias.py` - **Orientation (forward) bias analysis**
- `interpretability/analysis/optimality_test_controlled_orientation.py` - **Optimality test with controlled orientation**
- `interpretability/analysis/preview_optvar_maps.py` - Preview map layouts before running eval
- `interpretability/analysis/preview_opteq_maps.py` - Preview equidistant map layouts
- `interpretability/probing/probe_optvar_planning.py` - Probing for planning representations on optvar env

### Training Scripts
- `run_zones.py` - Standard training script (15M steps, discount=0.998)
- `run_optimality_sweep.py` - Sweep script for optimality experiments

### Environment Registration
- `src/envs/zones/safety-gymnasium/safety_gymnasium/__init__.py` - Environment registration
- `src/envs/zones/safety-gymnasium/safety_gymnasium/tasks/__init__.py` - Task class imports
- `src/envs/zones/safety-gymnasium/safety_gymnasium/utils/task_utils.py` - Task ID → class name mapping

### Custom Environments
- `src/envs/zones/safety-gymnasium/safety_gymnasium/tasks/ltl/ltl_optimality_varied.py` - LtlOptimalityVaried class
- `src/envs/zones/safety-gymnasium/safety_gymnasium/tasks/ltl/ltl_optimality_equidistant.py` - LtlOptimalityEquidistant class

## Common Patterns

### Loading a Model
```python
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from envs import make_env

exp_name = 'fresh_baseline'
env_name = 'PointLtl2-v0'

temp_sampler = CurriculumSampler.partial(curricula[env_name])
temp_env = make_env(env_name, temp_sampler, sequence=True)

config = model_configs[env_name]
model_store = ModelStore(env_name, exp_name, seed=0)
training_status = model_store.load_training_status(map_location='cpu')
model_store.load_vocab()

model = build_model(temp_env, training_status, config)
model.eval()
propositions = list(temp_env.get_propositions())
```

### Creating LTL Tasks
```python
from ltl.automata import LDBASequence
from ltl.logic import Assignment

# F(blue) THEN F(green)
int_reach = frozenset([Assignment.single_proposition('blue', propositions).to_frozen()])
goal_reach = frozenset([Assignment.single_proposition('green', propositions).to_frozen()])
task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])
```

### Getting Agent Position and Zone Positions
```python
def get_unwrapped(env):
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped

unwrapped = get_unwrapped(env)
agent_pos = unwrapped.agent_pos[:2]
zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}
```

## Visualization

### Built-in Zone Visualization (`src/visualize/zones.py`)
Use this for paper-quality trajectory plots:

```python
from visualize.zones import draw_zones, draw_path, draw_diamond, setup_axis, FancyAxes
from matplotlib import projections

# Register the fancy axes projection
projections.register_projection(FancyAxes)

# Create subplot with rounded corners and grid
ax = fig.add_subplot(1, 1, 1, projection='fancy_box_axes', edgecolor='gray', linewidth=0.5)
setup_axis(ax)  # Sets up grid, limits, clean look

# Draw zones (uses paper's color palette)
draw_zones(ax, zone_positions)  # zone_positions = {'blue_zone_0': [x, y], ...}

# Draw start position (orange diamond)
draw_diamond(ax, agent_pos, color='orange')

# Draw trajectory
draw_path(ax, trajectory_points, color='#4caf50', linewidth=3, style='solid')
# style can be: 'solid', 'dashed', 'dotted'
```

Color palette (matches paper):
- Blue: `#2196f3`
- Green: `#4caf50`
- Yellow: `#fdd835`
- Magenta: `violet`

## Optimality Training Experiments

### Curriculum Variants
Three curriculum variants to test the "nearest zone bias" hypothesis:

| Variant | Curriculum Key | Description |
|---------|----------------|-------------|
| **V0: baseline** | `PointLtl2-v0` | Original (starts with 1-step reach) |
| **V1: twostep** | `PointLtl2-v0.twostep` | Starts with 2-step only (no 1-step bias) |
| **V2: mixed** | `PointLtl2-v0.mixed` | 75% 2-step + 25% 1-step (stability) |

### Sweep Script
```bash
# Single experiment with specific settings
python run_optimality_sweep.py --discount 0.95 --curriculum twostep

# Full sweep: 5 discounts × 3 curricula = 15 experiments
python run_optimality_sweep.py --sweep

# Sweep discount only (with fixed curriculum)
python run_optimality_sweep.py --sweep_discount --curriculum twostep

# Dry run to preview
python run_optimality_sweep.py --sweep --dry_run
```

### Discount Factor Values
Sweep values: {0.94, 0.97, 0.99, 0.995, 0.998}
- At γ=0.998: 100-step difference → return ratio ~0.82 (minimal)
- At γ=0.95: 100-step difference → return ratio ~0.006 (strong signal)

### Key Training Flags
- `--discount`: Discount factor (default 0.998)
- `--curriculum`: Curriculum key (baseline, twostep, mixed)
- `--entropy_coef`: Entropy regularization (default 0.003)
- `--lr`: Learning rate (default 0.0003)

### Documentation
- Full experiment documentation: `interpretability/zone_env/OPTIMALITY_EXPERIMENTS.md`
- Spatial bias analysis: `interpretability/results/empirical_difficulty/ANALYSIS_SUMMARY.md`

## Notes for Future Work
- The LtlOptimalityVaried environment extracts custom config keys (`layout_seed`, `intermediate_color`, `goal_color`) before passing config to parent class (which validates keys)
- When overriding registered config, must include `agent_name: 'Point'`
- Layout sampling can fail occasionally - wrap in try/except
- Zone separation constraints in optvar: min 1.0 distance between zone centers, min 0.6 from agent to zones
- The `_check_min_distances()` method in LtlOptimalityVaried enforces these constraints
- **Agent orientation**: Consider logging heading direction during rollouts to check forward-motion bias
