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
  - `fresh_baseline/` - Standard trained agent
  - `combined_aux02_trans01/` - Agent with auxiliary losses
  - `aux_loss_*` - Various auxiliary loss experiments

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
- `interpretability/analysis/preview_optvar_maps.py` - Preview map layouts before running eval
- `interpretability/analysis/preview_opteq_maps.py` - Preview equidistant map layouts
- `interpretability/probing/probe_optvar_planning.py` - Probing for planning representations on optvar env

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

## Notes for Future Work
- The LtlOptimalityVaried environment extracts custom config keys (`layout_seed`, `intermediate_color`, `goal_color`) before passing config to parent class (which validates keys)
- When overriding registered config, must include `agent_name: 'Point'`
- Layout sampling can fail occasionally - wrap in try/except
- Zone separation constraints in optvar: min 1.0 distance between zone centers, min 0.6 from agent to zones
- The `_check_min_distances()` method in LtlOptimalityVaried enforces these constraints
