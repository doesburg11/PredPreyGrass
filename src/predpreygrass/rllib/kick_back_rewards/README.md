# Environment Comparison: walls_occlusion vs kick_back_rewards

This document focusses on the feature changes from `walls oclussion` towards `kick_back_rewards`

## Overview: from reusing "dead" agents to using only "unique/fresh" agents

`kick_back_rewards` fixes a long standing work around in the `PredPreyGrass` repo. In the past (up until `walls_oclussion`), "dead" agent were "recreated" back to live to make reusement of non-active agents possible. The original goal was to limit the `possible_agent` (which needs to be predefinied and caps the number of possible_agents at run time). Although this increases the pool of (re)created agents susbstantially in practice,it on the othere side is not compeletely aligned with RLLib protocol. In our previous setup(s) we did not actually `terminated` an agent in the `RLLib` (and `Gymnasium`) sense. We used a activation/deactivation flag in our logic as a workaround. That meant in practice that agents not really terminated when dying, but that a "true" trajectory existed of multiple lives of different agents (of the same species obviously), stitched together with intervals of "inactivenes". That worked magically well but is not according to the strict RLlib protocol and thefore can result in strange unexpected learning behavior.

## Termination Protocol Change Notes: Purpose
- Document the behavioral difference between historical agent-slot reuse and the corrected termination handling.
- Provide guidance for interpreting metrics and evaluating experiments that span the two implementations.

## Terminology
- **Agent slot**: fixed identifier (e.g., `type_1_predator_0`) allocated to an active agent instance.
- **Lifetime**: period between an agent spawning and being marked terminated/truncated.
- **Episode return**: sum of rewards accrued during a single lifetime; reset when a termination signal is emitted.

## Historical Implementation ("Old" Approach)
- Agent slots were recycled without emitting `terminated=True` when a creature despawned.
- RLlib kept an episode open across multiple lifetimes that re-used the same slot.
- Episode returns accumulated rewards across reincarnations, inflating metrics.
- Training logs reported long episode lengths and high totals (e.g., predator return ~100) even when individuals only reproduced a few times per lifetime.
- Credit assignment for learners was noisy because rewards from new lifetimes were credited to prior hidden states.
- num_possible_agents can remain limited to the possible maximum number of active agents at one point in time, without bumping into capacity constraints in the spawning of "new" agents since "died" agents earlier on in the episode can be reused.

## Corrected Implementation ("New" Approach)
- Every lifetime ends with an explicit `terminated=True` or `truncated=True` signal before a slot is reused.
- RLlib resets episode accounting when a creature despawns; new spawns start fresh episodes.
- Episode returns now reflect rewards earned within a single lifetime (e.g., reproduction reward of 10 → return spikes proportional to average births per predator).
- Metrics align with true per-agent performance, improving credit assignment and reproducibility.
- Debug invariants guard against accidental slot reuse without termination.
- Every new spawned agent cannot have been active in the past. If it otherwise should then, after flagged terminations=True, it will produce an error at revivaval, since SingleAgentEpisode.done=True cannot be undone per RLlib protocol. This means num_possible_agents must be as large as the total number of agents ever existed during an episode, in order to not run into capcity constraints of newly spawned agents.

## Observable Differences
- **Episode Return Curves**: values dropped to realistic ranges (e.g., predators near 40 ≈ 4 reproductions per lifetime) versus inflated historical peaks (≈100) caused by reward carry-over.
- **Episode Length Metrics**: average lengths no longer grow indefinitely; they track actual lifetime durations in steps.
- **Learner Stability**: reduced variance in policy updates because hidden states are reset when lifetimes end.
- **Logging**: terminated agents appear in final observation batches, eliminating "acted then truncated" warnings.

## What Did *Not* Change
- Environment dynamics, reward magnitudes, and visual behavior in evaluation grids remain the same.
- Reproduction probability, energy budgets, and other config parameters were untouched; gameplay still matches prior intuition.


## Implications for Experiment Tracking
- Compare historical runs to new ones with caution: legacy metrics overstate per-episode returns and lengths.
- When reproducing published results that relied on the old protocol, note the bookkeeping bug and prefer reruns with the fix for accuracy.
- Hyperparameter searches using the new implementation yield more trustworthy objective signals (e.g., `score_pred`).

## Recommendations
- Treat old logs as qualitatively informative but quantitatively inflated; annotate analyses that included pre-fix data.
- For fair comparisons, rerun key baselines under the corrected termination protocol.
- Keep the debug invariants enabled when extending the environment to catch future regressions in slot management.
or Experiment Tracking
- Compare historical runs to new ones with caution: legacy metrics overstate per-episode returns and lengths.
- When reproducing published results that relied on the old protocol, note the bookkeeping bug and prefer reruns with the fix for accuracy.
- Hyperparameter searches using the new implementation yield more trustworthy objective signals (e.g., `score_pred`).


Both `walls_occlusion` and `kick_back_rewards` environments implement the same core predator-prey-grass ecosystem with walls and line-of-sight mechanics. However, they differ significantly in their **configuration philosophy**, **ID management strategy**, and **implementation maturity**.

---

## Summary Key Architectural Differences

### 0. Termination logic agents (explained above)

### 1. Configuration Access Pattern

**walls_occlusion: Defensive with Defaults**
```python
self.debug_mode = config.get("debug_mode", False)
self.max_steps = config.get("max_steps", 10000)
self.num_walls = config.get("num_walls", 20)
self.wall_placement_mode = config.get("wall_placement_mode", "random")
```
- Uses `.get()` with fallback defaults throughout
- More forgiving: works even with incomplete config dicts
- Suitable for experimentation and quick prototyping
- Safer for config evolution (missing keys won't crash)
- Nevertheless, we've removed fallback options as much as possible on purpose to minimize unexpected behavior (and unexpected parameter use) during development.

**kick_back_rewards: Strict Direct Access**
```python
self.debug_mode = config["debug_mode"]
self.max_steps = config["max_steps"]
self.manual_wall_positions = config["manual_wall_positions"]
```
- Uses direct `config[]` dictionary access
- Requires all keys to be present (KeyError if missing)
- Enforces complete configuration contracts
- Better for production: explicit about requirements

**Migration impact**: When moving from walls_occlusion to kick_back_rewards, you MUST provide ALL config keys (no defaults available).

---

### 2. Agent ID Management System

**walls_occlusion: Simple Reuse with Unique Tracking**
```python
# Reuse agent IDs within an episode, track with unique_id
self.unique_agents = {}  # Maps agent_id → unique_id
self.agent_activation_counts = {}  # Reuse counter per slot

def _register_new_agent(self, agent_id, parent_unique_id=None):
    reuse_index = self.agent_activation_counts[agent_id]
    unique_id = f"{agent_id}_{reuse_index}"
    self.unique_agents[agent_id] = unique_id
    self.agent_activation_counts[agent_id] += 1
```
- Agent IDs can be reused multiple times in same episode
- Tracked via `unique_id = agent_id_reuse_count`
- Simpler bookkeeping, less memory overhead
- Requires disambiguation in lineage tracking

**kick_back_rewards: Never-Reuse ID Pools**
```python
# Never reuse IDs within an episode, use deque pools
self.used_agent_ids = set()  # All IDs ever used this episode
self._available_id_pools = {
    "type_1_predator": deque([...]),
    "type_2_predator": deque([...]),
    "type_1_prey": deque([...]),
    "type_2_prey": deque([...])
}

def _alloc_new_id(self, species: str, type_nr: int):
    """O(1) allocation from per-type pool, never reuses."""
    key = f"type_{type_nr}_{species}"
    dq = self._available_id_pools[key]
    while dq:
        cand = dq.popleft()
        if cand not in self.used_agent_ids and cand not in self.agents:
            return cand
    return None
```
- Each agent ID used only once per episode
- Deque-based O(1) allocation per species/type
- Capacity tracking: `reproduction_blocked_due_to_capacity_predator/prey`
- Guarantees unique IDs → simpler lineage analysis
- Higher memory usage (must track all used IDs)

**Trade-offs**:
- **walls_occlusion**: More agents possible per episode (via reuse), complex lineage tracking
- **kick_back_rewards**: Simpler lineage tracking, limited by total possible agents per type

---

### 3. Wall Placement Modes

**walls_occlusion: Two Modes**
```python
self.wall_placement_mode = config.get("wall_placement_mode", "random")
self.num_walls = config.get("num_walls", 20)  # For random mode
self.manual_wall_positions = config.get("manual_wall_positions", None)  # For manual mode

# In reset():
if self.wall_placement_mode == "manual":
    # Use manual_wall_positions list
elif self.wall_placement_mode == "random":
    # Randomly sample num_walls positions
```

**kick_back_rewards: Manual Only**
```python
self.manual_wall_positions = config["manual_wall_positions"]
# No random wall generation mode
```

**Migration note**: kick_back_rewards requires explicit wall positions; no random fallback.

---

### 4. RNG Initialization

**walls_occlusion: Constructor Seeding**
```python
def _initialize_from_config(self):
    self.rng = np.random.default_rng(config.get("seed", 42))
    
def _init_reset_variables(self, seed):
    self.rng = np.random.default_rng(seed)  # Re-seed on reset
```

**kick_back_rewards: Reset-Only Seeding**
```python
def _initialize_from_config(self):
    # RNG will be initialized during reset to ensure per-episode reproducibility
    pass  # No constructor RNG
    
def _init_reset_variables(self, seed):
    if seed is None:
        seed = self.config["seed"]  # Fallback to config seed
    self.rng = np.random.default_rng(seed)
```

**Difference**: kick_back_rewards delays RNG creation until first reset, with explicit seed priority handling.

---

### 5. Additional Tracking in kick_back_rewards

**Episode-level counters** (not in walls_occlusion):
```python
self.used_agent_ids = set()  # Prevent ID reuse
self.reproduction_blocked_due_to_capacity_predator = 0
self.reproduction_blocked_due_to_capacity_prey = 0
self.spawned_predators = 0
self.spawned_prey = 0
self._printed_termination_ids = set()  # Debug print guard
```

**Purpose**: Track capacity limits and provide diagnostics for ID exhaustion scenarios.

---

### 6. Precomputed LOS Masks

**kick_back_rewards only:**
```python
def __init__(self, config=None):
    # ...
    self.los_mask_predator = self._precompute_los_mask(self.predator_obs_range)
    self.los_mask_prey = self._precompute_los_mask(self.prey_obs_range)
```

**Note**: This suggests kick_back_rewards may have optimizations for line-of-sight calculations (though `_precompute_los_mask` method implementation not visible in snippets).

---

### 7. Reproduction Cooldown Access

**walls_occlusion: Inline Default**
```python
cooldown = self.config.get("reproduction_cooldown_steps", 10)
if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
    return
```

**kick_back_rewards: Direct Access**
```python
self.agent_last_reproduction[agent_id] = -self.config["reproduction_cooldown_steps"]
```

**Consistent with overall pattern**: walls_occlusion uses defensive `.get()`, kick_back_rewards requires explicit config key.

---

## Common Features (Both Environments)

Both share the following core mechanics:

### Grid World & Movement
- Static wall obstacles (`wall_positions` set)
- Line-of-sight (LOS) visibility calculations (Bresenham algorithm)
- Movement blocking by walls
- Optional LOS-based movement restrictions (`respect_los_for_movement`)
- Optional observation masking by visibility (`mask_observation_with_visibility`)
- Optional visibility channel (`include_visibility_channel`)

### Energy & Lifecycle
- Per-step energy decay (predator/prey)
- Energy thresholds for reproduction
- Grass regeneration (`energy_gain_per_step_grass`)
- Max grass energy cap (`max_energy_grass`)

### Observation & Action
- Multi-channel observations (walls, predators, prey, grass)
- Type-specific observation ranges (predator vs prey)
- Type-specific action ranges (type_1 vs type_2)

### Agent Types & Policies
- Two predator types, two prey types
- Separate configs for possible vs initially active agents
- Mutation during reproduction (type switching)

### Tracking & Diagnostics
- Unique agent stats (distance_traveled, times_ate, energy_gained, etc.)
- Per-step agent data logging
- LOS rejection counters
- Cumulative rewards
- Death cause tracking ("eaten" vs "starved")

---

## Migration Guide

### From walls_occlusion → kick_back_rewards

**Required changes:**

1. **Complete your config dict** - remove reliance on defaults:
   ```python
   # walls_occlusion worked with:
   config = {"grid_size": 10}  # Other keys got defaults
   
   # kick_back_rewards requires:
   config = {
       "debug_mode": False,
       "max_steps": 10000,
       "grid_size": 10,
       "manual_wall_positions": [(1,1), (2,2)],
       # ... ALL keys must be present
   }
   ```

2. **Switch from random to manual walls**:## Recommendations
- Treat old logs as qualitatively informative but quantitatively inflated; annotate analyses that included pre-fix data.
- For fair comparisons, rerun key baselines under the corrected termination protocol.
- Keep the debug invariants enabled when extending the environment to catch future regressions in slot management.
   ```python
   # walls_occlusion:
   "wall_placement_mode": "random",
   "num_walls": 20
   
   # kick_back_rewards:
   "manual_wall_positions": [(x1,y1), (x2,y2), ...]  # Must specify all
   ```

3. **Understand ID capacity limits**:
   - Your episode may end earlier if ID pools exhaust
   - Monitor `reproduction_blocked_due_to_capacity_*` counters
   - Consider increasing `n_possible_type_*` config values

4. **Expect stricter error messages**:
   - Missing config keys → immediate KeyError
   - Invalid wall positions → no silent fallback

**What you gain:**
- Guaranteed unique IDs (simpler lineage tracking)
- Explicit configuration contracts (no hidden defaults)
- Capacity diagnostics (know when ID pools exhaust)
- Potential LOS optimizations (precomputed masks)

**What you lose:**
- Random wall placement convenience
- Forgiving config defaults
- Ability to reuse agent IDs within episode (higher capacity)

---

### From kick_back_rewards → walls_occlusion

**Required changes:**

1. **Add default handling in your config loaders**:
   - walls_occlusion tolerates incomplete configs
   - But best practice: still provide all keys explicitly

2. **Decide on wall placement mode**:
   ```python
   # Can now use random walls:
   "wall_placement_mode": "random",
   "num_walls": 20
   # Or keep manual:
   "wall_placement_mode": "manual",
   "manual_wall_positions": [...]
   ```

3. **Adjust for ID reuse semantics**:
   - Agent IDs may repeat (same ID, different `unique_id`)
   - Lineage tracking requires checking `unique_id` not just `agent_id`
   - Use `agent_activation_counts` to see reuse frequency

**What you gain:**
- Random wall generation (faster experimentation)
- Tolerance for incomplete configs
- Higher effective capacity (ID reuse)

**What you lose:**
- Simple 1:1 agent_id mapping (reuse complicates tracking)
- Capacity diagnostics
- Precomputed LOS masks (if applicable)

---

## Implementation Maturity

### walls_occlusion
- **Stability**: Production-ready with defensive programming
- **Flexibility**: Random walls, default configs
- **Use case**: Research prototyping, config experimentation

### kick_back_rewards
- **Stability**: Strict contracts, explicit requirements
- **Optimization**: ID pools, precomputed masks, capacity tracking
- **Use case**: Reproducible experiments, lineage studies, production training

---

## Recommendations

**Choose walls_occlusion if:**
- You want rapid prototyping with minimal config setup
- You need random wall generation for procedural environments
- You want to maximize agent capacity via ID reuse
- You prefer forgiving defaults during development

**Choose kick_back_rewards if:**
- You need strict reproducibility guarantees
- You're analyzing lineage/genealogy (simpler with unique IDs)
- You want explicit capacity limits and diagnostics
- You prefer explicit contracts over implicit defaults
- You need LOS computation optimizations

**Hybrid approach:**
- Start with walls_occlusion for rapid iteration
- Migrate to kick_back_rewards for final experiments
- Use walls_occlusion's random walls to generate fixed `manual_wall_positions` for kick_back_rewards

---

## Future Unification

Potential merged features:
- **Config flexibility**: Support both `.get()` defaults AND strict mode (via flag)
- **Wall modes**: Add random generation to kick_back_rewards
- **ID strategy**: Make ID reuse vs never-reuse a configurable option
- **Hybrid capacity**: Allow ID reuse with optional capacity cap

---

## File Locations

- **walls_occlusion**: `src/predpreygrass/rllib/walls_occlusion/predpreygrass_rllib_env.py`
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_1.py`

---

*Document created: 2025-12-03*

## Further feature changes
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_3.py`

-kinship rewards for parents when offspring survive time steps

- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_4.py`
-kinship rewards for parents when offspring survive time steps
-integrated kinship rewards into config_env_rllib_termination.py
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_5.py`
-adjust the kinship rewards for parents when offspring succeeds in 
producing offspring themseleves (instead of only surviving time steps (works_4))
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_6.py`
- simplify config options
- remove unused rewards
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_7.py`
 Limited intake per step:
 - Predator–prey: each predator can only consume up to a fixed
   energy bite from a caught prey per step (bite = min(prey_energy,
   max_energy_gain_per_prey)); any remaining prey energy stays on the
   prey so it can be bitten again later instead of being fully removed
   in one go.
 - If bite >= prey_energy, the prey is fully caught and removed as usual.
 - Prey–grass: each prey can only consume up to a fixed energy bite
   from grass per step (bite = min(grass_energy,
   max_energy_gain_per_grass)); any remaining grass energy stays on
   the patch and continues to regrow, enabling multiple partial
   grazings over time.
 - If bite >= grass_energy, the grass is fully eaten and its energy is
   reset to zero as usual, with regeneration starting in the next step.
- **kick_back_rewards**: `src/predpreygrass/rllib/kick_back_rewards/predpreygrass_rllib_env_works_8.py`
 - Dead-prey carcasses:
   * When a prey is first bitten but still has remaining energy,
     it becomes "dead" (added to dead_prey) but is not immediately
     terminated. It acts as a static carcass:
       - cannot move,
       - cannot eat grass,
       - cannot reproduce,
       - cannot receive kin kickback rewards,
       - cannot age further
   * Predators can continue to take limited-intake bites from a
     dead prey’s remaining energy until it reaches zero, after
     which the prey is fully removed as usual.


# Kin Kickback Rewards in `kick_back_rewards`

This document describes the "kin kickback" reward mechanism added to the `PredPreyGrass` environment in the `kick_back_rewards` variant.

## Overview

The goal of kin kickbacks is to track and optionally reward parents based on the survival and activity of their offspring. This allows analysis of lineage success and, in future variants, could support experiments on kin-based altruism.

Key ideas:

- Each agent keeps a counter `kin_kickbacks` in its lifetime statistics.
- On each step, if a child is alive and its parent is also alive, the parent may receive a small reward increment (a "kickback").
- These kickbacks are accumulated into both the per-step reward stream and the parent’s `cumulative_reward`.
- At the end of an episode, `kin_kickbacks` is written into the agent’s final record and exported in evaluation summaries.

## Configuration

The environment reads kin kickback magnitudes from the config:

- `kin_kick_back_reward_predator`
- `kin_kick_back_reward_prey`

These are exposed as:

```python
self.kin_kick_back_predator_config = config["kin_kick_back_reward_predator"]
self.kin_kick_back_prey_config = config["kin_kick_back_reward_prey"]
```

They can be scalars or type-specific dicts, analogous to other reward dictionaries. A value of `0.0` disables kin kickback rewards (counters will still be tracked if the logic is active).

## Environment Logic

### Tracking parent–child relations

When a new agent is created via reproduction (predator or prey):

- The parent’s `agent_id` is recorded in `self.agent_parents[child_id]`.
- The parent’s offspring bookkeeping is updated:
  - `self.agent_offspring_counts[parent] += 1`
  - `self.agent_live_offspring_ids[parent].append(child_id)`
- The child is registered via `_register_new_agent`, which initializes:

```python
self.agent_stats_live[agent_id] = {
    ...
    "parent": parent_agent_id,
    "offspring_count": 0,
    "offspring_ids": self.agent_live_offspring_ids[agent_id],
    ...
    "cumulative_reward": 0.0,
    "kin_kickbacks": 0,
}
```

### Per-step kin survival reward

In `step`, after engagements and reproduction and after assembling default reward/termination/truncation dictionaries, the environment applies the kin survival reward:

- For each active agent `child` in `self.agents`:
  - Look up its parent: `parent = self.agent_parents.get(child)`.
  - If `parent is not None` and `parent` is still in the set of alive agents:
    - Add a kin reward to the parent’s per-step reward:
      - `self.rewards[parent] = self.rewards.get(parent, 0.0) + kin_reward`
    - Update the parent’s live record:
      - `parent_record["cumulative_reward"] += kin_reward`
      - `parent_record["kin_kickbacks"] = parent_record.get("kin_kickbacks", 0) + 1`

The actual `kin_reward` scalar can be chosen per species (e.g. via `kin_kick_back_reward_predator` / `kin_kick_back_reward_prey`) and is intended to be small relative to primary task rewards.

This logic means:

- `kin_kickbacks` counts **how many per-step kin rewards** a parent received during its lifetime.
- `cumulative_reward` includes both normal task rewards (eating, reproduction, step rewards) and kin survival rewards.

### Finalization and persistence

At agent death or episode truncation, `_finalize_agent_record` moves the live record into `agent_stats_completed` and ensures `kin_kickbacks` is preserved:

```python
record = self.agent_stats_live.pop(agent_id, None)
if record is None:
    return
...
record["kin_kickbacks"] = record.get("kin_kickbacks", 0)
self.agent_stats_completed[agent_id] = record
```

`get_all_agent_stats()` then exposes both live and completed records with `kin_kickbacks` included.

## Evaluation Outputs

The kin kickback statistics are surfaced in the evaluation/debug scripts under `kick_back_rewards`:

- `evaluate_ppo_from_checkpoint_debug.py` uses `env.get_all_agent_stats()` to build per-agent stats.
- For each agent, it records:
  - `cumulative_reward`
  - `lifetime`
  - `offspring_count`
  - `kin_kickbacks`
- These are written to:
  - `agent_fitness_stats.json` – detailed per-agent records.
  - `reward_summary.txt` – human-readable ranked fitness summary, including a `KinKick` column.

There is also a dedicated analysis script:

- `test/analyze_kin_rewards.py`
  - Loads `agent_fitness_stats.json`.
  - Groups agents by policy group (e.g. `type_1_predator`).
  - Computes correlations between `kin_kickbacks`, offspring counts, and cumulative rewards.
  - Compares agents with `kin_kickbacks > 0` vs `kin_kickbacks == 0`.

This allows you to quantify how strongly kin survival rewards and lineage success are aligned in a given experiment.

## Usage Notes and Experiments

- To **disable** kin rewards but still track lineage structure:
  - Set `kin_kick_back_reward_predator = 0.0` and `kin_kick_back_reward_prey = 0.0` in the environment config.
  - `kin_kickbacks` counters will still be updated by the step logic (if enabled), but they will contribute `0` to rewards.
- To **enable** kin survival reward shaping:
  - Choose small positive values (e.g. `0.01` or `0.1`) relative to main reproduction/foraging rewards to avoid dominating the learning signal.
- For evolutionary comparisons:
  - Run a baseline experiment with kin rewards disabled.
  - Run a matched experiment with the same config but non-zero kin rewards.
  - Compare the outputs of `analyze_kin_rewards.py` across runs.

## Limitations and Future Directions

- The current kin kickback mechanism rewards parents whenever children are alive, but it does not add new actions (e.g., energy transfer) or explicit kin markers in observations.
- As a result, it primarily reinforces lineages that are already successful rather than creating qualitatively new "altruistic" behaviors by itself.
- Future extensions may add:
  - Explicit energy-sharing actions between kin.
  - Limited per-step resource intake and carcass mechanics to create sharing opportunities.
  - Additional shaping terms tied to offspring survival milestones or descendants’ reproduction events.

This file should be kept in sync with any changes to the kin reward logic in `predpreygrass_rllib_env.py` and the related evaluation scripts.
