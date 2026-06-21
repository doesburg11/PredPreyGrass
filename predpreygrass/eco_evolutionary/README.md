# Eco-Evolutionary PredPreyGrass

This module is the biological-realism branch of PredPreyGrass. It starts from
`lineage_rewards` because that module already has the required lifecycle
scaffold: never-reused IDs, parent-child tracking, lineage logs, fertility caps,
age limits, and juvenile constraints.

The new layer is an explicit heritable genome. In the base experiment, the
genome controls movement speed, not learned PPO policy weights. This keeps the
default model Darwinian/Baldwinian rather than Lamarckian.

## Current Scope

Active heritable trait:

- `speed`: selects the movement band. Slow agents are clipped to distance-1
  moves; fast agents can use distance-2 moves from the shared 5x5 action space.
  Speed also increases locomotion cost.

At reproduction, offspring inherit the parent's genome with bounded mutation.
Founders receive genomes sampled from role-specific founder distributions.
Reproduction thresholds and offspring starting energy are fixed environment
parameters in the base experiment, so the first treatment isolates the speed
tradeoff.

## Open Grid Observations

The eco-evolutionary environment is an open grid without internal walls or
line-of-sight occlusion. Observations have three channels:

- predators
- prey
- grass

Grid edges are not represented by a separate wall channel. Instead, observation
windows are clipped to the valid grid coordinates, and cells outside the grid
remain zero in the returned observation tensor. Movement uses the same boundary
logic: proposed moves are clipped to stay inside the grid.

## Speed-Efficiency Tradeoff

Predators and prey use a shared extended Moore action space:

```text
action_range = 5
actions = 5^2 = 25
raw movement vectors dx,dy in {-2,-1,0,1,2}
```

The action space is shared for RLlib compatibility. The genome decides how much
of that action space is physically effective:

```text
speed < speed_distance_threshold  -> max distance 1
speed >= speed_distance_threshold -> max distance 2
```

If a slow agent chooses a distance-2 action, the vector is clipped to distance 1
in the same direction. For example, `(2, 0)` becomes `(1, 0)`.

Fast movement is not free. Energy loss is split into basal metabolism plus
locomotion cost:

```text
basal_loss_per_step
+ movement_cost_per_cell * actual_distance * speed^movement_speed_cost_exponent
```

By default, `movement_speed_cost_exponent = 2.0`, so high speed is
superlinearly expensive when the agent actually moves. Staying still still costs
basal metabolism, but it does not incur locomotion cost. This makes speed useful
for chasing, escaping, and reaching resources, while allowing slow or efficient
agents to remain competitive in some ecological conditions.

## What This Is Not

This module does not copy parent PPO weights into offspring. Learned policy
weights remain shared by policy group unless a future experiment explicitly adds
individual policy inheritance.

For policy-weight evolution, use `checkpoint_genomes` as a comparison treatment.

## Key Files

- `predpreygrass_rllib_env.py`: environment with lineage + genome inheritance.
- `config/config_env_eco_evolutionary.py`: trait bounds, founder distributions,
  mutation rate/std, and inherited lifecycle settings.
- `utils/genome.py`: `Genome` dataclass plus founder/mutation helpers.
- `tests/test_eco_evolutionary_validation.py`: lineage regression tests and
  genome inheritance tests.

## Quick Test

```bash
PYTHONPATH=src pytest -q predpreygrass/eco_evolutionary/tests/test_eco_evolutionary_validation.py
```

## Interpretation

The intended model is:

```text
genome -> speed trait -> lifetime behavior through shared PPO policy
       -> survival/reproduction -> inherited genome with mutation
```

This supports demographic selection, lineage tracking, and heritable biological
variation without claiming that acquired policy weights are biologically
inherited.
