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

## Population Carrying Capacity

Two parameters control grid crowding and generational turnover:

```python
"energy_gain_per_step_grass": 0.04,   # halved from 0.08
"max_agent_age": {"predator": None, "prey": 400},
```

**Grass regrowth** (`0.04`): halving the food supply limits how many prey the
grid can sustain. Without this, prey populations grew to 100+ agents on a 25×25
grid (~18% occupancy by prey alone), leaving little space for movement and
eroding the fitness advantage of high speed.

**Maximum prey age** (`400` steps): prey that reach this age are auto-terminated
regardless of energy. This forces generational turnover — old agents vacate
slots, offspring with mutated genomes replace them, and the genome pool refreshes
at a faster rate. Without a lifespan cap, long-lived individuals hold slots
without contributing additional reproduction, slowing the evolutionary signal.
The two changes together keep grid occupancy low enough for speed to confer a
meaningful movement advantage.

## Genome Configuration

Key parameters in `config/config_env_eco_evolutionary.py`:

```python
"founder_genome": {
    "predator": {"speed_mean": 1.0, "speed_std": 0.2},
    "prey":     {"speed_mean": 1.0, "speed_std": 0.2},
},
"genome_mutation": {"rate": 0.05, "std": 0.1},
"trait_bounds":    {"speed": (0.5, 2.0)},
"speed_distance_threshold": 1.5,
```

**Founder distribution** (`speed_std: 0.2`): each founder's speed is sampled from
N(1.0, 0.2) clipped to [0.5, 2.0], so the initial population spans roughly 0.6–1.4.
Setting `speed_std` to 0.0 removes all initial genetic diversity; evolution would then
depend entirely on mutation accumulation, which takes many hundreds of iterations to
produce a measurable signal.

**Mutation** (`rate: 0.05, std: 0.1`): at each reproduction event there is a 5% chance
the offspring's speed is perturbed by a draw from N(0, 0.1). With `std: 0.1`, reaching
the fast threshold (1.5) from the population mean (1.0) requires roughly 5 consecutive
favourable mutations in an unbroken lineage. With the previous value of `std: 0.03`
that required ~17, making the threshold signal essentially invisible during training.

**Threshold** (`speed_distance_threshold: 1.5`): agents above this value gain access to
distance-2 moves. `fraction_fast` in TensorBoard tracks the share of all agents born in
the episode that exceed this threshold.

## Monitoring Speed Evolution

The callback logs per-episode genome summaries under the `eco_evolution/` prefix.
Speed metrics are computed over **all agents that lived in the episode** (both
completed and still-active at episode end), so they reflect the full generational
sample rather than just the survivors.

| TensorBoard metric | What it shows |
|---|---|
| `eco_evolution/{species}_speed_mean` | Population mean speed |
| `eco_evolution/{species}_speed_std` | Spread — rising std means diversification; falling std means convergence |
| `eco_evolution/{species}_speed_p25/p50/p75` | Quartiles — shift in p50 indicates directional selection |
| `eco_evolution/{species}_fraction_fast` | Share of agents with speed ≥ threshold; non-zero once fast agents reproduce |
| `eco_evolution/{species}_agent_count` | Mean number of agents born per episode (averaged across workers) |
| `eco_evolution/{species}_offspring_count_mean` | Average reproductive success per agent |
| `eco_evolution/{species}_distance_traveled_mean` | Realised locomotion — distinguishes fast-genome from fast-moving agents |

A healthy eco-evolutionary run shows `speed_p50` drifting over hundreds of iterations
and `fraction_fast` eventually rising if fast agents reproduce more successfully than
slow ones.

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
