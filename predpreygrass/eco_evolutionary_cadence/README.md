# Eco-Evolutionary PredPreyGrass — Cadence Variant

This module is a variant of `eco_evolutionary` in which **speed controls movement
frequency** rather than movement distance.  Every agent stays in the loop every
step — it always receives an observation and submits an action — but the action
is only executed when the agent's per-genome cooldown counter reaches zero.

The motivation: in the original `eco_evolutionary` module, the continuous genome
trait maps onto a binary outcome (slow = max distance 1, fast = max distance 2).
The cadence mechanic preserves the discrete grid and simple Moore action space
while giving the speed genome a genuinely graded effect on fitness: each speed
value maps to a distinct movement frequency across 10 cooldown levels.

## Current Scope

Active heritable trait:

- `speed`: determines how often the agent may move.  The genome value maps to a
  cooldown period in `[1, max_cooldown]`.  An agent with cooldown 1 moves every
  step; an agent with cooldown 10 moves once every 10 steps.  Speed also
  increases locomotion cost when movement executes.

At reproduction, offspring inherit the parent's genome with bounded mutation.
Founders receive genomes sampled from role-specific founder distributions.

## Cadence Mechanic

### Action Space

```text
action_range = 3
actions = 3^2 = 9
movement vectors dx, dy in {-1, 0, 1}   (Moore neighbourhood, distance ≤ 1)
```

Distance-2 moves are removed.  Speed no longer unlocks extra reach; it controls
*when* the agent can move.

### Cooldown Mapping

Each agent's genome speed maps deterministically to a cooldown period:

```text
normalized = (speed - speed_min) / (speed_max - speed_min)
cooldown   = max_cooldown - round(normalized * (max_cooldown - 1))
```

With the default `trait_bounds: speed=(0.0, 1.0)` and `max_cooldown=10`:

| speed | cooldown | moves every … steps |
|-------|----------|---------------------|
| 0.0   | 10       | 10                  |
| 0.1   | 9        | 9                   |
| 0.2   | 8        | 8                   |
| 0.3   | 7        | 7                   |
| 0.4   | 6        | 6                   |
| 0.6   | 5        | 5                   |
| 0.7   | 4        | 4                   |
| 0.8   | 3        | 3                   |
| 0.9   | 2        | 2                   |
| 1.0   | 1        | 1 (every step)      |

### Phase Offset

At birth each agent is assigned a random phase offset drawn uniformly from
`[0, cooldown)`.  This prevents all agents with the same cooldown from moving
in synchronised bursts.

### Frozen Steps

On steps where an agent's cooldown counter is > 0, its submitted action is
discarded and the agent stays in place.  The agent still:
- receives a full observation
- loses basal metabolism energy
- can be eaten (predator can still move onto it)
- ages normally

Only movement is suppressed.

## Observation Channels

The observation tensor has up to 5 channels:

| channel | content |
|---------|---------|
| 0 | predator energies in observation window |
| 1 | prey energies in observation window |
| 2 | grass energies in observation window |
| 3 | agent's normalised speed (`include_speed_in_obs: True`) |
| 4 | `move_available` flag: 1.0 if action executes this step, 0.0 otherwise |

The `move_available` channel is the key addition over the parent module.
Without it, the shared policy has no direct signal for whether this particular
step is a move step or a frozen step.  With it, the policy can learn:

- on `move_available=1` steps: choose a meaningful direction
- on `move_available=0` steps: any action is equivalent (no movement)

This eliminates gradient noise from frozen steps where the action had no effect.

## Population Carrying Capacity

```python
"energy_gain_per_step_grass": 0.04,
"max_agent_age": {"predator": None, "prey": 400},
```

These settings carry over from `eco_evolutionary` Run 2 where they proved
necessary to keep grid occupancy low enough for speed to confer a meaningful
fitness advantage.

## Genome Configuration

Key parameters in `config/config_env_eco_evolutionary.py`:

```python
"founder_genome": {
    "predator": {"speed_mean": 0.5, "speed_std": 0.1},
    "prey":     {"speed_mean": 0.5, "speed_std": 0.1},
},
"genome_mutation": {"rate": 0.05, "std": 0.05},
"trait_bounds":    {"speed": (0.0, 1.0)},
"max_cooldown":    10,
```

**Trait bounds** (`speed: (0.0, 1.0)`): the genome value is a normalised float.
`0.0` is the slowest possible agent (cooldown = `max_cooldown`); `1.0` is the
fastest (cooldown = 1, moves every step).  Because the trait is already
normalised, the speed observation channel broadcasts the raw genome value
unchanged — no rescaling needed.

**Founder distribution** (`speed_mean: 0.5, speed_std: 0.1`): founders start at
mid-range, spanning roughly 0.3–0.7, which covers cooldowns 4–7.  This gives
equal room for speed to evolve upward or downward from the starting population.

**Mutation** (`rate: 0.05, std: 0.05`): at each reproduction event there is a 5%
chance the offspring's speed is perturbed by N(0, 0.05).  With `max_cooldown=10`
a shift of ~0.11 in genome value moves the cooldown by one step, so `std=0.05`
means roughly one favourable mutation is needed per cooldown step — a moderate
selection gradient.

**Max cooldown** (`max_cooldown: 10`): sets the maximum number of steps between
moves for the slowest possible genome.  Increase for a wider behavioural range;
decrease if 10-step freezes cause slow agents to die before acting.

## Energy Cost

Movement energy cost applies only when movement actually executes:

```text
basal_loss_per_step
+ movement_cost_per_cell * actual_distance * speed^movement_speed_cost_exponent
```

Agents that are frozen on a given step incur only basal metabolism.  High-speed
agents that move every step pay the locomotion surcharge every step;
low-speed agents pay it rarely, but their reduced mobility limits food access.

## Monitoring Speed Evolution

The callback logs per-episode genome summaries under the `eco_evolution/` prefix.

| TensorBoard metric | What it shows |
|---|---|
| `eco_evolution/{species}_speed_mean` | Population mean speed |
| `eco_evolution/{species}_speed_std` | Spread — rising std means diversification; falling std means convergence |
| `eco_evolution/{species}_speed_p25/p50/p75` | Quartiles — shift in p50 indicates directional selection |
| `eco_evolution/{species}_fraction_mobile` | Share of agents with cooldown=1 (move every step) |
| `eco_evolution/{species}_cooldown_mean` | Mean cooldown across population — falling value = speed selection |
| `eco_evolution/{species}_agent_count` | Mean number of agents born per episode |
| `eco_evolution/{species}_offspring_count_mean` | Average reproductive success per agent |
| `eco_evolution/{species}_distance_traveled_mean` | Realised locomotion — confirms fast-genome agents are actually moving more |

A healthy eco-evolutionary run shows `speed_p50` drifting upward and
`cooldown_mean` drifting downward over hundreds of iterations.

## Baldwinian Enhancement

With both `include_speed_in_obs: True` and `include_move_available_in_obs: True`,
the policy has two scalar conditioning signals:

1. **Speed** — what genome value does this agent carry?  Lets the policy learn
   whether to invest in movement at all (slow agents may prefer to stay near grass).
2. **Move available** — will this step's action execute?  Lets the policy
   condition its directional choice on whether movement is possible.

Together they close the Baldwinian loop:

```text
genome → speed trait
       → cooldown period
       → policy conditions on speed + move_available
       → fast agents move purposefully, slow agents conserve energy
       → fitness differential amplified
       → stronger heritable selection on speed
```

## Experimental Treatments

The two observation flags define three controlled treatments that can be compared
directly to isolate the Baldwinian contribution:

| `include_speed_in_obs` | `include_move_available_in_obs` | Process |
|---|---|---|
| `False` | `False` | **Pure Darwinian** — speed affects fitness through movement frequency, but the policy is blind to the genome entirely |
| `True` | `False` | **Genome-aware** — policy knows its genome value and can learn speed-specific strategies, but cannot condition on whether this step executes |
| `True` | `True` | **Full Baldwinian loop** — policy conditions on both genome and movement availability (default) |

The fourth combination (`speed=False, move_available=True`) is not a useful
treatment: the policy learns when to move but identically for all agents
regardless of their speed genome, so it cannot differentiate strategies by
genome value.  The Baldwinian loop requires genome awareness as its foundation.

The scientifically interesting comparisons are:
- **Pure Darwinian vs. Genome-aware**: does knowing the genome accelerate selection?
- **Genome-aware vs. Full Baldwinian**: does the `move_available` flag add further amplification on top of genome awareness?

If `speed_p50` drifts faster and `cooldown_mean` falls more steeply at each
step up the ladder, the Baldwin effect is real and measurable in this
environment.

## Differences from `eco_evolutionary`

| | `eco_evolutionary` | `eco_evolutionary_cadence` |
|---|---|---|
| Action space | 5×5 = 25 (distance 0–2) | 3×3 = 9 (Moore, distance 0–1) |
| Speed effect | binary: dist-1 vs dist-2 | graded cooldown: 1–10 steps |
| Obs channels | 3 + 1 speed = 4 | 3 + 1 speed + 1 move_available = 5 |
| Config keys | `speed_distance_threshold`, `slow/fast_max_move_distance` | `max_cooldown` |
| TensorBoard | `fraction_fast` | `fraction_mobile`, `cooldown_mean` |

## Key Files

- `predpreygrass_rllib_env.py`: environment with cadence mechanic, genome inheritance, lineage tracking.
- `config/config_env_eco_evolutionary.py`: trait bounds, founder distributions, cooldown settings.
- `utils/genome.py`: `Genome` dataclass plus founder/mutation helpers (shared with `eco_evolutionary`).

## Quick Test

```bash
python -c "
from predpreygrass.eco_evolutionary_cadence.config.config_env_eco_evolutionary import config_env
from predpreygrass.eco_evolutionary_cadence.predpreygrass_rllib_env import PredPreyGrass
env = PredPreyGrass(config_env)
obs, _ = env.reset()
print('obs shape:', next(iter(obs.values())).shape)  # expect (5, 7, 7) for predator
"
```

## References

**Eco-evolutionary dynamics:**
- Lotka, A. J. (1925). *Elements of Physical Biology.* Williams & Wilkins. / Volterra, V. (1926). *Fluctuations in the Abundance of a Species Considered Mathematically.* Nature
- Van Valen, L. (1973). *A New Evolutionary Law.* Evolutionary Theory, 1, 1–30 — Red Queen hypothesis
- Dawkins, R. & Krebs, J. R. (1979). *Arms Races Between and Within Species.* Proceedings of the Royal Society B, 205(1161), 489–511

**Baldwin Effect:**
- Hinton, G. E. & Nowlan, S. J. (1987). *How Learning Can Guide Evolution.* Complex Systems, 1(3), 495–502

**Digital / agent-based evolution:**
- Ofria, C. & Wilke, C. O. (2004). *Avida: A Software Platform for Research in Computational Evolutionary Biology.* Artificial Life, 10(2), 191–229
- Stanley, K. O. & Miikkulainen, R. (2002). *Evolving Neural Networks Through Augmenting Topologies (NEAT).* Evolutionary Computation, 10(2), 99–127

**Life history tradeoffs:**
- Stearns, S. C. (1992). *The Evolution of Life Histories.* Oxford University Press

## Interpretation

```text
genome → speed trait → cooldown period → lifetime movement budget
       → policy learns speed-specific + move_available-conditioned behaviour
       → survival/reproduction → inherited genome with mutation
```
