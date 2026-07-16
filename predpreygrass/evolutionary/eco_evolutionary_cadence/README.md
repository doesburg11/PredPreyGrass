# Eco-Evolutionary PredPreyGrass — Cadence Variant

This module is a variant of `eco_evolutionary` in which **speed controls movement
frequency** rather than movement distance.  Every agent stays in the loop every
step — it always receives an observation and submits an action — but the action
is only executed once the agent's continuous move accumulator reaches 1.0.

The motivation: in the original `eco_evolutionary` module, the continuous genome
trait maps onto a binary outcome (slow = max distance 1, fast = max distance 2).
The cadence mechanic preserves the discrete grid and simple Moore action space
while giving the speed genome a genuinely graded effect on fitness.

**Revision note:** the first version of this module mapped speed to one of 10
integer cooldown levels via `round()`. In practice this still produced flat
plateaus — any two genome values rounding to the same cooldown were
phenotypically identical, so most mutations had zero effect on fitness. Real
training runs showed weak or absent selection on speed as a result (see
`predpreygrass/eco_evolutionary_investment/README.md`, which deliberately
built on `eco_evolutionary` rather than this module for exactly that reason).
The mechanic below replaces the rounded lookup table with a continuous
accumulator so every genome value produces a distinct, measurable movement
frequency.

## Current Scope

Active heritable trait:

- `speed`: determines how often the agent may move. The genome value maps
  *linearly, with no rounding*, to a per-step move rate in
  `[1/max_cooldown, 1.0]`. Each agent carries a move accumulator that
  increases by its rate every step; movement executes once the accumulator
  reaches 1.0 (which is then decremented by 1.0). An agent at the fastest
  possible rate (1.0) moves every step; an agent at the slowest possible rate
  (`1/max_cooldown`) moves on average once every `max_cooldown` steps — but
  unlike a rounded cooldown, every value in between yields a distinct,
  continuously-varying long-run movement frequency. Speed also increases
  locomotion cost when movement executes.

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

### Move-Rate Mapping and Accumulator

Each agent's genome speed maps deterministically to a per-step move rate — a
direct linear map, no rounding:

```text
normalized = (speed - speed_min) / (speed_max - speed_min)
min_rate   = 1 / max_cooldown
move_rate  = min_rate + normalized * (1.0 - min_rate)
```

Each agent carries a continuous `move_accumulator`. Every step:

```text
accumulator += move_rate
if accumulator >= 1.0:
    execute movement; accumulator -= 1.0
else:
    stay frozen this step
```

With the default `trait_bounds: speed=(0.0, 1.0)` and `max_cooldown=10`, this
gives a continuous range of realised long-run move frequencies rather than 10
discrete levels — a handful of illustrative points:

| speed | move_rate | moves every … steps (on average) |
|-------|-----------|-----------------------------------|
| 0.0   | 0.10      | 10                                |
| 0.25  | 0.325     | ≈3.08                             |
| 0.5   | 0.55      | ≈1.82                             |
| 0.75  | 0.775     | ≈1.29                             |
| 1.0   | 1.00      | 1 (every step)                    |

Unlike the earlier rounded-cooldown version, values *between* these points
(e.g. speed=0.51 vs. 0.52) each produce their own distinct move rate — there
is no plateau where a mutation has zero phenotypic effect.

### Phase Offset

At birth each agent's `move_accumulator` is initialised to a random value
drawn uniformly from `[0.0, 1.0)`.  This prevents all agents with the same
move rate from moving in synchronised bursts.

### Frozen Steps

On steps where an agent's accumulator has not yet reached 1.0, its submitted
action is discarded and the agent stays in place.  The agent still:
- receives a full observation
- loses basal metabolism energy
- can be eaten (predator can still move onto it)
- ages normally

Only movement is suppressed.

## Observation Channels

The RLlib observation is a dictionary with a spatial tensor and an action mask:

| key | content |
|---|---|
| `observations` | spatial tensor with environment channels plus optional speed channel |
| `action_mask` | binary vector over the 9 actions; frozen agents can only choose stay |

With the default `include_speed_in_obs: True`, the spatial tensor has 4 channels:

| channel | content |
|---------|---------|
| 0 | predator energies in observation window |
| 1 | prey energies in observation window |
| 2 | grass energies in observation window |
| 3 | agent's normalised speed (`include_speed_in_obs: True`) |

There is no separate spatial `move_available` channel in the current
implementation. Movement availability is represented by `action_mask` instead.
When an agent's accumulator has not yet reached 1.0, every action except stay
is masked out. Once it will reach 1.0 this step, all 9 Moore-neighbourhood
actions are available.

This prevents the policy from sampling actions that would be discarded on frozen
steps and removes gradient noise from movement choices that cannot execute.

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
`0.0` is the slowest possible agent (move rate = `1/max_cooldown`); `1.0` is
the fastest (move rate = 1.0, moves every step).  Because the trait is
already normalised, the speed observation channel broadcasts the raw genome
value unchanged — no rescaling needed.

**Founder distribution** (`speed_mean: 0.5, speed_std: 0.1`): founders start at
mid-range, spanning roughly 0.3–0.7, which covers move rates of roughly
0.37–0.73 (expected cooldowns of ~1.4–2.7 steps). This gives equal room for
speed to evolve upward or downward from the starting population.

**Mutation** (`rate: 0.05, std: 0.05`): at each reproduction event there is a 5%
chance the offspring's speed is perturbed by N(0, 0.05). Because the move rate
is a direct linear function of the genome value with no rounding, every such
mutation produces a nonzero, measurable change in realised movement frequency
— there is no minimum mutation size needed to "cross a step" as there was
under the earlier rounded-cooldown mechanic.

**Max cooldown** (`max_cooldown: 10`): sets the slowest possible genome's move
rate (`1/max_cooldown`), i.e. the average number of steps between moves for
`speed=0.0`. Increase for a wider behavioural range; decrease if long freezes
cause slow agents to die before acting.

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
| `eco_evolution/{species}_fraction_mobile` | Share of agents at the literal fastest rate (speed=1.0, move every step) — a narrow edge case now that rates aren't rounded into a wide top bucket |
| `eco_evolution/{species}_cooldown_mean` | Mean *expected* cooldown (1/move_rate, continuous, unrounded) across population — falling value = speed selection |
| `eco_evolution/{species}_agent_count` | Mean number of agents born per episode |
| `eco_evolution/{species}_offspring_count_mean` | Average reproductive success per agent |
| `eco_evolution/{species}_distance_traveled_mean` | Realised locomotion — confirms fast-genome agents are actually moving more |

A healthy eco-evolutionary run shows `speed_p50` drifting upward and
`cooldown_mean` drifting downward over hundreds of iterations.

## Baldwinian Enhancement

With `include_speed_in_obs: True`, the policy sees the heritable speed genome as
a scalar spatial channel. Separately, the action mask tells RLlib which actions
are valid on the current step.

1. **Speed channel** — what genome value does this agent carry? Lets the policy
   learn speed-specific strategies, such as fast agents ranging farther and slow
   agents conserving energy near resources.
2. **Action mask** — is movement available this step? Lets the policy avoid
   learning from impossible movement actions on frozen steps.

Together they close the Baldwinian loop:

```text
genome → speed trait
       → continuous move rate → accumulator
       → policy conditions on speed while action masking handles frozen steps
       → fast agents move purposefully, slow agents conserve energy
       → fitness differential amplified
       → stronger heritable selection on speed
```

## Experimental Treatments

The current implementation exposes one experimental observation flag:

| `include_speed_in_obs` | Process |
|---|---|
| `False` | **Pure Darwinian** — speed affects fitness through movement frequency, but the policy is blind to the genome |
| `True` | **Genome-aware / Baldwinian** — policy knows its speed genome and can learn speed-specific strategies |

The action mask is currently always part of the observation API for this module.
It is an implementation device for valid actions rather than a biological genome
signal. It should be held constant across treatments.

The scientifically interesting comparison is:
- **Pure Darwinian vs. Genome-aware**: does knowing the genome accelerate or
  redirect selection?

If `speed_p50` drifts faster and `cooldown_mean` falls more steeply at each
step in the genome-aware treatment, the Baldwin effect is real and measurable in
this environment.

## Differences from `eco_evolutionary`

| | `eco_evolutionary` | `eco_evolutionary_cadence` |
|---|---|---|
| Action space | 5×5 = 25 (distance 0–2) | 3×3 = 9 (Moore, distance 0–1) |
| Speed effect | binary: dist-1 vs dist-2 | continuous move rate via accumulator (no rounding) |
| Observation API | spatial tensor | dict: spatial tensor + action mask |
| Spatial channels | 3 + 1 speed = 4 | 3 + 1 speed = 4 |
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
first_obs = next(iter(obs.values()))
print('spatial shape:', first_obs['observations'].shape)  # predator: (4, 7, 7)
print('action mask:', first_obs['action_mask'].shape)     # 9 actions
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
genome → speed trait → continuous move rate → accumulator-gated lifetime movement budget
       → policy learns speed-specific behaviour with action-masked frozen steps
       → survival/reproduction → inherited genome with mutation
```
