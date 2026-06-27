# Metabolic Rate as a Second Genome Trait

## Why offspring_investment_fraction alone is incomplete

`offspring_investment_fraction` creates a one-way link between learning and evolution:

```
RL behavior → energy surplus → which fraction gets selected
```

The genome evolves *because of* what the policy has learned, but the fraction itself never
feeds back into what the policy needs to learn. Policy and genome operate in separate domains:
the policy governs foraging and movement; the investment fraction governs reproduction
economics. There is no second arrow.

This is only half the Darwin/Baldwin story.

## What metabolic_rate adds

`metabolic_rate` scales both sides of the energy equation symmetrically:

```python
energy_decay = basal_energy_cost * metabolic_rate   # basal cost per step
energy_gain  = raw_food_energy   * metabolic_rate   # intake per food item
```

The food source loses the same amount regardless (digestive efficiency model). Whether a
high or low metabolic rate is advantageous depends entirely on how reliably the agent can
find food:

- **food-rich / policy has learned to forage well**: high rate wins — the gain bonus
  outweighs the basal cost
- **food-sparse / policy is still poor, or prey have learned evasion**: low rate wins —
  the cost dominates

This creates a genuine two-way coupling between genome and policy:

```
Direction 1 (Darwin):
  RL behavior quality → determines which metabolic rate is advantageous
                      → selection propagates that rate through the population

Direction 2 (Baldwin):
  genome (metabolic rate) → changes energy dynamics every step
                          → changes what the RL policy must learn to survive
```

Both arrows now exist. As prey learn better evasion, food becomes scarcer and the optimal
predator metabolic rate shifts downward — which in turn changes the energy landscape that
future policy updates must adapt to. The Darwinian and Baldwinian layers are genuinely
coupled in both directions.

`offspring_investment_fraction` only has Direction 1. `metabolic_rate` has both.

## The problem with symmetric metabolic_rate: no individual-level stabilizing force

With symmetric scaling, the net energy per step is:

```
net energy = (food_found × metabolic_rate) - (base_cost × metabolic_rate)
           = metabolic_rate × (food_found - base_cost)
```

For an agent that survives at all, `food_found > base_cost` on average — otherwise it would
starve. This means the net energy is always proportional to `metabolic_rate`: higher is always
better at the individual level. There is no interior optimum. The genome races to the upper
bound (2.0) once the policy clears the survival threshold.

The only stabilizing force is **population-level density dependence**: high-metabolic predators
hunt harder → prey population crashes → food becomes scarce → high metabolic rate is penalized.
This is a real effect but it operates at the population level, is slow, and is hard to
interpret in training curves.

## Asymmetric metabolic_rate: closing the loop with a genuine tradeoff

The fix is **sub-linear gain with linear cost** — a digestive saturation model:

```python
energy_gain  = raw_food_energy * metabolic_rate ** α     # diminishing returns, α < 1
energy_decay = basal_energy_cost * metabolic_rate         # linear cost (config key: basal_energy_cost_predator / basal_energy_cost_prey)
```

Biological motivation: a higher metabolic rate extracts more energy per food item, but
digestive capacity saturates — the gut can only absorb so much regardless of how fast the
metabolism runs. The cost of running a faster metabolism (basal heat production, cellular
maintenance) does not saturate.

This creates a genuine interior optimum at the individual level. Taking the derivative and
setting to zero:

```
d(net energy)/d(metabolic_rate) = 0
→ optimal rate = (α × food_found / basal_energy_cost) ^ (1 / (1 - α))
```

The optimum depends on `food_found` per step — which depends directly on RL policy quality.
A better forager has a higher optimal metabolic rate. As the policy improves, the optimum
shifts upward. As food becomes scarce (prey learn evasion), the optimum shifts downward.

This preserves the two-way Baldwin loop:

```
Direction 1: RL quality → food_found → optimal metabolic rate → genome selection
Direction 2: genome (metabolic rate) → energy dynamics → what RL must learn to survive
```

And adds a genuine individual-level stabilizing force:

```
Too low:  leaves digestive capacity on the table when food is available
Too high: basal cost exceeds diminishing digestive returns even when food is found
```

A reasonable exponent is `α = 0.7`, giving moderately diminishing returns. The biological
plausible range is roughly 0.5–0.8.

## Tradeoffs between trait designs

| | `offspring_investment_fraction` | `metabolic_rate` (symmetric) | `metabolic_rate` (asymmetric, α<1) |
|---|---|---|---|
| Darwin/Baldwin loop | one-way (incomplete) | two-way (complete) | two-way (complete) |
| Individual stabilizing force | yes (interior optimum) | no (races to bound) | yes (interior optimum) |
| Domain overlap with RL | none (reproduction only) | full (every step) | full (every step) |
| Ease of training | easier | harder | harder |
| Interpretability | high | moderate | moderate |
| Single trait sufficient? | no (incomplete loop) | no (no tradeoff) | yes |

## Why training is harder with metabolic_rate

With `offspring_investment_fraction`, the genome only matters at reproduction — a relatively
rare event. The policy can be trained in a nearly-neutral genome landscape for most of its
lifetime. Selective pressure is episodic.

With `metabolic_rate`, the genome affects every single step. Early in training, when the
policy is near-random, high and low metabolic rates produce similar fitness because the
policy cannot reliably exploit food regardless. The selective signal is weak and noisy until
the policy reaches a threshold of competence. Once the policy is competent enough, selection
sharpens quickly. This means metabolic rate evolution lags behind policy learning — which is
actually ecologically realistic, but makes training curves harder to read.

## Founder distribution

Starting at `mean=1.0` makes the initial population behave identically to the no-genome
baseline. Any drift away from 1.0 is a direct signal that selection is acting:

```python
"metabolic_rate_mean": 1.0,   # neutral start
"metabolic_rate_std":  0.10,  # initial variation for selection to act on
"trait_bounds": (0.5, 2.0),
```

## What to watch in training

- `eco_evolution/{species}_metabolic_rate_mean` — drift from 1.0 indicates selection;
  direction reveals the dominant ecological pressure (food-rich vs. food-sparse regime)
- `eco_evolution/{species}_metabolic_rate_std` — narrowing means selection is acting and
  reducing variation; flat std means the trait is currently neutral
- Compare drift direction with episode return: rising returns (better foraging) should
  favour higher metabolic rates; falling or stagnant returns favour lower rates
- Cross-species: predator and prey metabolic rates should drift in response to each other
  as the coevolutionary arms race develops

## Relationship to the Darwin/Baldwin process

The Baldwin Effect specifically requires:

1. Individual learning discovers a beneficial behavior within a lifetime
2. That discovery creates selective pressure on heritable traits
3. Over generations, the trait that made learning easier (or made the learned behavior
   unnecessary) becomes fixed in the genome

`metabolic_rate` satisfies all three conditions in this simulation:

1. The RL policy learns which metabolic rate is viable given current food availability
2. Agents whose genome matches the policy-determined optimum reproduce more
3. The genome drifts toward encoding the metabolic strategy that the policy converged on —
   the Darwinian layer gradually fixes what the Baldwinian layer discovered

`offspring_investment_fraction` satisfies conditions 1 and 2 but not 3: the fraction never
becomes a substitute for learning because it operates in a domain (reproduction) that the
policy does not control.
