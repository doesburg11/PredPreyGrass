# Eco-Evolutionary Investment

This module is intended to test Darwinian evolution with Baldwinian learning in
PredPreyGrass using a biologically grounded life-history trait:

```text
offspring_investment_fraction
```

It starts from `predpreygrass/eco_evolutionary`, not from the cadence variant.
The cadence experiments showed weak or absent selection on movement speed, likely
because the speed phenotype was indirect and partly discretized. Offspring
investment should give a cleaner evolutionary signal because it acts directly at
reproduction.

## Conceptual Goal

The intended model is Darwinian/Baldwinian, not Lamarckian.

Darwinian evolution:

- individuals carry heritable genome traits;
- offspring inherit parent traits with mutation;
- traits change in frequency when some genomes produce more surviving offspring.

Baldwinian learning:

- agents learn within their lifetime through shared PPO policies;
- learned policy weights are not inherited by individual offspring;
- learned behavior still feeds back into selection because it changes survival,
  energy acquisition, and reproduction.

In short:

```text
genome -> phenotype -> learned behavior changes fitness -> genome frequencies shift
```

The experiment asks whether a learned predator/prey ecology creates selection
pressure on parental investment strategies.

## Why Offspring Investment

Offspring investment is a life-history trait with a real biological analogue:
organisms vary in how much energy they invest per offspring.

The basic tradeoff:

```text
low investment  -> parent keeps more energy, offspring start weaker
high investment -> offspring start stronger, parent pays a larger cost
```

This is attractive for this project because:

- it is biologically interpretable;
- it affects reproduction directly;
- it is continuous, so small mutations can matter;
- it does not require changing observation or action spaces;
- learned foraging/escape/hunting behavior can alter which investment strategy is
  favored.

## Planned Trait

Add a genome trait:

```python
offspring_investment_fraction: float
```

Suggested initial bounds:

```python
"offspring_investment_fraction": (0.10, 0.80)
```

Suggested founder distribution:

```python
"founder_genome": {
    "predator": {"offspring_investment_fraction_mean": 0.35, "offspring_investment_fraction_std": 0.08},
    "prey": {"offspring_investment_fraction_mean": 0.35, "offspring_investment_fraction_std": 0.08},
}
```

Suggested mutation:

```python
"genome_mutation": {
    "rate": 0.05,
    "std": 0.04,
}
```

Exact values should be tuned after the first smoke tests. The key is to avoid a
founder distribution that starts pinned to a bound.

## Planned Reproduction Rule

The cleanest first implementation is conserved-energy investment.

At reproduction:

```python
investment = parent_energy * offspring_investment_fraction
offspring_energy = clamp(investment, min_offspring_energy, max_offspring_energy)
parent_energy -= offspring_energy
```

This makes the tradeoff explicit:

- high-investment parents produce offspring with more starting energy;
- the parent has less energy after reproduction;
- low-investment parents preserve energy but produce more fragile offspring.

Offspring energy is unclamped: `offspring_energy = parent_energy × offspring_investment_fraction`.
The reproduction threshold already ensures parents have sufficient energy before spawning.
or excessive parent starvation.

## Metrics To Add

Genome distribution:

- `eco_evolution/{species}_offspring_investment_fraction_mean`
- `eco_evolution/{species}_offspring_investment_fraction_std`
- `eco_evolution/{species}_offspring_investment_fraction_p25`
- `eco_evolution/{species}_offspring_investment_fraction_p50`
- `eco_evolution/{species}_offspring_investment_fraction_p75`

Fitness and life-history diagnostics:

- `eco_evolution/{species}_offspring_count_mean`
- `eco_evolution/{species}_agent_count`
- `eco_evolution/{species}_offspring_initial_energy_mean`
- `eco_evolution/{species}_parent_energy_after_reproduction_mean`
- `eco_evolution/{species}_reproduction_energy_invested_mean`

The most important early readout is whether investment p50 moves while offspring
survival/reproduction metrics change coherently.

## Expected Selection Patterns

Possible outcomes:

- **Investment drifts upward**: offspring starting energy is limiting; stronger
  juveniles survive and reproduce more.
- **Investment drifts downward**: parent survival or repeated reproduction is
  more important than offspring starting energy.
- **Intermediate investment emerges**: the ecological tradeoff is balanced.
- **No directional movement**: investment may be neutral under current energy
  thresholds, or PPO behavior may not create strong enough differences in
  offspring survival.

A useful result is not necessarily monotonic upward selection. A stable middle
strategy would be especially interesting because it would indicate a real
life-history tradeoff.

## Implementation Plan

1. Rename copied imports and package references to
   `predpreygrass.eco_evolutionary_investment`.
2. Replace the copied speed genome with `offspring_investment_fraction`.
3. Update founder sampling, mutation bounds, and genome tests.
4. Modify reproduction so offspring energy is controlled by the parent genome.
5. Record investment-specific reproduction metrics.
6. Update callback/tests so TensorBoard logs the new trait summaries.
7. Add a small sweep script only after the single-regime smoke test works.

## First Smoke Test

Use one simple baseline first:

```python
offspring_investment_fraction_mean = 0.35
offspring_investment_fraction_std = 0.08
trait_bounds = (0.10, 0.80)
mutation_rate = 0.05
mutation_std = 0.04
```

Run long enough to confirm:

- offspring inherit mutated investment fractions;
- parent energy decreases by the invested amount;
- offspring initial energy reflects the parent trait;
- investment metrics appear in Ray/TensorBoard;
- no obvious population collapse occurs.

Only then add a focused sweep over founder mean, mutation std, or min/max
offspring energy clamps.

## What This Module Is Not

This module does not inherit learned PPO weights per offspring. Policies remain
shared by species/policy group. That keeps the experiment Baldwinian rather than
Lamarckian.

This module also should not include cadence-specific speed/cooldown mechanics.
Those belong to `eco_evolutionary_cadence` and are intentionally excluded here.

## Key Files

- `predpreygrass_rllib_env.py`: environment lifecycle, reproduction, genome
  inheritance, and metrics.
- `config/config_env_eco_evolutionary.py`: founder distributions, trait bounds,
  mutation settings, and energy parameters.
- `utils/genome.py`: `Genome` dataclass plus founder/mutation helpers.
- `utils/episode_return_callback.py`: logging of evolutionary metrics.
- `tests/test_eco_evolutionary_validation.py`: regression tests for lifecycle,
  genome inheritance, and metric logging.

## Second Trait: Metabolic Rate

The `metabolic_rate` genome trait closes the loop that `offspring_investment_fraction` alone
cannot. It is added alongside the investment fraction as a second heritable trait.

### The tradeoff

```text
high metabolic_rate -> extracts more energy per food item AND burns more per step
low metabolic_rate  -> extracts less energy per food item AND burns less per step
```

Both sides of the tradeoff scale by the same factor:

```python
# basal energy loss per step
energy_decay = base_decay_rate * metabolic_rate

# energy gained from food (grass for prey, prey for predator)
energy_gain = raw_food_energy * metabolic_rate
```

The food source loses the same amount regardless (digestive efficiency model). Whether a
high or low metabolic rate is advantageous depends on how easily the agent can find food:

- food-rich environment (policy has learned to forage well): high rate wins — the gain
  bonus outweighs the cost
- food-sparse environment (policy is still random, or prey have learned to evade well):
  low rate wins — the cost dominates

### Why this closes the Baldwinian loop

`offspring_investment_fraction` only affects energy at reproduction — the genome never
influences what the policy sees or how it learns. `metabolic_rate` is different: it is
active every step of every agent's life. The fitness advantage of a given metabolic rate
depends directly on how effective the current shared policy is at accumulating food.

```text
genome (metabolic_rate) -> food-accumulation efficiency -> reproduction frequency
RL policy quality       -> determines which metabolic rate is advantageous
```

Both arrows now run in both directions:

```text
RL behavior -> energy accumulation -> reproduction -> genome propagation   (exists)
genome      -> energy dynamics     -> what RL must learn to survive        (now exists)
```

A policy that hunts effectively makes high-metabolic-rate predators viable. As prey learn
better evasion, food becomes scarcer and the optimal predator metabolic rate shifts
downward — which in turn changes the energy dynamics that future policy updates must
adapt to. The Darwinian and Baldwinian layers are now genuinely coupled in both
directions.

### Founder distribution and bounds

```python
"metabolic_rate_mean": 1.0,   # neutral: identical to base config at initialisation
"metabolic_rate_std": 0.10,
"trait_bounds": (0.5, 2.0),
```

Starting at mean 1.0 means the initial population behaves identically to the pre-trait
baseline. Selection pressure determines which direction the trait drifts.

### What to watch in training

- `eco_evolution/{species}_metabolic_rate_mean` — should drift from 1.0 as the policy
  develops; direction reveals the dominant ecological pressure
- `eco_evolution/{species}_metabolic_rate_std` — narrowing indicates selection is acting;
  flat std means the trait is neutral under current conditions
- Compare drift direction with episode return trajectory: rising returns (better foraging)
  should favour higher metabolic rates; declining returns or prey-dominated episodes
  should favour lower rates

## Important Note On The Baldwinian Structure

All predators share one neural network policy and all prey share another. A
newly spawned agent immediately acts using the current shared policy weights —
it does not learn from scratch. This might look Lamarckian: a parent acquires
foraging skill during its lifetime and its offspring inherit that skill.

The distinction is at the level at which inheritance operates. In Lamarckian
inheritance this specific parent's acquired behavior is passed to this specific
offspring. Here the child of a strong predator and the child of a weak predator
receive exactly the same policy weights. What is shared is not the individual
parent's experience but species-level collective knowledge built from all agents
across all training iterations. No individual parent has more influence on the
child's starting policy than any other.

However, this reveals a real tension with the classical Baldwinian definition.
In classical Baldwinian evolution the genome and learning interact directly:
agents with certain genes find certain behaviors easier to learn, which is what
creates selection pressure on those genes. In this model the
`offspring_investment_fraction` genome does not influence the policy or learning
at all — it only affects energy dynamics at reproduction. The Darwinian and
Baldwinian layers are coupled purely through energy:

```text
RL behavior -> energy accumulation -> reproduction -> genome propagation
```

The coupling is real but asymmetric. The RL policy creates selection pressure on
the genome — a policy that accumulates energy slowly rarely triggers
reproduction, so the investment fraction is barely selected at all; a policy
that accumulates energy fast triggers reproduction frequently, making the
investment fraction matter a lot. But the genome never feeds back into the
learning. The investment fraction does not influence what the agent observes,
how fast it learns, or what behavior PPO reinforces. This one-way coupling is
what makes the structure weaker than classical Baldwinian.

A classical Baldwinian genome trait would close the loop in both directions:

```text
gene -> learning capacity -> fitness -> gene frequencies shift
```

In this model only the second arrow exists:

```text
RL behavior -> energy -> reproduction -> genome propagation   (exists)
genome -> learning capacity                                    (does not exist)
```

This places the experiment between classical Baldwinian and pure Darwinian on a
spectrum:

- Lamarckian: learned weights inherited directly by offspring
- Classical Baldwinian: genome influences learning, learning influences fitness
- This experiment: RL learning influences fitness, fitness selects the genome,
  genome does not influence learning
- Pure Darwinian: no learning at all, only selection

Whether the one-way coupling is strong enough to produce meaningful
evolutionary dynamics in the investment fraction is an open empirical question
that training will have to answer.

## How To Know If The Interaction Is Working

The tests verify the mechanism is correctly implemented. Whether it is producing
the intended evolutionary dynamics is a separate question answered by watching
training metrics in TensorBoard.

### Darwinian signal

The investment fraction should drift away from its starting value (0.35) and
stabilise somewhere. Watch:

- `eco_evolution/predator_investment_fraction_mean` and
  `eco_evolution/prey_investment_fraction_mean` — should move over training, not
  stay flat at 0.35.
- `eco_evolution/predator_investment_fraction_std` and
  `eco_evolution/prey_investment_fraction_std` — should narrow as the population
  converges on a fit strategy.
- `p25`, `p50`, `p75` percentiles — spread indicates how strong selection
  pressure is.

### Baldwinian signal

RL behavior improving should increase selection pressure on the genome. As agents
learn to forage better they hit the reproduction threshold more often, which means
more reproductive events and more genome selection. Watch:

- Episode returns increasing — agents learning to forage.
- `eco_evolution/predator_offspring_count_mean` and
  `eco_evolution/prey_offspring_count_mean` — should rise as RL improves and
  agents accumulate energy faster.

### The interaction specifically

If it is working, the investment fraction should evolve faster or more clearly as
episode returns improve. Early in training agents barely reproduce (Baldwinian
gate rarely opens), so selection on the genome is weak and the fraction stays
near 0.35. As the policy improves, reproduction becomes frequent and the genome
has something to select on.

### Red flags

- Investment fraction stays flat at 0.35 throughout training — mutation rate or
  selection pressure too weak.
- Episode returns never improve — agents cannot learn to forage, the Baldwinian
  gate never opens, the genome is never selected.
- `offspring_count_mean` stays near zero — reproduction never happens, the
  Darwinian layer is effectively switched off.
