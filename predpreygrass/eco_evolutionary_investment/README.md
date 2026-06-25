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

Recommended guardrails:

```python
"min_offspring_energy_predator": 1.0
"min_offspring_energy_prey": 1.0
"max_offspring_energy_predator": initial_energy_predator
"max_offspring_energy_prey": initial_energy_prey
```

Alternative later variant:

```text
offspring_energy = fixed_base + fraction * surplus_above_reproduction_threshold
```

That may be safer if direct fraction-of-current-energy creates runaway collapse
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
