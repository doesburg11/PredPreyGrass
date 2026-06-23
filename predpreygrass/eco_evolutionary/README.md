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

## Baldwinian Enhancement: Speed in Observation

To strengthen the Baldwinian loop, the agent's own speed is exposed as a fourth
observation channel:

```python
"include_speed_in_obs": True,
```

Without this flag, the policy is blind to the genome — a prey with speed 0.6 and
one with speed 1.8 receive identical policy outputs for the same grid observation.
The genome only constrains which actions are physically effective, but the policy
cannot adapt its strategy to exploit or compensate for its own speed value.

With `include_speed_in_obs: True`, the observation tensor gains a fourth channel
filled with the agent's normalised speed:

```
speed_norm = (genome.speed - speed_min) / (speed_max - speed_min)
obs[channel_3, :, :] = speed_norm   # broadcast scalar across all spatial positions
```

This lets the policy learn distinct strategies per genome value — e.g. a fast prey
learns to flee aggressively using distance-2 moves, while a slow prey learns to
hide near grass patches. That behavioural differentiation increases the fitness
differential between fast and slow genomes, which in turn accelerates selection and
closes the Baldwinian loop:

```
genome → speed trait → policy learns speed-specific behaviour
       → stronger fitness differential → faster heritable selection
```

**Why knowing its own speed matters:**

Without speed in the observation, the policy learns one strategy averaged over the
entire genome distribution. A fast prey and a slow prey see the same grid and
receive the same action probabilities. Speed helps, but only passively — the fast
agent accidentally benefits because its distance-2 moves happen to execute; the slow
agent's distance-2 attempts get silently clipped. The genome influences outcomes
without the policy ever exploiting it deliberately.

With speed in the observation, the policy can pair each genome value with the
strategy that actually suits it. A fast prey actively pursues an aggressive flee
strategy; a slow prey learns not to bother fleeing and conserves energy near grass
instead. Each genome now carries its own matched behaviour, so the fitness gap
between fast and slow agents grows larger than it would from clipping effects alone.

Selection speed depends directly on the size of this fitness differential — a larger
gap produces a faster, cleaner evolutionary signal. The policy becomes an *amplifier*
of genome fitness rather than a neutral scaffold sitting on top of it. That
amplification is what closes the Baldwinian loop properly.

The network architecture is unchanged: the existing CNN simply receives four input
channels instead of three. The speed channel is spatially constant (same value
everywhere), so the network learns to use it as a global conditioning signal rather
than a spatial feature.

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

## Training Run History

### Run 1 — `PPO_REPRODUCTION_REWARD_ECO_EVOLUTION_2026-06-22_10-09-49` (87 iterations)

Config: `energy_gain_per_step_grass: 0.08`, `max_agent_age: prey=None`, `speed_std: 0.2`, `mutation_std: 0.1`

Key observations:
- Prey count grew to ~115 by iter 56, causing ~40% grid occupancy — movement restricted, speed advantage eroded
- Episode length hit 1000 (max) from iter 56 onward; all episodes maxing out
- `prey_speed_p50`: 0.929 → 0.942 (+0.013) — consistent upward evolutionary drift confirmed
- `predator_speed_p50`: flat at 0.863 throughout — no predator speed selection
- `prey_fraction_fast`: 0.0001 → 0.0004 — slow but upward
- Predator entropy: 3.21 → 0.89 (nearly converged); prey entropy: 3.21 → 1.54 (still converging)
- **Diagnosis**: overcrowding undermines speed selection pressure → halved grass regrowth and added prey age cap

### Run 2 — `PPO_ECO_EVOLUTION_2026-06-22_14-59-05` (in progress)

Config: `energy_gain_per_step_grass: 0.04`, `max_agent_age: prey=400`, `speed_std: 0.2`, `mutation_std: 0.1`

Observations at iter 63:
- Episode max lengths no longer consistently hitting 1000: peaked at 1000 in iters 28–34 then settled to 400–850 range — populations are cycling naturally rather than exploding; food scarcity working as intended
- Iteration time stable at 2–3 min throughout
- `predator_speed_p50`: 0.882 → 0.888 — slow but consistent upward drift (+0.006 total); directional predator speed selection visible and not seen in run 1 at this stage
- `prey_speed_p50`: oscillating narrowly in [0.944, 0.946] — flat; the slight downward drift from early iters has stopped but no upward reversal yet; expected once predator policy fully converges
- `predator_speed_mean`: 0.910 → 0.911 — very slight upward trend mirroring p50
- `prey_speed_mean`: stable at ~1.024–1.027 throughout
- `prey_fraction_fast`: 0.000466 (iter 29) → 0.000903 (iter 63) — nearly doubled in 34 iterations; consistent growth rate confirms faster prey are reproducing at a higher rate
- `predator_fraction_fast`: 0.0 throughout — predator p50 at 0.888, threshold at 1.5; no fast predators expected for many iterations
- Predator entropy: 3.21 → 1.08 at iter 63 — nearly converged; prey should begin feeling stronger hunting pressure imminently
- Prey entropy: 3.21 → 2.09 — still learning, more room to improve than predators
- **Watch**: prey_speed_p50 upward reversal expected as predator entropy approaches 1.0 (iter 70–80); `prey_fraction_fast` growth rate as a leading indicator of selection strength

## References

**Eco-evolutionary dynamics:**
- Lotka, A. J. (1925). *Elements of Physical Biology.* Williams & Wilkins. / Volterra, V. (1926). *Fluctuations in the Abundance of a Species Considered Mathematically.* Nature — predator-prey population oscillations; the theoretical foundation
- Van Valen, L. (1973). *A New Evolutionary Law.* Evolutionary Theory, 1, 1–30 — Red Queen hypothesis: predator-prey arms race drives continuous evolution
- Dawkins, R. & Krebs, J. R. (1979). *Arms Races Between and Within Species.* Proceedings of the Royal Society B, 205(1161), 489–511 — co-evolutionary speed/ability tradeoffs

**Baldwin Effect:**
- Hinton, G. E. & Nowlan, S. J. (1987). *How Learning Can Guide Evolution.* Complex Systems, 1(3), 495–502 — the canonical paper showing learned behaviour can steer heritable evolution without Lamarckian inheritance; directly relevant to the genome→PPO-policy architecture used here

**Digital / agent-based evolution:**
- Ofria, C. & Wilke, C. O. (2004). *Avida: A Software Platform for Research in Computational Evolutionary Biology.* Artificial Life, 10(2), 191–229 — closest precedent: digital organisms with heritable traits under selection
- Stanley, K. O. & Miikkulainen, R. (2002). *Evolving Neural Networks Through Augmenting Topologies (NEAT).* Evolutionary Computation, 10(2), 99–127 — genome-based evolution of agent behaviour

**Life history tradeoffs:**
- Stearns, S. C. (1992). *The Evolution of Life Histories.* Oxford University Press — speed-reproduction-survival tradeoffs; theoretical backing for the energy cost model

**Open-ended co-evolution:**
- Wang, R., Lehman, J., Clune, J. & Stanley, K. O. (2019). *Paired Open-Ended Trailblazer (POET).* arXiv:1901.01753 — endlessly co-evolving agents and environments; most similar in spirit to the long-run goal of this project

## Interpretation

The intended model is:

```text
genome -> speed trait -> lifetime behavior through shared PPO policy
       -> survival/reproduction -> inherited genome with mutation
```

This supports demographic selection, lineage tracking, and heritable biological
variation without claiming that acquired policy weights are biologically
inherited.
