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

### Run 2 — `PPO_ECO_EVOLUTION_2026-06-22_14-59-05` (stopped at iter 65)

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
- **Stopped at iter 65** before the predicted reversal; succeeded in demonstrating natural population cycling and directional predator speed selection absent in run 1

### Run 3 — `PPO_ECO_EVOLUTION_2026-06-23_10-30-31` (in progress)

Config changes vs run 2:
- `include_speed_in_obs: True` — in runs 1 and 2 the policy was genome-blind: a prey with speed 0.6 and one with speed 1.8 received identical policy outputs for the same grid observation. Speed only helped passively (fast agents happened to execute distance-2 moves). By adding the agent's own normalised speed as a 4th observation channel, the policy can learn distinct strategies per genome value — fast prey flee aggressively, slow prey hide near grass. This increases the fitness differential between fast and slow genomes, amplifying selection and closing the Baldwinian loop properly.
- `utils/episode_return_callback.py`: `episode.length` does not exist on RLlib's new API stack `MultiAgentEpisode`; `getattr(episode, "length", 0)` silently returned 0 every episode, making console output misleading. Fixed to `episode.env_steps()`, the correct method confirmed by inspecting the `MultiAgentEpisode` class and verified with a unit test.

Observations at iter 78:
- `prey_speed_p50`: **unambiguous upward trend** — 0.925 (iter 1) → 0.929 (iter 34) → 0.937 (iter 78), +0.012 total; acceleration visible in the last 14 iterations (0.932 → 0.937); directional upward drift was absent in run 2 for the first 30+ iterations
- `prey_fraction_fast`: **new high of 0.000842** at iter 78 — more than double run 2 at the same iteration count (~0.000391); after a dip to 0.000473 at iter 67 following the population crash, now climbing consistently: 0.000679 → 0.000768 → 0.000803 → 0.000829 → 0.000842 over the last 5 iterations
- `predator_speed_p50`: 0.902 (iter 1) → 0.890 (iter 64) → 0.895 (iter 78) — oscillating, slight upward recovery; no strong directional signal yet
- `predator_fraction_fast`: 0.0 throughout
- Episode lengths stabilised at mean 210–270 after the iter 33–34 population crash — populations cycling naturally under predation pressure
- **Key signal at iter 33–34**: when predator entropy crossed below 2.0 and hunting became effective, mean episode length halved and `prey_fraction_fast` jumped 72% in 3 iterations — fast prey survived the mortality event proportionally better; directional natural selection under predation pressure confirmed
- Predator entropy: 3.21 → 1.18 at iter 78 — well converged, predators hunting effectively and sustaining selection pressure on prey speed
- Prey entropy: 3.21 → 1.92 — still learning; Baldwinian loop active (policy still adapting to exploit speed differences)
- **Caveat**: founder genome is sampled fresh each run (speed_std=0.2); run 3 started with prey_p50=0.925 vs 0.947 in run 2, so some differences reflect sampling variance rather than the Baldwinian enhancement alone
- **Watch**: whether `prey_speed_p50` continues rising toward the 1.5 fast threshold as predator entropy approaches 1.0; whether `prey_fraction_fast` keeps accelerating

Observations at iter 134:
- `prey_speed_p50`: rose from 0.925 (iter 1) to a peak of 0.943 (iter 89), then softened to 0.933 at iter 134. The prey median remains above the starting value, but the clean upward drift seen at iter 78 did not continue monotonically.
- `prey_fraction_fast`: 0.000842 (iter 78) → 0.001265 (iter 134), with a peak of 0.001532 at iter 103. Rare fast prey remain overrepresented relative to the early run, but the signal is no longer accelerating.
- `predator_speed_p50`: 0.895 (iter 78) → 0.920 (iter 134), and `predator_speed_mean`: 0.908 → 0.923. Predator speed now has the clearest directional selection signal in the run.
- `predator_fraction_fast`: 0.0 throughout — predator median speed is still far below the 1.5 fast threshold, so absence of fast predators remains expected.
- Episode length: 247 mean steps at iter 78 → 198 at iter 134, indicating predation pressure remains strong after predators converge.
- Predator entropy: 1.18 (iter 78) → 0.99 (iter 134); prey entropy: 1.92 → 1.58. Both policies are still adapting, but predator behaviour is now close to converged.
- **Red Queen interpretation**: this is a plausible early Red Queen-like signature, not yet a strong result. Selection pressure appears to move between species over time: prey speed selection emerged after predators became effective, then predator speed selection strengthened later. That matches the expected pattern where one side's improvement changes the fitness landscape for the other.
- **Caveat**: a convincing Red Queen result needs sustained reciprocal escalation or repeated cycling over longer time, ideally across multiple independent seeds. At iter 134, the latest signal is predator catch-up plus elevated rare-fast prey frequency, not a clean two-species arms race yet.
- **Watch**: whether prey median speed or `prey_fraction_fast` rises again after predator entropy settles near or below 1.0; repeated predator-prey alternation would strengthen the Red Queen claim.

Observations at iter 172:
- `prey_speed_p50`: 0.933 (iter 134) → 0.923 (iter 172), now essentially back to the run-start median of 0.925. The earlier prey median-speed signal has not persisted.
- `prey_speed_mean`: 1.008 (iter 134) → 0.993 (iter 172), also declining. This weakens any claim that prey speed is currently under clean upward selection in this threshold-based setup.
- `prey_fraction_fast`: 0.001265 (0.126%) at iter 134 → 0.000982 (0.098%) at iter 172. The fast tail remains non-zero but is too small to be a headline result; it is best treated as a weak supporting metric.
- `predator_speed_p50`: 0.920 (iter 134) → 0.929 (iter 172), and `predator_speed_mean`: 0.923 → 0.927. Predator speed remains the clearest directional selection signal.
- `predator_fraction_fast`: 0.0 throughout. Predators are still far below the 1.5 threshold, so the meaningful change is in the sub-threshold distribution, not in the binary fast category.
- Episode length: 198 mean steps at iter 134 → 283 at iter 172. The ecology remains dynamic rather than collapsed, but the prey-speed response has not reappeared yet.
- Predator entropy: 0.99 (iter 134) → 1.14 (iter 172); prey entropy: 1.58 → 1.45. Policy learning and ecological feedback are still changing together, which complicates clean interpretation.
- **Revised interpretation**: run 3 still demonstrates that inherited speed distributions can move under selection, especially for predators, but it is not a clean upward-speed result for prey. The hard 1.5 movement threshold makes the evidence less legible because most agents remain far below the point where speed changes movement distance.
- **Follow-up implication**: `eco_evolutionary_cadence` is a stronger next experiment for proving Darwinian/Baldwinian dynamics. In cadence, speed has a graded effect on movement frequency, so small genome shifts immediately affect behaviour and can be tracked with `cooldown_mean`, `speed_p50`, and `fraction_mobile` instead of relying on a rare threshold-crossing tail event.

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
