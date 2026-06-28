# Eco-Evolutionary Metabolic Rate — Training Results

Results from two PPO training runs on the `eco_evolutionary_metabolic_rate` environment.
Analysis focuses on whether the Darwin/Baldwin loop is operating.

---

## Environment recap

Agents carry a **metabolic rate genome** (MR, founder mean 1.0 ± 0.10, mutation rate 5%, std 0.04, bounds [0.5, 2.0]).

The tradeoff is designed to create a policy-dependent optimum:

| Effect | Formula | Direction |
|---|---|---|
| Energy cost | `basal_cost × MR` | Higher MR → costlier to exist |
| Energy gain (eating) | `food_energy × MR^α` (α = 0.70) | Higher MR → more energy per bite |

With α < 1 (sub-linear gain, linear cost) there is an interior optimum whose location depends on *how often the agent eats* — i.e., on policy quality. This is what links Darwin (genome selection) to Baldwin (RL learning).

Rewards: only **reproduction** yields signal (`+10`). Catching prey (predators) and eating grass (prey) give `0`. This is intentionally sparse.

---

## Run 1 — PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-06-28_14-48-16

**80 iterations, 82K steps, 8 checkpoints.**

### Ecology

Predator:prey ratio (live_genome counts) climbed from ~1:9 at iter 1 to nearly 1:1 by iter 79. This is ecologically unsustainable and caused frequent within-episode population crashes: 52 of 80 iterations report NaN on all eco-evolution metrics. The environment did not reach a stable attractor in this run.

### RL learning (Baldwin)

Prey spawned_total grew from 6 → ~530 per episode; predator spawned_total grew from 0 → ~185. Both policies learned. Predator `episode_return_p50 = 0.000` throughout — not a failure signal but a structural artifact of the reproduction chain (see below).

### Genome evolution (Darwin)

Prey live_genome MR: weak upward drift from ~0.960 to ~0.970 (+0.010). Predator live_genome MR: slight downward drift (~1.015 → ~0.996, −0.020). Both small relative to noise. No directional gene-selection signal detectable.

### Darwin/Baldwin loop verdict

**Not yet cycling.** The run is too short and the ecosystem too unstable for the policy-genome feedback to register. Darwinian selection for lower MR (survival-dominant) is present but weak. No selection gap reversal observed.

---

## Run 2 — PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-06-28_15-14-26

**219 iterations, 225K steps, 10+ checkpoints. Still running at iter 217 at time of analysis.**

This is the main run. All findings below refer to it.

---

### Clarification: why predator episode_return_p50 = 0 throughout

`predator_episode_return_p50` is 0.000 in every iteration of both runs. This is **not evidence that predators fail to learn or fail to reproduce**.

The callback measures the individual-lifetime return of *every* predator agent ID that ever lived in the episode, including the large majority of short-lived offspring who die before reproducing. With ~170 unique predator IDs per episode and fewer than 50% reproducing before death, the median is 0 by construction regardless of how good the policy is. The real learning signal is in:

- `predator_spawned_total`: grew from 0 (iter 1) → ~127 (iter 21) → ~170–195 (iter 41+). That jump is the policy learning to hunt.
- `predator_offspring_count_mean ≈ 0.97`: survivors at episode end each produced ~1 offspring on average. Nearly every surviving predator reproduced.

Predators do reproduce. The p50 = 0 reflects population structure (deep reproduction chain of ~1 offspring per parent), not learning failure.

---

### Ecology (225K steps)

At the stable phase (iter 40 onward):
- Prey alive (live_genome): ~15–22 per step
- Predators alive (live_genome): ~16–20 per step
- Ratio ~1:1 — ecologically tight, consistent with a persistent predator-prey arms race

Predator reproduction never hit the `n_possible_predators = 500` capacity ceiling (reproduction_blocked = 0 throughout). The ecosystem is under pressure but not collapsing.

About 55% of iterations still report NaN on eco-evolution metrics (population too depleted mid-episode to compute stats). This is an inherent feature of the boom-bust ecology, not a bug.

---

### RL learning trajectory

| Quarter | Iters | Prey spawned | Prey ret p75 | Pred spawned |
|---|---|---|---|---|
| Q1 | 1–67 | 387 | 5.60 | 127 |
| Q2 | 70–126 | 522 | 6.24 | 171 |
| Q3 | 128–169 | 530 | 6.08 | 168 |
| Q4 | 170–214 | 529 | 6.69 | 164 |

Prey policy improved steadily through Q1–Q2, then plateaued at ~530 spawns/episode. Prey return p75 is slowly rising (5.6 → 6.7), suggesting continued slow improvement. Predator spawned_total plateaued around 165–180, consistent with the predator:prey ratio ceiling.

---

### Genome evolution — the Darwin signal

#### Prey live_genome MR trajectory

| Quarter | Iters | Live_genome MR | Std |
|---|---|---|---|
| Q1 | 1–67 | 0.974 | ±0.019 |
| Q2 | 70–126 | 1.015 | ±0.009 |
| Q3 | 128–169 | 1.032 | ±0.009 |
| Q4 | 170–214 | 1.021 | ±0.016 |

**+0.058 peak shift from Q1 to Q3.** The population-wide MR converged sharply in Q2–Q3 (variance halved) before broadening slightly in Q4. This is not random drift: both the direction and the narrowing of variance indicate selection around a target value.

#### Predator live_genome MR trajectory

| Quarter | Iters | Live_genome MR | Std |
|---|---|---|---|
| Q1 | 1–67 | 0.998 | ±0.022 |
| Q2 | 70–126 | 0.995 | ±0.011 |
| Q3 | 128–169 | 0.991 | ±0.017 |
| Q4 | 170–214 | 1.000 | ±0.011 |

Predators converged to MR ≈ 1.0 with no directional drift. Variance shrinking (±0.022 → ±0.011). The genome stabilized at the founder mean.

---

### The Baldwin signal — selection gap reversal

The **selection gap** = `eco_evolution MR − live_genome MR`.

- `eco_evolution MR`: metabolic rate of agents *alive at episode end* (survivors).
- `live_genome MR`: time-weighted average MR of the living population across all steps.

A **negative gap** means lower-MR agents survive to episode end more — survival selection favours lower cost. A **positive gap** means higher-MR agents survive to episode end more — reproduction/energy-gain selection dominates.

#### Prey selection gap

| Quarter | Gap | Interpretation |
|---|---|---|
| Q1 | −0.002 | Cost-side dominant: live longer by burning less |
| Q2 | −0.002 | Same |
| Q3 | **+0.012** | **Gain-side dominant: eat enough that higher MR pays off** |
| Q4 | +0.003 | Still positive, weakening toward equilibrium |

**The sign flipped between Q2 and Q3.** This is the Baldwin mechanism: the prey policy improved enough (spawned_total 387 → 530) that agents eating grass frequently enough to tip the energy tradeoff. Higher MR now extracts more energy per grass eaten → reach reproduction threshold faster → more offspring before dying. The genome followed.

#### Predator selection gap

| Quarter | Gap | Interpretation |
|---|---|---|
| Q1 | −0.001 | Neutral / slight cost-side |
| Q2 | +0.007 | Gain-side emerging |
| Q3 | +0.017 | Gain-side dominant |
| Q4 | +0.005 | Weakening |

Predators show the same pattern but the genome didn't move, because the predator policy improvement was smaller (p50 return structurally 0; spawned_total plateaued earlier). The selection pressure for higher MR is present in the data, but hasn't been strong or sustained enough to shift the genome away from 1.0.

---

### Darwin/Baldwin loop verdict

**Yes, the loop is working — most clearly for prey.**

The loop requires three things:

1. **RL improvement changes which genome is adaptive.** ✓ The prey selection gap reversed sign as policy improved. This is direct evidence that RL learning altered the fitness landscape for the genome.

2. **The genome responded to that new selection pressure.** ✓ Prey MR drifted from 0.974 to 1.032, converging with reduced variance.

3. **The genome shift feeds back into RL.** Plausible but not directly measurable from these logs. Higher MR changes the energy landscape the policy operates in (more gain per action). Prey return p75 rising slowly (5.6 → 6.7) is consistent with this, but cannot be cleanly separated from ordinary policy improvement.

For predators, points 1 and 2 are partially present (selection gap turned positive, genome stable at 1.0 — possibly already at equilibrium for the current policy quality) but no clear genome shift.

**Caveats:**

- The genome shift is modest (~6% for prey peak). Whether it would continue to grow or has found its equilibrium is unclear. The Q4 pullback (1.032 → 1.021) may indicate the system is settling.
- The loop is not amplifying — it found a fixed point near MR ≈ 1.0–1.03 rather than generating a runaway selection sweep. This is consistent with the design (interior optimum, bounded trait).
- 55% NaN episode rate means that in the majority of episodes the population collapses enough that no eco-evolution stats can be reported. This limits the effective selection bandwidth: meaningful selection only operates in the surviving 45% of episodes.
- The reverse leg (genome → RL) would require a controlled experiment (freeze genome, compare RL trajectories) to confirm directly.

---

### Current state (iter 217, 225K steps)

Both genomes have stabilized:
- Prey live_genome: oscillating around 1.00–1.05
- Pred live_genome: oscillating around 0.98–1.01
- Prey eco_evol: 0.92–1.10 (noisy — selection direction varies episode to episode)
- Reproduction stable: prey ~530/episode, predators ~165/episode, no capacity blocking

The loop has found an approximate equilibrium. Whether further training (> 500K steps) would reveal another phase of genome drift — driven by continued slow policy improvement — remains to be seen.
