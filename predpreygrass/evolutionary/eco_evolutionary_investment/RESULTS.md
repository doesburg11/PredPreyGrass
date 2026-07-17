# Training Analysis — eco_evolutionary_investment
## First ~178 Iterations, Run Started 2026-06-26

---

## 1. Experiment Setup

### Environment
| Parameter | Value |
|---|---|
| Grid | 25 × 25 |
| Max steps per episode | 1000 |
| Observation channels | 3 (predators, prey, grass) |
| Predator obs window | 7 × 7 |
| Prey obs window | 9 × 9 |
| Actions | 9 (3×3 Moore neighbourhood, stay included) |
| Movement energy cost | 0.0 (disabled) |
| Predator basal decay | 0.15 / step |
| Prey basal decay | 0.05 / step |
| Predator reproduction threshold | 12.0 energy |
| Prey reproduction threshold | 8.0 energy |
| Predator initial energy | 5.0 |
| Prey initial energy | 3.0 |
| Offspring energy bounds (predator) | clipped to [1.0, 5.0] |
| Offspring energy bounds (prey) | clipped to [1.0, 3.0] |
| Reproduction reward (both) | 10.0 |
| Grass patches | 100, max energy 2.0, regrowth 0.04/step |
| Initial population | 6 predators + 8 prey |
| Max population pool | 200 predators + 1000 prey |

### Genome (Darwinian layer)
| Parameter | Value |
|---|---|
| Heritable trait | `offspring_investment_fraction` |
| Founder mean / std | 0.35 / 0.08 (both species) |
| Trait bounds | [0.10, 0.80] |
| Mutation rate | 0.05 per reproduction |
| Mutation std | 0.04 |

The investment fraction determines what proportion of the parent's current energy is transferred to each offspring at birth. It is inherited with Gaussian mutation — it is never directly observable by or accessible to the PPO policy. Selection operates entirely through differential reproductive success.

### PPO configuration
| Parameter | Value |
|---|---|
| lr | 5 × 10⁻⁵ |
| gamma | 0.99 |
| lambda | 0.95 |
| train batch / learner | 1024 steps |
| minibatch size | 128 |
| epochs | 6 |
| entropy coeff | 0.01 |
| env runners | 6 |
| KL target | 0.01 |

---

## 2. Overview: Three Training Runs

The `progress.csv` contains data from three consecutive training sessions, all writing to the same trial directory:

| Run | Date | Iterations | Env steps (est.) | Outcome |
|---|---|---|---|---|
| **R1** | 2026-06-26 21:27 → 23:22 | 1 – 59 (59 iters) | ~60K | Full learning trajectory, healthy ecology, peak ep_ret ~6300 |
| **R2** | 2026-06-27 08:56 → 10:24 | 51 – 93 (43 iters) | ~104K | Resumed from checkpoint ~iter 51; complete ecological collapse |
| **R3** | 2026-06-27 10:33 → 13:49 | 91 – 178 (88 iters) | ~194K | Resumed from checkpoint ~iter 91; slow recovery, plateau at ~310 |

R2 and R3 resume from different checkpoints of the same experiment. R2's investment fraction metrics are unreliable due to CSV column ordering differences on resume. R3 runs for as long as R1 (88 vs 59 iterations) but only reaches ep_ret ~314 — about 5% of R1's peak. Analysis of the Darwinian/Baldwinian interaction focuses on R1 where data quality is best.

---

## 3. Run 1 in Detail: Five Phases

### Phase 0 — Chaos (iters 1–7, env_steps 0–7K)

| iter | ep_ret | ep_len | n_ep | pred_spawn | prey_spawn |
|---:|---:|---:|---:|---:|---:|
| 1 | 181 | 79 | 7 | 0.00 | 20.2 |
| 2 | 204 | 85 | 15 | 0.66 | 27.8 |
| 7 | 167 | 75 | 0 | 0.65 | 27.8 |

- Episodes are very short (75–85 steps); only prey are reproducing (predators barely survive past their initial 5.0 energy)
- Policy entropy is near-maximum: predator 2.19, prey 2.19 nats — agents are acting almost uniformly at random
- VF explained variance is near 0 (value function has no predictive power yet)
- The rolling episode return averages ~180 — this is noise from very few completed episodes (many iterations show 0 new completions)
- Prey reproduction is active from the start (threshold=8 is achievable by eating 2 grass patches from 3.0 initial energy)

### Phase 1 — Rapid Learning (iters 8–22, env_steps 8K–23K)

This is the most dramatic period in R1. Episode length explodes from 75 to 982 steps in 14 iterations:

| iter | ep_ret | ep_len | pred_ent | prey_ent | live_pred | live_prey |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1167 | 294 | 1.95 | 1.98 | 12.2 | 29.0 |
| 10 | 1708 | 401 | 1.93 | 1.91 | 15.8 | 20.5 |
| 14 | 3058 | 604 | 1.79 | 1.78 | 16.8 | 21.1 |
| 18 | 4160 | 755 | 1.72 | 1.56 | 19.2 | 16.9 |
| 22 | 5935 | 982 | 1.61 | 1.41 | 18.9 | 18.1 |

The jump at iter 8 marks the moment the predator policy crosses a competence threshold: predators begin catching enough prey to accumulate energy above the 12.0 reproduction threshold. This initiates the predator reproduction loop, sharply increasing the mean episode reward.

Notable: **prey entropy drops faster than predator entropy** (2.19 → 1.41 vs 2.19 → 1.61 by iter 22). Prey are under stronger selection pressure — they are dying and need to evolve evasion faster. Predators improve hunting more gradually.

The live population self-organises from 6 predators + 8 prey to ~19 + 18 by iter 22 — a near-equal predator-prey balance emerging organically from random initialization.

### Phase 2 — Plateau (iters 22–44, env_steps 23K–45K)

Episode length saturates around the 930–986 step range (close to the 1000-step maximum). Returns stabilize between 6200 and 6370:

| iter | ep_ret | ep_len | pred_ent | prey_ent | live_pred | live_prey |
|---:|---:|---:|---:|---:|---:|---:|
| 22 | 5935 | 982 | 1.61 | 1.41 | 18.9 | 18.1 |
| 27 | 6226 | 930 | 1.32 | 1.37 | 19.2 | 17.0 |
| 33 | 6320 | 930 | 1.15 | 1.28 | 18.1 | 20.7 |
| 40 | 6236 | 942 | 0.95 | 1.30 | 19.6 | 18.1 |
| 44 | 6368 | 986 | 1.18 | 1.15 | 17.0 | 20.1 |

Policy entropy continues to decline — agents are committing more strongly to learned strategies. Predator entropy falls from 1.61 to ~0.95 by iter 40. This is the PPO/Baldwinian phase consolidating.

The ecosystem holds a remarkably stable dynamic equilibrium: predators and prey counts oscillate in the 15–20 range with no extinction events. Reproduction rates rise steadily (predator spawns 5 → 10 per episode, prey spawns 40 → 57).

### Phase 3 — Decline (iters 44–59, env_steps 45K–60K)

A gradual deterioration begins after iter 44:

| iter | ep_ret | ep_len | pred_ent | prey_ent | pred_spawn | prey_spawn |
|---:|---:|---:|---:|---:|---:|---:|
| 44 | 6368 | 986 | 1.18 | 1.15 | 10.6 | 56.7 |
| 48 | 5841 | 942 | 1.00 | 1.11 | 11.5 | 59.3 |
| 52 | 5235 | 871 | 0.76 | 1.05 | 12.5 | 62.3 |
| 56 | 4839 | 803 | 0.89 | 1.05 | 13.4 | 64.6 |
| 59 | 4715 | 773 | 0.86 | 1.02 | 13.9 | 66.1 |

This decline is counterintuitive: predator spawn rates are *rising* (10.6 → 13.9) but episode lengths are *falling*. What is happening is that increased predator efficiency is destabilizing the predator-prey balance — prey are being consumed faster than they can replenish. The PPO policy for predators is over-optimized for short-term hunting, eroding the long-run ecological base. This is a classic overharvesting dynamic in a learned predator-prey system.

Prey entropy (1.05) remains above predator entropy (0.86), consistent with prey still searching for better evasion strategies while predators have partially converged on aggressive pursuit.

---

## 4. Darwinian Evolution of the Investment Fraction

The investment fraction (`offspring_investment_fraction`) is the sole heritable trait. It is initialized from a Gaussian with mean 0.35 and std 0.08, clipped to [0.10, 0.80]. Every reproduction event may mutate the offspring's trait by Gaussian noise (rate 0.05, std 0.04).

### Observed drift over 59 iterations (R1)

| Species | iter 1 mean | iter 59 mean | Δmean | iter 1 std | iter 59 std | Δstd |
|---|---:|---:|---:|---:|---:|---:|
| Predator | 0.3408 | 0.3353 | −0.0055 | 0.0712 | 0.0696 | −0.0016 |
| Prey | 0.3281 | 0.3220 | −0.0061 | 0.0718 | 0.0666 | −0.0052 |

The full trajectory (selected iterations):

| iter | pred_mean | pred_std | pred_p25 | pred_p50 | pred_p75 | prey_mean | prey_std | prey_p25 | prey_p50 | prey_p75 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.3408 | 0.0712 | 0.2759 | 0.3604 | 0.3854 | 0.3281 | 0.0718 | 0.2627 | 0.3046 | 0.4004 |
| 10 | 0.3352 | 0.0706 | 0.2727 | 0.3428 | 0.3821 | 0.3248 | 0.0682 | 0.2622 | 0.3051 | 0.3917 |
| 20 | 0.3350 | 0.0703 | 0.2729 | 0.3423 | 0.3817 | 0.3244 | 0.0679 | 0.2626 | 0.3049 | 0.3906 |
| 30 | 0.3352 | 0.0702 | 0.2734 | 0.3423 | 0.3821 | 0.3236 | 0.0675 | 0.2624 | 0.3044 | 0.3891 |
| 40 | 0.3350 | 0.0701 | 0.2734 | 0.3420 | 0.3820 | 0.3231 | 0.0670 | 0.2627 | 0.3041 | 0.3875 |
| 50 | 0.3352 | 0.0699 | 0.2739 | 0.3421 | 0.3820 | 0.3226 | 0.0668 | 0.2627 | 0.3040 | 0.3861 |
| 59 | 0.3353 | 0.0696 | 0.2746 | 0.3420 | 0.3820 | 0.3220 | 0.0666 | 0.2624 | 0.3037 | 0.3848 |

### Interpretation of the drift

**Direction (lower is fitter).** Both species drift toward lower investment fractions. The ecological mechanism is straightforward: a parent that invests less energy per offspring retains more energy after reproduction, enabling it to reach the reproduction threshold again sooner. Under pure r-strategy pressure, the optimal trait would converge toward the lower bound (0.10). The fact that it hasn't reached that bound in 59 iterations reflects the tension between:
- Offspring viability: offspring that receive less energy (less than 1.0 is clipped to min, but values near 1.0 survive poorly as they have less margin before starvation)
- Parent survival: parent retains more energy but still must survive

**Asymmetry between species.** Prey drift is slightly faster (Δ = −0.0061) than predator drift (Δ = −0.0055), and prey std compresses more strongly (Δstd = −0.0052 vs −0.0016). This is consistent with prey having ~5× higher reproduction rate (prey spawn: 20–66 per episode vs predator spawn: 0.6–14), giving far more generational turnover and hence faster natural selection per training iteration.

**The p25-p50-p75 structure.** Prey show a pronounced rightward skew in the founder distribution (p50 = 0.305 << mean = 0.328), which compresses over time (prey_p75: 0.400 → 0.385, mean: 0.328 → 0.322). This shows the high-investment tail being trimmed by selection. Predators show less skew (their trait distribution is more symmetric and changes more slowly).

**Rate of drift.** Approximately 0.0001 per iteration for both species. At this rate, reaching the lower bound (0.10) from the founder mean (0.35) would require ~2500 iterations. This confirms that 59 iterations is very early in the Darwinian process.

---

## 5. Baldwinian-Darwinian Interaction

The **Baldwinian (PPO) layer** learns within an agent's lifetime: policy weights update via gradient descent, allowing agents to improve their behavior over the course of training. The **Darwinian layer** acts across generations: the `offspring_investment_fraction` genome is selected over reproduction events.

These two processes interact in this design:

### Adaptive landscape shaping

The PPO policy determines how efficiently agents survive and accumulate energy. As the policy improves, the reproductive payoff landscape changes:

- In Phase 0 (random policy): most agents never reach the reproduction threshold at all — there is no meaningful selection on investment fraction yet, just drift + mutation
- In Phase 1 (rapid learning): predators start reproducing. Now selection can act: an agent whose genome gives it offspring with higher starting energy has more surviving lineage
- In Phase 2 (plateau): the policy has stabilized around a competent hunting/evasion strategy. The fitness advantage of a given investment fraction is now relatively deterministic — selection acts more consistently

This is the core Baldwin coupling: **the genome drift direction and rate change as the policy learns**. A random-walking policy creates a noisy selection landscape; a competent policy creates a clear one.

### Evidence from R1

**1. Std compression coincides with rapid learning.**
Policy entropy drops most steeply in iters 8–22 (predator: 2.19 → 1.61). During this same window, prey_inv_std falls from 0.0718 to 0.0677 — a compression of 0.0041. In the subsequent plateau (iters 22–59), it falls a further 0.0011. The Darwinian signal is stronger during active policy learning than during the plateau.

**2. Direction of drift is consistent with PPO-discovered strategy.**
A trained predator policy that can hunt effectively makes it *optimal* for parents to reproduce multiple times with moderate-sized offspring rather than once with a large offspring. Lower investment fraction supports this: parent retains energy above threshold for a second reproduction. The genome drift toward lower fractions is thus in the direction that the learned policy makes adaptive. This is consistent with the Baldwin effect: learning renders the low-fraction genotype more fit.

**3. Species-specific coupling asymmetry.**
Prey drift faster precisely because they reproduce more — the feedback loop between policy competence and genome selection runs through reproduction frequency. Predator policy learning is slower (their entropy at iter 59 is still 0.86, not fully converged) and predator reproduction is rarer, so predator genome selection runs more slowly. This double asymmetry (slower policy learning + fewer reproductions) explains why predator Δstd (−0.0016) is so much smaller than prey Δstd (−0.0052).

**4. What is not yet visible.**
True genetic assimilation — where a learned behavior becomes genetically canalized — would require hundreds more iterations. The current data shows the *preconditions* for the Baldwin effect: (a) learning is clearly occurring, (b) the genome is evolving, (c) the direction of genome evolution is consistent with the fitness landscape shaped by learning. The actual causal link between policy state and genome selection rate would require ablation experiments (e.g., a control run with genome disabled or frozen).

### Reproduction energy mechanics

The investment fraction translates to offspring energy as:
```
offspring_energy = clip(parent_energy × fraction, min_offspring_energy, max_offspring_energy)
```

From the R1 data:

| iter | pred_repro_energy_inv | prey_repro_energy_inv | pred_parent_E_after | prey_parent_E_after |
|---:|---:|---:|---:|---:|
| 1  | 0.007 | 2.727 | 0.021 | 5.871 |
| 10 | 0.255 | 2.711 | 1.675 | 5.886 |
| 22 | 0.330 | 2.707 | 1.835 | 5.890 |
| 40 | 0.450 | 2.697 | 2.064 | 5.897 |
| 59 | 0.598 | 2.687 | 2.356 | 5.902 |

**Prey:** Reproduction energy invested is remarkably stable at ~2.7 units throughout, despite the policy improving substantially. This is because prey consistently reach threshold with ~8.5–9 energy and fraction ~0.32, giving 2.7–2.9 units invested. Prey parent energy after reproduction stays near 5.9 — prey consistently retain ~70% of their energy after reproducing, giving them a large buffer for survival.

**Predators:** Reproduction energy invested rises from near-zero (early training, very few reach threshold) to 0.6 by iter 59. The low absolute value reflects the reality that most predators in early–mid training die without reproducing: the mean is diluted by the many agents who contribute 0. Parent energy after reproduction rises from 0.021 to 2.356 — predators at iter 59 are reproducing with ~12–14 energy and retaining a meaningful buffer.

The contrast is instructive: prey are efficient reproducers from early training (their policy for finding grass is simpler). Predators are late reproducers — they require substantially more learned competence to accumulate the 12.0 threshold energy. This gap drives the faster prey genome evolution observed above.

---

## 6. Run 2: Ecological Collapse After Checkpoint Resume

Run 2 resumed from the checkpoint saved at approximately iter 51 of Run 1. The results were immediately catastrophic:

| iter (R2) | ep_ret | ep_len | n_episodes |
|---:|---:|---:|---:|
| 51 | 0.26 | 11.8 | 60 |
| 53 | 0.26 | 70.0 | 178 |
| 56 | 0.26 | 146 | 415 |
| 80–93 | 0.26–0.27 | 145–147 | 414–416 |

Three observations stand out:

**Collapse is immediate.** At iter 51 in R1, the ecosystem had ep_len ≈ 890 and ep_ret ≈ 5398. After resuming from the iter 51 checkpoint with new random seeds, episode length drops to 11.8 steps and return to 0.26. The policy weights themselves appear intact (loaded from checkpoint), but the ecological dynamics are completely different.

**Plateau is absolute.** Over 43 further iterations (iter 51 to 93), episode return stays within [0.26, 0.28] and episode length stabilizes at ~146 steps. There is no meaningful learning gradient — the PPO has no reward signal to act on, and the ecosystem never recovers enough to allow reproduction. This constitutes a dead attractor: the policy is stuck and the environment provides no escape signal.

**Root cause hypothesis.** The policy checkpoint at iter 51 was trained on specific statistical regularities arising from particular random seeds. When exposed to fresh random seeds, the population dynamics diverge immediately: predators catch prey so quickly in the short (12-step) initial episodes that the prey population collapses before any reproduction occurs. With no prey remaining, predators also starve. The episode then resets and the same collapse repeats. The policy, trained to hunt aggressively, is too effective in the new seed context — a failure mode consistent with overfitting to the specific dynamical patterns encountered during R1.

The genome metrics for R2 are unreliable due to column ordering differences in the resumed run's CSV metrics output. No Darwinian analysis is possible from R2.

---

## 7. Run 3: Slow Recovery and New Plateau (iters 91–178)

Run 3 resumed from the checkpoint saved at approximately iter 91 of R2 and ran for 88 iterations, matching R1's duration. It shows a qualitatively different trajectory.

### Episode return trajectory

| isr | iter | ep_ret | n_ep | est. ep_len |
|---:|---:|---:|---:|---:|
| 1  | 91  | 423 | 5.9 | 174 |
| 2  | 92  | 111 | 6.0 | 170 |
| 5  | 95  | 218 | 6.0 | 170 |
| 6  | 96  | 298 | 6.0 | 170 |
| 10 | 100 | 300 | 6.0 | 170 |
| 20 | 110 | 302 | 6.0 | 170 |
| 30 | 120 | 304 | 6.0 | 170 |
| 40 | 130 | 307 | 6.0 | 170 |
| 50 | 140 | 309 | 6.0 | 170 |
| 60 | 150 | 311 | 6.0 | 170 |
| 70 | 160 | 311 | 6.0 | 170 |
| 80 | 170 | 313 | 6.0 | 170 |
| 88 | 178 | 314 | 6.0 | 170 |

`episode_len_mean` reports 0.0 due to a metric column issue on resume. The estimated episode length of ~170 steps is derived from `num_env_steps_sampled` (1024 per iteration) ÷ `num_episodes` (~6.0), and is consistent throughout all 88 iterations.

### Three phases of R3

**R3-Phase 1 — Burst and crash (isr 1–5):** The very first iteration opens with ep_ret = 423, likely from one or two lucky long episodes. By isr 2 the ecosystem collapses back to ep_ret = 111 — the cold-start fragility documented in R2 re-appears briefly. Over isr 3–5 it recovers to 218.

**R3-Phase 2 — Rapid stabilization (isr 6–10):** Episode return jumps from 218 to ~300 and episode count stabilizes at exactly 6 per iteration. This phase takes only ~5 iterations, much faster than R2's 43-iteration dead plateau. The policy checkpoint at iter ~91 (from R2) had developed sufficient cold-start competence to avoid re-entering the death attractor.

**R3-Phase 3 — Slow creep (isr 11–88):** Returns inch from 300 to 314 over 78 iterations — an average gain of ~0.18 per iteration. The ecosystem is in a stable but suboptimal equilibrium: 170-step episodes with ~31 reproduction events each, compared to R1's 930-step episodes with ~630 reproduction events.

### Why R3 is stuck

The gap is instructive. R3 ep_ret of 314 at ~170 steps implies ~31 reproduction events per episode. R1 peak ep_ret of ~6300 at 930 steps implies ~630 reproduction events. R3 generates roughly **5% of R1's reproductive throughput**.

The mechanism: R3 episodes are short because cold-start dynamics dominate. With 6 predators and 8 prey on fresh seeds, the aggressive predator policy quickly depletes prey before the population can self-organise into the 18–20 agent equilibrium that R1 sustained. Episodes end after ~170 steps, reset, and the cycle repeats. The policy cannot break out of this cycle because it was shaped by R2's collapsed dynamics — it has never encountered the extended mid-episode ecology that R1 learned.

**Predator energy dynamics in R3 (live_investment metrics):**
- Predator `live_investment_fraction_mean`: 0.310 (isr 1) → 0.278 (isr 6) → 0.281 (isr 88) — stable, below R1's founder mean (0.35), consistent with continued downward Darwinian drift
- Prey `live_investment_fraction_mean`: consistently 0.007 throughout — clearly a column misalignment artifact; not interpretable

**Metric reliability note:** In addition to `episode_len_mean = 0`, the PPO entropy columns show implausible values (up to 45 nats, vs the theoretical max of ln(9) ≈ 2.2) during the early R3 iterations, confirming metric column corruption on resume. By isr ~30 the values settle to 0.41 (predator) and 0.97 (prey), which are individually plausible but also suspect. No Darwinian genome drift analysis is possible from R3 metrics.

### Comparison: R1 vs R3 at equivalent iteration count

| Metric | R1 at iter 59 | R3 at iter 178 (88 isr) |
|---|---:|---:|
| ep_ret | 4715 | 314 |
| est. ep_len | 773 | ~170 |
| repro events / episode | ~47 | ~31 |
| repro events / step | 0.061 | 0.182 |

The per-step reproduction rate is actually *higher* in R3 than in late R1 (0.182 vs 0.061). Within the short 170-step episodes that do occur, the ecosystem is reasonably productive. The problem is episode length, not within-episode performance — confirming that the bottleneck is the cold-start transition, not the policy quality once the ecosystem is established.

---

## 8. Summary of Findings

### Baldwinian (PPO) learning — confirmed

Run 1 shows clean, rapid policy improvement. Predator entropy drops from 2.19 to 0.86 over 59 iterations; prey entropy drops from 2.19 to 1.02. The value function shows consistently positive but low VF explained variance (~0.05–0.10), a known artifact of multi-agent partial observability where return variance is dominated by other agents' behavior. The rapid episode length growth (75 → 982 steps in 14 iterations) confirms genuine behavior improvement.

### Darwinian (genome) evolution — confirmed, slow

Both species show directional drift of the investment fraction toward lower values:
- Predator: 0.3408 → 0.3353 (−0.0055) over 59 iterations
- Prey: 0.3281 → 0.3220 (−0.0061) over 59 iterations

The standard deviation compresses in both species (most pronounced in prey), indicating natural selection is narrowing the trait distribution around a fitness-optimal value. The direction (lower investment) is ecologically consistent: smaller per-offspring investment enables higher lifetime reproductive output.

### Baldwinian-Darwinian coupling — early evidence, not yet conclusive

The evidence is suggestive rather than definitive at 59 iterations:
- Genome std compression is strongest during the phase of rapid policy improvement (iters 8–22), consistent with learning sharpening the selection landscape
- The direction of genome drift is the direction a learned (PPO-optimized) agent would make advantageous
- Prey show faster genome evolution than predators, mirroring the asymmetry in policy learning speed and reproduction frequency
- Formal confirmation would require a control run with genome disabled (fixed founder fraction) to isolate the Darwinian contribution, and far more iterations (~500+) for the signal to strengthen

### Critical vulnerability: checkpoint fragility — confirmed and partly understood

Runs 2 and 3 confirm that a policy trained on specific random seeds collapses or degrades sharply when resumed under new seeds. The root cause is distributional shift: the policy trained in R1 only encountered mid-episode dynamics (large stable populations), never the sparse cold-start state. On resume, every env runner resets from scratch, and the aggressive policy drives prey extinct in ~12 steps (R2) or ~170 steps (R3).

R3 further shows that even 88 additional training iterations are insufficient to escape this sub-optimal equilibrium — the policy slowly improves (300 → 314 over 78 iterations) but cannot break through to the long-episode regime.

**Mitigations now implemented for the next training run:**
1. **Randomised initial population** — `reset()` now draws initial counts from [2, 6] predators and [2, 8] prey (uniform, seeded per episode). The policy will encounter sparse cold-start states during all future training, including R1-equivalent fresh runs.
2. **Entropy warmup on resume** — `resume_training_investment.py` sets `entropy_coeff = 0.05` (vs. the training default of 0.01) to encourage exploratory behaviour during the fragile cold-start phase after checkpoint restore.

### Current status and outlook

At ~194K environment steps (178 total iterations across three runs), the experiment has:
- Confirmed clean Baldwinian learning in R1 (75 → 982 ep_len in 14 iterations)
- Confirmed directional Darwinian genome drift in R1 (investment fraction declining in both species)
- Identified and partially fixed the cold-start fragility that caused R2 collapse and R3 sub-optimal plateau

The target for the next run (R4) is to re-enter the R1-equivalent long-episode regime (ep_len > 500, ep_ret > 2000) under the new randomised initialisation. If achieved, the R4 data will provide investment-fraction genome trajectory data from a more exploration-robust policy, enabling a more rigorous test of the Baldwinian-Darwinian coupling hypothesis.

**Correction (added after the `eco_evolutionary_metabolic_rate` line concluded):** the "confirmed directional Darwinian genome drift" claim above was based on a single 59-iteration run with no neutral-drift control. The same kind of single-run drift read looked equally "confirmed" for `metabolic_rate` early on and turned out, after a proper 3-seed-each real-vs-control replication (Mann-Whitney U), to be statistically indistinguishable from pure mutation + finite-population noise. Treat R1's drift claim as an unverified early signal, not an established result, until R4+ actually tests it against a control. See `predpreygrass/evolutionary/RESULTS.md` for the full cross-module trial log.

---

## Resuming with the validated methodology — R4 onward

**Why now, and why here.** `eco_evolutionary_metabolic_rate` (Iterations 0-6) is the module where
the actual rigorous methodology got built: a biologically-grounded satiation throttle for
sustainability, a `genome_neutral_drift_control` flag to isolate real selection from neutral
drift, and a 3-seed-each replication compared via Mann-Whitney U. Applied to `metabolic_rate`,
that methodology returned a null result — no drift-magnitude gap between real and control, for
either species, at two different fitness-gradient steepnesses, and no individual-level
MR-vs-reproduction correlation either. `offspring_investment_fraction` was never given that same
test — R1's drift signal was larger, in far fewer iterations, than anything `metabolic_rate` ever
produced, and work stopped for an unrelated engineering bug (checkpoint-resume collapse, see
above), not because the trait looked bad. It's the more promising untested candidate, not a
fallback.

**Numbering note:** from here on this log uses the same flat convention as
`metabolic_rate`'s "Iteration N" — one R-number per distinct trial/run/config, no sub-numbering.
(An earlier draft of this section nested three trials inside a single "R4" as "Pilot 1/2" and
"Step 2"; renumbered below so R4, R5, R6... each mean exactly one trial, consistent with the rest
of this project's documentation. See `predpreygrass/evolutionary/RESULTS.md` for how "Trial"
at the cross-module level relates to "R"/"Iteration" at the per-module level.)

**Plan, in order (R-numbers assigned as each trial actually runs):**

1. **Port the satiation throttle from `metabolic_rate`.** (→ R4, R5 below.) Done —
   `predator_satiation_cooldown` and `max_energy_gain_per_prey` added to
   `config_env_eco_evolutionary.py`, and the cooldown + per-catch cap logic ported into
   `_handle_predator_engagement` in `predpreygrass_rllib_env.py` (same starting values as
   `metabolic_rate` Iteration 2, since the base energy economy here matches that module closely).

2. **A reverse-leg test, done early and cheap, before any replication.** (→ R6 below.) Freeze
   the genome at several fixed values (`genome_enabled: False`, sweep `founder_genome` mean
   across the trait bounds) and check whether fitness outcomes (episode length, reproduction
   events, survival) vary across values at all. Pure config sweep plus evaluation —
   `genome_enabled` and `founder_genome` are already config-level knobs (config lines 30-38), no
   new code needed. This is the check `metabolic_rate` skipped from Iteration 0 onward (its own
   original next-steps list called it "Priority 1" and it was never actually done, only
   approximated later by the `mr_repro_spearman` proxy metric — which turned out to carry no
   signal anyway). If outcomes are flat here too, stop before spending compute on a full
   replication.

3. **If R6 shows a real fitness gradient (→ R7, planned):** port the neutral-drift control (a
   `genome_neutral_drift_control` flag plus a dedicated `tune_ppo_investment_neutral_control.py`,
   mirroring `metabolic_rate`'s), then run the same 3-real + 3-control-seed, 1000-iteration
   replication, compared via Mann-Whitney U.

4. **If that looks promising (→ R8, planned):** the expensive step — train under different
   frozen genome regimes and compare *learned policy behavior*, not just fitness outcome, for the
   strongest version of the reverse-leg claim (does evolution measurably reshape what RL learns,
   not just which outcomes result).

### R4 — satiation-throttle pilot, 100 iterations (complete, inconclusive)

**Config:** the throttle port (`predator_satiation_cooldown: 8`, `max_energy_gain_per_prey:
8.0`), `--max-iters 100`. Run: `PPO_ECO_EVOLUTION_INVESTMENT_2026-07-14_23-17-11`.

**Result:** 100/100 iterations, no errors. Episode length grew substantially early on (36 → 48 →
80 → 95 → 120 steps across iterations 1-8) — a good early sign the throttle is preventing rapid
predator-driven episode termination, same direction as `metabolic_rate` Iteration 2. But the
predator:prey ratio (from the always-populated step-level `live_investment` counts, since
episode-level metrics go NaN past iteration ~8 — same Ray `Stats`-reducer cadence artifact
documented in `metabolic_rate`'s Iteration 6, more pronounced here because this module has only 7
parallel env runners vs. `metabolic_rate`'s 29) climbed steadily the whole run with no sign of
leveling off:

| iter | predators | prey | ratio |
|---|---|---|---|
| 10 | 7.8 | 56.0 | 0.14 |
| 30 | 13.8 | 39.7 | 0.35 |
| 60 | 17.2 | 28.5 | 0.60 |
| 100 | 18.6 | 18.8 | **0.99** |

**Verdict: inconclusive.** Promising on episode length, but the ratio trending toward 1:1 with no
plateau is worse than `metabolic_rate`'s eventual stabilized ratio (~0.5-0.6). Can't yet tell
whether this is a genuine overshoot (throttle constants need retuning for this trait's different
post-reproduction energy balance, as flagged when porting) or just needs more runway —
`metabolic_rate` itself took several hundred iterations to visibly stabilize under the same
throttle constants, and this pilot is only 1/10th that length.

**Adjustment → R5:** extend rather than conclude — a longer pilot to see whether the ratio
plateaus or keeps climbing toward a crash.

### R5 — satiation-throttle pilot, 400 iterations (complete)

**Config:** identical to R4, `--max-iters 400`, fresh run (not resumed — same seed-fresh
approach as `metabolic_rate` throughout). Run:
`PPO_ECO_EVOLUTION_INVESTMENT_2026-07-15_08-32-43`.

**Launched:** 2026-07-15 08:32. **Finished:** 2026-07-15 12:59, 400/400 iterations, no errors,
~4h26m.

**Result: the R4 concern doesn't hold up — the ratio plateaus into an oscillating band
rather than climbing toward a crash.**

| checkpoint | predators | prey | ratio |
|---|---|---|---|
| iter 100 | 19.1 | 21.8 | 0.87 |
| iter 150 | 19.5 | 20.7 | 0.94 |
| iter 200 | 20.6 | 17.8 | 1.16 (peak) |
| iter 250 | 19.0 | 19.2 | 0.99 |
| iter 300 | 15.7 | 21.5 | 0.73 |
| iter 350 | 18.9 | 20.4 | 0.93 |
| iter 400 | 16.5 | 24.8 | 0.66 |

Quintile-averaged ratio: Q1 0.38 → Q2 0.75 → Q3 0.80 → Q4 0.91 → **Q5 0.77**. It climbed through
Q1-Q4, matching the trajectory R4 alone suggested — but Q5 turned back down rather than
continuing to climb. Fine-grained checkpoints from iter 200 onward show noisy oscillation roughly
in the 0.6-1.2 range, not a monotonic trend in either direction. Absolute population sizes stay
healthy throughout (predators ~14-21, prey ~16-40) — no near-zero excursions at any point, i.e.
no close call with collapse even during the iter-200/260 peaks.

**Verdict: cautiously validated.** This reads as noisy predator-prey oscillation (Lotka-Volterra
character — real ecological systems cycle, they don't sit flat) rather than unbounded predator
overshoot. Less cleanly stable than `metabolic_rate`'s eventual near-flat plateau, but not the
slow-motion crash R4's endpoint alone suggested — that endpoint (ratio 0.99 at iter 100)
turned out to be mid-cycle, not the start of a runaway trend. The throttle constants
(`predator_satiation_cooldown: 8`, `max_energy_gain_per_prey: 8.0`, ported unchanged from
`metabolic_rate`) are good enough to proceed without retuning; revisit only if R7's full
replication shows episode-completion rates that don't clear a reasonable bar.

**Adjustment → proceed to R6** (fixed-genome fitness sweep) without further sustainability
tuning.

---

**Status:** R4 + R5 (satiation throttle) validated.

### R6 — fixed-genome fitness sweep (complete)

**Config:** `offspring_investment_fraction` frozen at 5 values spanning the trait bounds (0.10,
0.80) — 0.15, 0.25, 0.35 (founder default), 0.55, 0.70 — via a new `--fixed-investment-fraction`
CLI arg on `tune_ppo_investment.py` (sets `genome_enabled=False`, which makes
`_get_offspring_investment_energy` fall back to `founder_genome`'s mean for every agent, every
reproduction — no inheritance, no mutation, no per-agent variation). 100 iterations each,
sequential, via the new `run_fixed_genome_sweep.sh`. Everything else identical to the validated
Phase-A config (satiation throttle included).

**Rationale:** tests whether the trait affects fitness at all, independent of any
selection/drift question — this is the check `metabolic_rate` skipped from Iteration 0 onward
(see `predpreygrass/evolutionary/RESULTS.md`). If outcomes are flat across all 5 values, that's a
cheap, early signal to reconsider before spending compute on the full replication (R7). If
outcomes vary meaningfully, there's a real landscape for selection to act on.

**Launched:** 2026-07-15 18:45, via `run_fixed_genome_sweep.sh` (5 runs × 100 iterations
sequential on one GPU).

**Bug found and fixed mid-run (2026-07-15 19:47).** The first sweep run (0.15) hit 146 env-runner
crashes in 37 iterations (`Worker exits with an exit code 1`, `IndexError: list index out of
range` inside RLlib's `episode.get_rewards()`/`inf_lookback_buffer.get()`) — Ray auto-recovered
each time so the trial kept running, but at heavy cost: constant worker respawns explain most of
the pace slowdown seen since this sweep started (~90-190s/iter vs. ~40-50s/iter in R4/R5).
Root cause: `utils/episode_return_callback.py`'s `on_episode_end` called
`episode.get_rewards()`, which indexes a global-step-aligned lookback buffer that goes out of
range for agents whose local step count diverges from the episode's — exactly what short-lived
offspring under a low fixed investment fraction produce. `metabolic_rate`'s callback uses a
different RLlib API (`episode.agent_episodes.items()` + per-agent `get_return()`) that sidesteps
this global indexing entirely, and has never shown this crash across thousands of iterations —
ported that pattern into `episode_return_callback.py` here. Smoke-tested (0 crashes in 3
iterations, vs. dozens under the old code), then the aborted 0.15 run was killed and the full
5-run sweep relaunched clean.

**This bug predates R6** — grepping R4's and R5's raw logs shows the identical crash (138
occurrences in R4's 100 iterations, 488 in R5's 400), never noticed before because monitoring
only checked `ps`/`progress.csv`/`error.txt`, not log content for mid-run errors. Their
conclusions are not affected: both relied on `live_investment/*` metrics, which are logged via
the separate, unaffected `on_episode_step` path, not the buggy `on_episode_end` path. But it
does mean R4/R5 ran slower and with more episode-data loss than necessary — worth knowing if
their exact wall-clock timings are ever compared against a clean run.

**All 5 runs completed 2026-07-15 23:43, 100/100 iterations each, zero crashes after the fix
(vs. 146 in the first, aborted 0.15 attempt).**

**Result: fitness outcomes are not flat — there's a real shift, concentrated between the lowest
value and everything above it.** Late-run (second-half) averages, using
`peak_active_predators`/`peak_active_prey` as the fitness/survival proxy (population-level
outcome, not genome-tracking metrics — `live_investment/*` reports 0 throughout this sweep since
those are built by iterating `agent_genomes`, which stays empty under `genome_enabled: False`;
not a bug, just the wrong metric family for this config):

| fixed value | ep_len | peak predators | peak prey | predator spawned | prey spawned |
|---|---|---|---|---|---|
| 0.15 | 912.8 | **14.4** | **73.7** | 90.6 | 555.4 |
| 0.25 | 1000.0 | 24.6 | 57.2 | 155.0 | 550.3 |
| 0.35 | 1000.0 | 19.2 | 51.7 | 96.0 | 427.8 |
| 0.55 | 991.1 | 27.0 | 47.9 | 138.2 | 502.5 |
| 0.70 | 1000.0 | 26.6 | 53.0 | 176.7 | 534.7 |

Sustainability holds at every value (episode length 913-1000, no crashes at any fixed point).
But the predator:prey balance shifts clearly: at the lowest investment (0.15), predators peak
much lower (14.4) and prey much higher (73.7) than at any of the other four values (predators
19-27, prey 48-57). From 0.25 upward the picture is noisier and doesn't show a clean further
monotonic gradient — most of the visible effect is concentrated in the low-investment regime,
more like a threshold/step than a smooth dose-response across the full range. Sample sizes are
modest (n=15-28 non-NaN iterations per value in the later half, due to the same
episode-completion-cadence NaN pattern documented elsewhere in this file) — real, but noisy.

**Verdict: not flat.** There is a real landscape here — `offspring_investment_fraction` does
change fitness/ecological outcomes when frozen at different values, independent of any selection
question. This is the opposite of a stop signal: R6 existed specifically to catch a flat result
before spending R7's much larger compute budget, and it didn't find one.

**Adjustment → proceed to R7.** Port the neutral-drift control (`genome_neutral_drift_control`
flag + dedicated neutral-control tune script, mirroring `metabolic_rate`'s) and run the 3-real +
3-control-seed, 1000-iteration replication compared via Mann-Whitney U — the test that actually
answers whether selection drives genome drift beyond neutral noise, now on a trait confirmed to
have real fitness leverage to select on.

**Status:** R6 complete. R7 in progress (see below).

### R7 — neutral-control replication (in progress)

**Config:** ported the neutral-drift control mechanism from `metabolic_rate` —
`genome_neutral_drift_control` flag (config + `__init__`) and the corresponding branch in
`_inherit_genome` (offspring genome template becomes a uniformly random currently-alive
same-species agent instead of the true parent, when the flag is set; reproduction eligibility,
timing, and energy dynamics unchanged). New files: `config/config_env_eco_evolutionary_neutral_control.py`,
`tune_ppo_investment_neutral_control.py`, `analyze_replication_seeds.py` (Mann-Whitney U on
drift magnitude, same design as `metabolic_rate`'s), `run_replication_seeds.sh` (3 real + 3
control seeds, 1000 iterations each, console-log auto-archiving built in from the start this
time). Smoke-tested (3 iterations, confirmed `genome_neutral_drift_control: True` engages
correctly, no crashes) before launching the full run.

**Launched:** 2026-07-16 10:14, via `run_replication_seeds.sh` (real seeds 42/43/44 then control
seeds 42/43/44, sequential on one GPU, 1000 iterations each).

**Status:** in progress (real seed 42 running as of launch). R8 not started.

---

*Analysis date: 2026-06-27. Data source: `~/ray_results/PPO_ECO_EVOLUTION_INVESTMENT_2026-06-26_21-25-01/`*
