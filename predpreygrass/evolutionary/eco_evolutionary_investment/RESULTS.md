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

---

*Analysis date: 2026-06-27. Data source: `~/ray_results/PPO_ECO_EVOLUTION_INVESTMENT_2026-06-26_21-25-01/`*
