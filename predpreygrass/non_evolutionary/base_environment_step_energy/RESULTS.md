# base_environment_step_energy: investigation log and results

This documents the reasoning and results behind `base_environment_step_energy`, in the order they actually happened, including the mistakes and corrections along the way. See [`README.md`](./README.md) for the module's structure and how to run it; this file is the narrative behind the numbers.

**Naming.** In this document `base_environment_step_energy` means the module at its current default settings (homeostatic cost 0.10 / 0.035, predator move cost 0.08, prey move cost 0.035). While it was being tuned (sections 10-12) this setting was called "run B"; "run A" is the earlier setting with predator move cost 0.10.

**The configurations compared, and how their energy costs differ** (energy per step; each cell is predator / prey):

| configuration | cost per step when resting (noop) | cost per step when moving | extra cost of moving over resting |
|---|:---:|:---:|:---:|
| `base_environment` | 0.15 / 0.05 | 0.15 / 0.05 | 0 / 0 (one flat tax, same whatever the action) |
| `base_environment_step_energy` (default) | 0.10 / 0.035 | 0.18 / 0.07 | +0.08 / +0.035 |
| run A (earlier setting, section 10) | 0.10 / 0.035 | 0.20 / 0.07 | +0.10 / +0.035 |
| earlier free-resting run (a design since removed from the code, sections 5-9) | 0 / 0 | 0.15 / 0.05 | +0.15 / +0.05 |

So relative to `base_environment`, the default module makes resting cheaper (0.10 vs 0.15 for a predator) and moving more expensive (0.18 vs 0.15): the difference between the two configurations is the *extra cost of moving*, which is zero in `base_environment` and +0.08 (predator) / +0.035 (prey) here. Everything else (grid, observations, rewards, reproduction thresholds, grass) is identical.

## 1. Starting question

`base_environment` charges `energy_loss_per_step_predator`/`_prey` unconditionally every step, regardless of the action taken — an agent that picks noop pays exactly the same tax as one that moves. The question: what happens if movement itself costs energy and standing still (noop) doesn't, while still keeping *some* form of the existing per-step tax present (per the original framing: "the energy_loss_per_step_predator/prey must also be present, but probably less so")? Would the ecosystem still be sustainable, and how far could the noop-vs-move cost gap be pushed?

`base_environment`'s `_get_movement_energy_cost` hook already existed for exactly this purpose but was stubbed to always return 0 — dead code. This module implements it for real.

## 2. Design: the move_fraction parametrization

Rather than pick one arbitrary split, `tune_ppo_base_environment_step_energy.py` exposes `--move-fraction f` (0-1): the environment's *fixed total* per-step energy budget (0.15 predator / 0.05 prey — identical total to `base_environment`) splits into an unconditional flat part `total × (1-f)` and a move-only part `total × f`. `f=0` reproduces `base_environment` exactly (noop pays full tax). `f=1` makes noop entirely free (the whole budget is move-conditional). An agent that moves every single step pays the same total at every `f` — only the cost of *standing still* changes.

## 3. First pass: 100-iteration sweep (2026-09-15/16)

Three probes, seed 42, `f ∈ {0.5, 0.75, 1.0}`, 100 iterations each. Rationale for 100 iterations: cheap enough to run three of them (~4 hours total) as an initial screen before committing to the ~6+ hour cost of a full run, on the assumption that early collapse would show up quickly if the config were unsustainable.

**Result at the time**: all three looked sustainable. Shape was consistent across all three: an early-training instability phase (some predator die-off while the near-random policy learns to survive, matching `base_environment`'s own documented early-training behavior) that resolved by iteration ~30-40 into 0% extinction, full 1000-step episodes, and stable-looking populations (~20 predators / ~19-22 prey) through iteration 100. No visible degradation as `f` increased from 0.5 to 1.0 — if anything, `f=1.0`'s late-training numbers (21/22) were marginally higher than `f=0.5`'s (20/19).

| `move_fraction` | early (&le;30) | mid (31-70) | late (&gt;70) |
|---|---|---|---|
| 0.5 | 50% predator extinction, ep_len 548 | 11% predator extinction, ep_len 905 | 0% extinction, ep_len 1000, ~20 pred / ~19 prey |
| 0.75 | 54% predator extinction, ep_len 520 | 0% extinction, ep_len 1000 | 0% extinction, ep_len 1000, ~20 pred / ~20 prey |
| 1.0 | 42% predator extinction, ep_len 624 | 0% extinction, ep_len 1000 | 0% extinction, ep_len 1000, ~21 pred / ~22 prey |

**Conclusion drawn at the time (turned out to be premature — see §6)**: `move_fraction=1.0` adopted as `config_env.py`'s shipped default, on the reasoning that it was both "sustainable" per this probe and the scientifically cleanest option (no residual always-on tax muddying the noop-vs-move comparison).

## 4. Infrastructure detour: background jobs and session teardown

The first attempt at a longer confirmation run was launched via the coding assistant's own background-task mechanism. When the user closed the VS Code folder partway through (85/100 iterations into a re-run), the job was killed — it turned out background tasks launched that way are tied to the assistant session's own process tree and don't survive the session ending, contrary to an earlier (incorrect) assumption that `setsid`-style detachment would be enough. Fixed by relaunching via `systemd-run --user --unit=...`, which runs as a transient systemd-managed service independent of the assistant's process tree and survives the session/window closing (as long as the desktop session itself stays logged in). All runs from that point on used this approach.

## 5. Second pass: 500-iteration confirmation run (2026-09-16)

Given the infrastructure fix, a full confirmation run at the shipped default (`move_fraction=1.0`, seed 42, no `--move-fraction` flag needed since it's now baked into `config_env.py`) was launched for 500 iterations — long enough to see whether the 100-iteration snapshot actually represented a converged equilibrium, using the existing `PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45` run (same seed, full 1000 iterations, no step energy) as a same-seed control already on disk.

**`base_environment` control, for reference** (already-existing run, no step energy): reaches its long-run range by iteration ~100 (18.0 pred / 24.4 prey) and stays essentially flat all the way to iteration 1000 (18.1 pred / 24.6 prey at 751-1000). 100 iterations really is close to enough *for this baseline*.

**`base_environment_step_energy` at `move_fraction=1.0`, tracked through all 500 iterations**:

| iterations | predators | prey | prey extinction rate | episode length |
|---|---|---|---|---|
| 31-70 | 20.6 | 20.6 | 0% | 1000 |
| 71-100 | 21.6 | 16.5 | 6% | 949 |
| 101-200 | 22.0 | 16.4 | 1% | 997 |
| 201-300 | 23.3 | 16.1 | 7% | 966 |
| 301-400 | 25.5 | 10.8 | 20% | 899 |
| 401-500 | 26.4 | **9.5** | **29%** | 875 |

**What the 100-iteration probe missed**: at iteration 100 the run looked converged (0% extinction, full-length episodes) — indistinguishable in shape from a healthy result. It wasn't. Tracked further, predator population climbed monotonically (20.6&rarr;26.4) while prey collapsed (20.6&rarr;9.5, more than halved from peak) and the prey-extinction rate kept *accelerating* rather than settling (0%&rarr;6%&rarr;1%&rarr;7%&rarr;20%&rarr;29%), with no plateau anywhere in the 500 iterations, unlike `base_environment`'s own trajectory which visibly flattens by iteration ~300. By the end, nearly a third of episodes are ending with prey wiped out.

**Why predator population growth is direct evidence of increased predation, not just correlation**: predators' only energy income in this environment is eating prey (no other food source), and predator reproduction requires crossing an energy threshold — so a rising predator population is only possible if predators are collectively consuming prey energy at a growing net rate. It isn't "predators surviving longer while idle"; it's "predators eating more, successfully, over time." Prey's food source (grass) isn't the bottleneck either: grass is capped at a fixed number of patches that always regrow toward full every step with no possibility of permanent depletion (unlike prey, which can hit population zero and stay there), and if anything a shrinking prey population faces *less* competition for the same regrowing grass supply over time — so the prey decline gets harder, not easier, to explain as a food-scarcity/starvation story as it progresses. That leaves predation as the parsimonious explanation for the whole trend.

## 6. The mechanism, and why the 100-iteration probe was misleading

**Working hypothesis**: making noop free is not a neutral, symmetric change. A predator can catch prey simply by having prey wander onto its cell — so a predator can sit motionless near a good ambush spot and it costs *nothing at all* to wait, indefinitely. Prey have no equivalent free path to calories: grass doesn't move, so eating requires prey to actively move onto a grass cell every time, which under `move_fraction=1.0` is never free. The same rule ("standing still costs nothing") is a much bigger gift to the predator's viable strategy space than to prey's, because ambush is a genuinely free tactic for predators while foraging can never be free for prey.

**A named ecological analogue for this shape of instability**: the *paradox of enrichment* (Rosenzweig, 1971) — loosening a resource/energy constraint on one side of a predator-prey system doesn't always produce a better equilibrium; it can destabilize a previously-stable one into growing oscillations or collapse. Making noop free is effectively a targeted "enrichment" of the predator's strategy space specifically, which fits the observed shape (a slow-starting, accelerating divergence rather than an immediate one) better than a simple parameter-magnitude explanation would.

**This project's own prior finding fits the same story**: `base_environment`'s `retrain_frozen_opponent.py` results (documented in `base_environment/README.md`) already found the predator side of this environment converges faster and more robustly than the prey side under the *original* rules — prey's ecological outcome improved ~80% once the predator stopped co-adapting, meaning ordinary co-training already shortchanges prey somewhat. A rule change that disproportionately helps predators' tactics on top of that pre-existing asymmetry is exactly what you'd predict to widen the gap further, which is what happened.

**The concrete biological gap, raised directly by the user and the actual crux of the issue**: real organisms have a non-zero basal metabolic rate — maintaining a living body costs energy continuously, even at complete rest (breathing, thermoregulation, cellular maintenance). Nothing survives on zero energy expenditure. At `move_fraction=1.0`, a predator that only ever picks noop has *constant, unchanging* energy forever — it cannot starve no matter how long it waits. That's not a subtle imbalance; it's a hole in the model that grants a strategy (infinite-patience ambush) no real predator has access to. This reframes the fix from "an empirical patch for an observed imbalance" to "a correction of a biologically implausible rule" — and it matches the module's own original framing (the per-step tax was meant to be reduced, not eliminated).

## 7. Conclusion and status

**`move_fraction=1.0` does not produce a sustainable ecosystem at training scale.** The 100-iteration probe was not long enough to catch this — it looked converged and wasn't. `config_env.py`'s current default (`move_fraction=1.0`, i.e. `energy_loss_per_step_predator/prey = 0`) is a known-bad recommendation as of this writing and needs to be walked back.

**Next step (not yet done)**: revert the shipped default away from `move_fraction=1.0`, and treat "noop must cost something nonzero" as a hard modeling requirement rather than a tuning option — both for the empirical reason above and the biological one. `move_fraction=0.75` is the next concrete candidate (noop still costs 25% of the total budget for each species), to be run for the full 500 iterations this time, not just 100, given what we now know about how late this particular failure mode shows up.

## 8. Standing lesson for future probes in this module

100 iterations is not sufficient to validate sustainability for this environment when movement/energy economics are being changed — a config can look fully converged (0% extinction, full-length episodes) at 100 iterations and still be in the early phase of a slow, accelerating collapse that only becomes visible past iteration ~300-400. Any future move-fraction (or similar economy-changing) sweep should budget for at least a 500-iteration confirmation on promising candidates before adopting a new default, not just a 100-iteration screen.

## 9. Redesign: independent additive costs, not a fixed-budget split (2026-09-16)

Discussing the §6/§7 findings surfaced a second problem with the `move_fraction` design itself, independent of which split value was chosen: `energy_loss_per_step_predator/prey` (the flat term) was defined as `total × (1-f)` — a *share of a fixed total* — rather than an independent quantity. That's not how metabolism actually works. Real basal metabolic rate doesn't shrink just because an animal also expends energy moving; locomotion cost is *additional* expenditure on top of a fixed baseline, not a reallocation of it. The `move_fraction` split was a bookkeeping choice made to keep "an agent that moves every step pays the same total as `base_environment`" true across the whole sweep, for experimental comparability — not a biological argument, and it actively obscured the fix: with a shared, fixed total, there is no way to give noop a nonzero cost without simultaneously taking something away from the move cost.

**The fix**: replace the single flat/move split with two independent, additive parameters, renamed for clarity:

- `homeostatic_energy_cost_per_step_predator`/`_prey` (was `energy_loss_per_step_predator`/`_prey`) — always charged, every step, regardless of action. Can never be zero by design intent, not just by current config value — this is what rules out the infinite-ambush loophole structurally, rather than by tuning a split fraction away from its extreme.
- `move_energy_cost_per_step_predator`/`_prey` (was `energy_loss_per_move_predator`/`_prey`) — charged *on top* of the homeostatic cost, only when the action isn't noop.

`tune_ppo_base_environment_step_energy.py`'s `--move-fraction` flag is removed; `--homeostatic-cost-predator`/`-prey` and `--move-cost-predator`/`-prey` set the four values directly and independently.

**New defaults** (`config_env.py`), chosen to sit symmetrically around `base_environment`'s original flat tax rather than at either extreme — resting somewhat cheaper than the original tax, moving somewhat more expensive than it:

| | resting (homeostatic only) | moving (homeostatic + move) | `base_environment`'s original flat tax |
|---|---|---|---|
| predator | 0.10 | 0.20 | 0.15 |
| prey | 0.035 | 0.07 | 0.05 |

This has **not yet been run at training scale** — it's a redesign motivated by the §6/§7 failure and the metabolic-rate argument, not itself a validated result. The next step is a 500-iteration run (not 100 — see §8) at these defaults, tracking the same predator/prey population and extinction-rate metrics used throughout this log, to see whether guaranteeing a nonzero resting cost actually prevents the runaway predator-favoring drift found in §5.

## 10. First additive-cost run: predators struggle, then recover, then wobble (2026-09-16/17)

500-iteration run at the §9 defaults (seed 42, `PPO_STEP_ENERGY_ADDITIVE_CONFIRM500_SEED42`, 577.5 min total). This run answered §9's question (does a nonzero resting cost prevent the §5 collapse — yes) but surfaced a different problem, the opposite of §5's.

| iterations | predator extinction | predators | prey | episode length |
|---|---|---|---|---|
| 1-30 | 91% | 0.4 | 42.1 | 222 |
| 31-100 | 41% | ~3.4 | ~48.7 | ~737 |
| 101-200 | 40% | 3.7 | 48.0 | 775 |
| 201-300 | 17% | 7.0 | 39.3 | 908 |
| 301-400 | 7% | 8.0 | 36.1 | 960 |
| 401-430 | 2% | 7.4 | 37.7 | 987 |
| 431-460 | 17% | 6.1 | 41.3 | 918 |
| 461-500 | 23% | 5.2 | 44.2 | 865 |

**No repeat of §5's accelerating one-sided collapse** — this is a fundamentally different, healthier shape: predators recovered from near-total early die-off (91% extinction, population 0.4) down to a low of 2% extinction and a real population (~7-9) by iteration ~400, and prey never went extinct once across all 500 iterations (`extinct_prey = 0.00` throughout). The nonzero homeostatic cost did what it was meant to do: it didn't reproduce the infinite-ambush, one-sided-drift failure mode from §5.

**But two things are unresolved.** First, predator population (~7-9 at its best) sits well below `base_environment`'s own long-run equilibrium (~18-19) — this cost structure still supports a meaningfully smaller, more fragile predator population than the original flat tax did, likely because moving costs 0.20 total (33% above `base_environment`'s 0.15) even though resting is cheaper (0.10, 33% below). Second, the final ~70 iterations show a mild reversal (extinction rate 2%→17%→23%, predators 7.4→6.1→5.2, prey 37.7→41.3→44.2) — could be ordinary oscillation around a real equilibrium (plausible for a predator-prey system) or the early edge of another slow drift; 500 iterations isn't enough to tell which.

## 11. Second probe: easing predator move cost (started 2026-09-17)

To address the "predator population too small relative to `base_environment`" finding in §10, without repeating §9's mistake of changing a shipped default before validating it: `move_energy_cost_per_step_predator` eased from 0.10 to **0.08** via CLI override (`--move-cost-predator 0.08`), leaving homeostatic cost (0.10) and both prey costs (0.035/0.035) unchanged. This drops predators' total moving cost from 0.20 to **0.18** — 20% above `base_environment`'s original 0.15, instead of 33% above. `config_env.py`'s shipped defaults are intentionally left unchanged pending this run's result.

500-iteration run launched, seed 42, `PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42`, 420.2 min total. **Complete, and this is the best result of the whole investigation:**

| iterations | predator extinction | predators | prey | episode length |
|---|---|---|---|---|
| 1-100 | 27% | 8.0 | 33.6 | 784 |
| 101-200 | 0% | 13.6 | 23.9 | 1000 |
| 201-300 | 0% | 14.1 | 23.9 | 1000 |
| 301-400 | 1% | 13.3 | 25.2 | 994 |
| 401-500 | 0% | 12.2 | 28.9 | 1000 |

After the usual early-training instability (27% extinction while the policy is still near-random, the same shape seen in every run including `base_environment` itself), it settled by iteration ~100 into a low-noise equilibrium that **held for the full remaining 400 iterations** — extinction pinned near 0%, episode length at or near the full 1000 throughout, populations (~12-14 predators, ~24-29 prey) in a tight, non-drifting band close to `base_environment`'s own equilibrium (~18-19 / ~19-25). No repeat of run A's late-run wobble.

**Full three-way comparison, same iteration ranges, all now complete:**

| iters | base_environment | | | | run A (homeostatic 0.10/0.035, move 0.10/0.035) | | | | `base_environment_step_energy` (homeostatic 0.10/0.035, move 0.08/0.035) | | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **extinction predator** | **pred/episode** | **prey/episode** | **len** | **extinction predator** | **pred/episode** | **prey/episode** | **len** | **extinction predator** | **pred/episode** | **prey/episode** | **len** |
| 1-100 | 22% | 12.3 | 35.2 | 818 | 58% | 2.3 | 46.7 | 558 | 27% | 8.0 | 33.6 | 784 |
| 101-200 | 0% | 18.6 | 19.5 | 1000 | 40% | 3.7 | 48.0 | 775 | 0% | 13.6 | 23.9 | 1000 |
| 201-300 | 0% | 19.8 | 18.0 | 997 | 17% | 7.0 | 39.3 | 908 | 0% | 14.1 | 23.9 | 1000 |
| 301-400 | 0% | 19.4 | 19.2 | 1000 | 7% | 8.0 | 36.1 | 960 | 1% | 13.3 | 25.2 | 994 |
| 401-500 | 0% | 19.1 | 18.8 | 992 | 15% | 6.1 | 41.6 | 914 | 0% | 12.2 | 28.9 | 1000 |

`base_environment_step_energy` tracks `base_environment` far more closely than run A at every matching range, and does so without run A's late-run reversal.

## 12. Decision: predator move cost 0.08 adopted as the shipped default (2026-09-17)

Easing only the predator's move cost (leaving homeostatic cost and both prey costs untouched) fixed the problem run A surfaced, without reopening the noop-must-cost-something requirement from §6-9 (homeostatic cost is still 0.10/0.035, never zero). `config_env.py` now ships `base_environment_step_energy`'s values as the default: `move_energy_cost_per_step_predator = 0.08` (was 0.10); `homeostatic_energy_cost_per_step_predator/prey` and `move_energy_cost_per_step_prey` unchanged at 0.10 / 0.035 / 0.035.

This closes the loop the module was built to investigate: `base_environment`'s flat, action-independent tax (§1) → a version where noop is completely free (§3, looked fine at 100 iterations, §5 showed it collapses at 500) → independent additive homeostatic+move costs so noop can never be free (§9) → still noticeably below-baseline predator population (§10) → predator move cost eased, now tracking baseline closely and stable for 400+ iterations (§11). Still open: only one seed (42) throughout: replication across 2-3 more seeds, and a run past 500 iterations to see how long the stability in §11 actually holds, are the natural next steps if this module is revisited.

*(Update 2026-09-19: replicated across six seeds in §16 and §19. The default is sustainable in five of six; seed 45 never established a stable predator population -- see §19.)*

## 13. A second, unplanned finding: predator spatial clustering (2026-09-17)

Watching `base_environment_step_energy`'s trained policy in the interactive PyGame viewer (checkpoint_000049, ~iteration 500) next to `base_environment`'s, the user noticed predators appeared to cluster together spatially in `base_environment_step_energy` more than in `base_environment`. This was investigated as a real, separate empirical question rather than dismissed as a visual impression.

**Metric**: the Clark-Evans index -- the ratio of the actual mean nearest-neighbor distance between predators to the distance expected under a purely random (CSR) spatial distribution with the same predator count on the same 25x25 grid. R < 1 = clustered; R = 1 = random; R > 1 = dispersed. Normalizing against the CSR expectation for the actual predator count matters because `base_environment` and `base_environment_step_energy` have different equilibrium population sizes (~18-19 vs ~12-14 -- see §12), and raw nearest-neighbor distance alone would confound "fewer points" with "more spread out."

**First measurement**: one full deterministic (greedy-action) episode each, seed 42, using each config's own final seed-42 checkpoint (`base_environment`: iteration 1000; `base_environment_step_energy`: iteration ~500).

| | Clark-Evans R | mean predators |
|---|---|---|
| base_environment | 1.11 (mildly dispersed) | 16.6 |
| `base_environment_step_energy` | 0.90 (mildly clustered) | 14.3 |

**Working mechanism hypothesis**: charging more for movement than for standing still (`base_environment_step_energy`'s design, per §9-12) is not a neutral change for spatial behavior. A predator can catch prey simply by having prey wander onto its cell, so a predator can sit motionless near a good ambush spot at minimal cost under `base_environment_step_energy`'s economy. `base_environment` has no such asymmetry -- moving costs exactly what resting costs, so there's no energetic reason to prefer standing still over continuous patrol. If multiple predators converge on the same handful of attractive waiting spots under `base_environment_step_energy`'s economy (e.g. near where prey must pass to reach grass), that would produce clustering that continuous, cost-free roaming under `base_environment` would not. This is the same underlying free-vs-costly-movement asymmetry already central to §6's mechanism for the §5 collapse, applied to a different observable (spatial pattern rather than population trajectory).

**Caveat flagged immediately**: one episode, one seed. Could be noise from that one episode's specific trajectory, or specific to these two particular trained policies, rather than a property of the design change itself.

## 14. Zero-training validation: robust for these two policies, not yet a generalization claim (2026-09-17)

Asked to statistically validate §13's finding, two genuinely different questions were distinguished before choosing a test:

1. *Is the R=1.11-vs-0.90 gap a robust, repeatable property of these two specific already-trained policies, or was it noise from one episode?* -- answerable with zero new training, just more evaluation episodes of the checkpoints already on disk.
2. *Would a different training run of the same design reproduce this gap?* -- requires training genuinely new, independently-seeded policies; "seed" here means training seed, not evaluation seed, so there is no way to answer this without more training.

Question 1 was answered first, at zero training cost: 30 evaluation episodes per config (different environment-reset seeds 100-129, deterministic actions, same two existing seed-42 checkpoints), one Clark-Evans R per episode (aggregated over that episode's ~1000 steps, not per-timestep, to avoid pseudoreplication from within-episode autocorrelation).

| | n | mean R | median R | std |
|---|---|---|---|---|
| base_environment | 30 | 1.072 | 1.069 | 0.032 |
| `base_environment_step_energy` | 30 | 0.928 | 0.924 | 0.038 |

The two distributions barely overlap (base_environment's single lowest value, 0.998, just touches `base_environment_step_energy`'s two highest, 1.005 and 1.022; everything else is cleanly separated). **Mann-Whitney U = 898, p ≈ 0.000000.**

**Conclusion**: not a one-episode fluke. The clustering difference is a highly robust, consistent property of these two specific trained policies across 30 independent episodes each. **Explicit scope limit**: this validates the finding only for these two particular policies (both trained at seed 42) -- it says nothing about whether a differently-seeded training run of the same design would show the same pattern. Question 2 remains open.

## 15. Follow-up in progress: does clustering generalize across independently-trained policies? (started 2026-09-17)

**Rationale**: §14 ruled out "one lucky/unlucky episode" but not "this particular training run happened to produce a clustering policy by chance, independent of the design." Seed 42 is one random training trajectory; a design-level claim ("`base_environment_step_energy`'s cost structure causes clustering") requires checking that the pattern reappears across policies trained from different random seeds, which necessarily means training more policies -- there is no training-free way to answer this question.

**Design chosen -- paired, not unpaired, by seed**: the same seed trains both a `base_environment` and a `base_environment_step_energy` policy, so seed is naturally a matched-pairs variable rather than two independent groups. A paired Wilcoxon signed-rank test only needs each pair's within-seed difference to point the same direction, not full separation between all individual values across groups -- a much easier bar to clear than an unpaired test, and one that needs far fewer seeds to reach conventional significance: a paired sign test can reach p < 0.05 with 6 consistent-direction pairs (in the best case), versus an unpaired Mann-Whitney U test needing roughly 4-vs-4 groups at minimum and still requiring full separation even then. 3 matched pairs already exist in principle from prior sections (seed 42's `base_environment` and `base_environment_step_energy` checkpoints are the first pair); 5 more matched seeds (43-47) are being trained now to reach 6 total pairs.

**Training scale -- 300 iterations, not 500**: both configs converge to their stable population/behavioral equilibrium by iteration ~100-300 (§5, §11-12) -- the extra iterations used in the 500-iteration sustainability-confirmation runs elsewhere in this document aren't needed just to obtain a representative, converged policy for a clustering measurement. This roughly halves the training cost (~2 days for 10 runs instead of ~3.5).

**A parallelization attempt was tried and abandoned -- worth recording so it isn't retried the same way**: running the 10 training runs concurrently (4-way, then 6-way) using Ray's `num_gpus_per_learner` fractional GPU request and reduced per-run CPU counts was attempted first, to cut wall-clock time. It failed on two counts: (1) the GPU fraction is only logical accounting inside each job's own separate Ray cluster, not a real CUDA memory cap, so 4 concurrent processes collectively exceeded the 16GB card and one crashed with `torch.OutOfMemoryError`; (2) even the surviving concurrent jobs ran at ~225-270s/iteration, 4-5x slower than a solo full-resource run, likely because reducing `num_cpus_per_env_runner` to 1 (from the usual 3) starved each env-runner rather than just adding total-CPU headroom -- the parallel approach was providing little to no net wall-clock benefit over sequential execution, possibly a net loss. Abandoned in favor of a plain sequential queue at full per-run resources (the settings that reliably gave ~45-70s/iteration in every other run this session).

**Plan once training completes**: run the same zero-training evaluation methodology from §14 (30 episodes per new policy, one Clark-Evans R per episode) on each of the 10 new policies, aggregate one mean R per seed per config, then run the paired Wilcoxon signed-rank test across all 6 seed-pairs (42 existing + 43-47 new).

*(Correction added 2026-09-19, see §16: the "pairing is more powerful" argument above turned out to be wrong for this data -- the seed-paired base and step_energy values are uncorrelated, and at n=6 an unpaired test actually has a finer p-value floor. Both tests agree; §16 reports both.)*

**Status**: training completed 2026-09-19 16:05 CEST (all 10 runs, sequential, no failures); evaluation and result in §16.

## 16. Result: predator clustering in base_environment_step_energy replicates across six independently-trained seeds (2026-09-19)

**Setup**: six seeds (42-47), each with one `base_environment` and one `base_environment_step_energy` (default settings) policy. Seed 42 uses the pre-existing runs; seeds 43-47 were trained for this test (300 iterations each, sequentially, full resources). To keep training length identical, **all 12 policies are evaluated at their iteration-300 checkpoint** (`checkpoint_000029`; for seed 42 this is an intermediate checkpoint of its longer run). Each policy: 30 deterministic evaluation episodes (environment reset seeds 100-129), one Clark-Evans R per episode (§13), and the mean over episodes as that policy's value. Script: [`evaluate_clustering.py`](./evaluate_clustering.py); raw per-episode values: [`clustering_results.json`](./clustering_results.json).

| seed | base_environment R | `base_environment_step_energy` R | difference | within-seed Mann-Whitney p (30 vs 30 episodes) |
|---|:---:|:---:|:---:|:---:|
| 42 | 1.049 | 0.921 | 0.127 | 1.5e-11 |
| 43 | 1.009 | 0.883 | 0.126 | 4.1e-10 |
| 44 | 0.996 | 0.974 | 0.022 | 0.023 |
| 45 | 1.029 | 0.883 | 0.146 | 8.1e-11 |
| 46 | 1.026 | 0.998 | 0.028 | 0.018 |
| 47 | 1.030 | 0.901 | 0.128 | 1.3e-10 |
| **mean** | **1.023** | **0.927** | **0.096** | |

**Primary test (pre-specified in §15): paired Wilcoxon signed-rank on the six seed-pair differences, H1: base R > `base_environment_step_energy` R.** W = 21 (the maximum possible), exact **one-sided p = 0.0156** (two-sided p = 0.031). All 6/6 pairs point the same direction (sign test gives the same p). Paired t-test, one-sided p = 0.004; 95% CI on the mean paired difference [0.038, 0.155].

**Supplementary, and arguably the more appropriate test: unpaired Mann-Whitney on the six vs six seed means, one-sided p = 0.0022.** Why this is reported alongside the pre-specified paired test: the pairing by seed number carries no signal (Spearman correlation between base R and `base_environment_step_energy` R across seeds = -0.09, p = 0.87 -- the same seed in two different environments does not couple their randomness), so the paired design gives no variance-reduction benefit, and at n = 6 the signed-rank test's smallest attainable p (0.0156) is coarser than the unpaired test's. The pre-specified result stays primary; the two agree.

**Episode-level picture**: pooled over seeds, `base_environment` R = 1.023 (sd 0.036) with 26% of episodes below 1; `base_environment_step_energy` R = 0.927 (sd 0.070) with 84% of episodes below 1. `base_environment_step_energy` is below 1 (clustered) in all 6 seeds; `base_environment` is below 1 in only 1 of 6 (seed 44, at 0.996 -- effectively 1).

**Sensitivity**: evaluating seed 42 at its final checkpoints instead (base iteration 1000, `base_environment_step_energy` iteration 500) gives R = 1.072 vs 0.928 and leaves the paired result unchanged (p = 0.0156).

*(Two statements in this section were corrected by the density analysis in §17: R ≈ 1 is not "random" on this bounded grid -- random placement gives R ≈ 1.13 -- so `base_environment` is also mildly clustered, just less than `base_environment_step_energy`; and the density confound described below biases in the opposite direction from what is stated here.)*

### What this establishes, and what it doesn't

- **Established**: `base_environment_step_energy`'s predators are spatially clustered relative to random placement, and `base_environment`'s are not, consistently across six independently-trained seeds -- the §13 observation is not an artifact of the seed-42 pair. A correction to §13: the seed-42 single episode gave R = 1.11 ("mildly dispersed") for `base_environment`; across seeds it is **≈ random (mean 1.02)**, not reliably dispersed. The robust contrast is "`base_environment_step_energy` clusters; base_environment is about random."
- **The effect size is heterogeneous.** Four seeds show a large gap (~0.13-0.15); seeds 44 and 46 show a small one (~0.02-0.03), and seed 46's `base_environment_step_energy` value (0.998) is practically random. All six are in the predicted direction, but "`base_environment_step_energy` is clustered" is far clearer in some seeds than others; the mean effect (0.096) is not what a typical single seed looks like.
- **p = 0.0156 is the floor for a paired signed-rank test with six pairs** -- the design cannot produce a smaller paired p, and six seeds is a small sample. This is significant at the conventional 0.05 level but is not overwhelming evidence.
- **The mechanism is still a hypothesis, not tested here.** §13's explanation (moving costs more than resting, so predators favor waiting near shared ambush spots) predicts the result, but these data don't test it. There is also a plausible confound this design cannot separate: `base_environment_step_energy` and `base_environment` differ not only in cost structure but in equilibrium predator population (~12 vs ~19). Clark-Evans normalizes the expected distance for the actual predator count, but agents cannot share cells, and that exclusion pushes toward dispersion more at higher density -- which would bias `base_environment` toward *higher* R and could account for part of the gap independent of movement costs.
- **Natural next test, not yet run**: a dose-response check across move-cost settings (run A at gap 0.10, the default at 0.08, the earlier free-resting run at 0.15, `base_environment` at 0) using existing checkpoints, which would test whether clustering scales with the move-minus-rest cost gap as the mechanism predicts, and could be paired with a density-matched comparison to address the confound above. (The `move_fraction=1.0` policy comes from a collapsing ecosystem, so it is a less clean point than the others.)

## 17. Density and dose-response checks: the clustering difference is robust, its cause is not (2026-09-19)

Two checks on §16, run on existing checkpoints only (no new training): the same 30-episode evaluation, now also recording mean predator count per episode, for all 12 policies, plus two extra seed-42 policies: run A (rest 0.10, move +0.10) and an earlier free-resting run (rest cost 0, move cost 0.15/0.05; a design since removed from the code, see §2-§9), evaluated with the equivalent additive-cost keys. Scripts: [`evaluate_clustering_density.py`](./evaluate_clustering_density.py), [`analyze_clustering_density.py`](./analyze_clustering_density.py); raw per-episode data in [`clustering_density_data/`](./clustering_density_data/). The R values reproduce §16's exactly (the evaluation is deterministic).

**A correction to §16's reference point.** The Clark-Evans "R = 1 is random" reading assumes an unbounded plane. On this bounded 25x25 grid, with no two agents sharing a cell, N randomly placed predators give a *higher* R because of edge effects, and higher still at low N (simulated, 3,000 placements per N):

| N predators | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 24 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| expected R under random placement | 1.24 | 1.20 | 1.18 | 1.16 | 1.15 | 1.14 | 1.13 | 1.13 | 1.13 | 1.13 |

Measured against this null (excess = observed R minus random R at that policy's predator count), **both configurations are clustered**: `base_environment` −0.110, `base_environment_step_energy` −0.235. So "`base_environment` is about random" (§16) was wrong; it is mildly clustered, and `base_environment_step_energy` considerably more.

**The density confound raised in §16 runs the other way.** §16 worried that the smaller `base_environment_step_energy` population might be misread because agents cannot share cells. The null model shows the geometric bias at low density *raises* R (the random baseline is higher at `base_environment_step_energy`'s lower N), which works against finding clustering in `base_environment_step_energy`. After adjustment the difference is unchanged in direction and slightly stronger: paired Wilcoxon one-sided p = 0.0156 (6/6 pairs; the floor for six pairs), unpaired Mann-Whitney one-sided p = 0.0011.

**But density is not separable from configuration here, and one pattern points the other way.**
- The configurations have **no overlap in predator count** (`base_environment` 18.2-18.5, `base_environment_step_energy` 8.6-14.4 per policy), so no density-matched comparison across them exists in this data.
- **Within `base_environment_step_energy`, clustering weakens as predator count rises**: dR/dN = +0.015 per predator (episode level, policy-demeaned, p = 3.6e-5, r = 0.30); within `base_environment` there is no significant slope. Extrapolating that slope from `base_environment_step_energy`'s ~12 predators to `base_environment`'s ~18 would raise R by about 0.09, roughly the size of the gap. That extrapolation is outside the observed range and the relation could run either way (clustered predators may compete and end up fewer), so it is a warning, not a result.
- Across all 14 policies R and predator count correlate at Spearman 0.73, confounded with configuration.

**The dose-response does not show what the move-cost mechanism predicts** (seed 42, one policy per level; "gap" = extra cost of moving over resting):

| gap | policy | R | mean predators | excess vs random |
|---|---|---|---|---|
| 0 | base_environment | 1.049 | 18.2 | −0.085 |
| 0.08 | `base_environment_step_energy` | 0.921 | 14.4 | −0.229 |
| 0.10 | run A | 0.983 | 10.0 | −0.195 |
| 0.15 | earlier free-resting run (rest cost 0; removed design) | 1.039 | 24.3 | −0.087 |

It is non-monotonic, and it is one policy per level while `base_environment_step_energy`'s own seed-to-seed range is 0.882-0.998 -- so run A's 0.983 lies inside `base_environment_step_energy`'s spread and 0.08 vs 0.10 cannot be told apart. The informative point is the last row: with resting free, waiting near an ambush spot is costless, the mechanism's most extreme case, yet those predators are no more clustered than `base_environment` (excess −0.087). That is a different regime (prey collapsing, predators abundant at ~24, so density again differs), so it is not a clean refutation, but it is evidence against a simple "cheaper waiting -> more clustering" story, not for it.

**Where this leaves the clustering finding**
- **Established:** `base_environment_step_energy` predators are more clustered than `base_environment` predators, consistently across six seeds, after correcting for the geometry of the bounded grid.
- **Not established:** that the move/rest cost gap causes it. The ambush explanation of §13 is untested by these data and partly disfavored by the dose-response point above; a predator-density explanation cannot be excluded because the configurations do not overlap in density.
- **What would separate them:** conditions that vary the cost gap while holding predator count fixed (or vice versa) -- for example the default costs with a larger initial or capped predator population, or several additional move-cost levels across seeds, which requires new training (~1.5-2 days, §15's estimate).

### Possible next steps (not planned; see §18 for what has since been checked)

Item 1 of the original note (test the ambush story via noop/movement rates) has been done in §18: movement differs only slightly. The remaining ideas, in order of cost:

1. **Zero-training causal test of the birth mechanism (see §18).** The policies only see local observations, so the same trained policies can be evaluated with a different spawn rule: place each newborn on a *random* free cell instead of an adjacent one, and compare the base-vs-`base_environment_step_energy` clustering gap under both rules. If the clustering excess in both configurations shrinks under random spawning, adjacent births are causal for clustering; if the gap between the configurations also shrinks, births explain the difference between them; if the gap persists, something else does. Caveat: the policies were trained with adjacent spawning, so this is a mild distribution shift.
2. **Separate density from configuration by matching predator count at evaluation time.** Run both configurations' policies at the same fixed number of predators (for example 12 and 18): set the initial predator count and disable predator births and deaths (or measure over an early window after a burn-in), and compare R at matched density.
3. **Replicated dose-response with new training** (several move-cost levels at a fixed rest cost, at least 3-5 seeds each; ~4-5 hours per 300-iteration run, i.e. days), only if the cheaper checks leave the question open.

## 18. Birth-and-dispersal mechanism: supported in direction, modest in size (2026-09-19)

**The hypothesis (from the user, replacing §13's ambush explanation as the leading candidate).** (1) Offspring are born on a cell adjacent to the parent (`_find_available_spawn_position` takes the first free of the four neighbouring cells; the code is identical in both environments), so every birth creates an adjacent pair. (2) When moving away is costly, those pairs and groups disperse more slowly, so clusters persist longer. Unlike the ambush story, this does not require predators to choose good waiting spots. It predicts: `base_environment_step_energy` predators move less; their offspring drift away from the parent more slowly; and adjacent pairs are over-represented.

**Method.** The same 14 policies and 30 deterministic episodes as §17 (12 base/`base_environment_step_energy` policies at iteration 300, plus run A and the earlier free-resting run at seed 42), instrumented (no training, CPU-only): per predator, the share of actions that are noop and the share of steps in which it changes cell; per-predator birth rate; mean parent-offspring distance at offspring ages of 1 to 100 steps (the spawn call is wrapped to capture each newborn's parent); and the share of predators with another predator within 1.5 cells. Scripts: [`evaluate_clustering_mechanism.py`](./evaluate_clustering_mechanism.py), [`analyze_clustering_mechanism.py`](./analyze_clustering_mechanism.py); raw data: [`clustering_mechanism_data/`](./clustering_mechanism_data/). The instrumented R values reproduce §16-17 exactly.

| measure | base_environment | `base_environment_step_energy` | seeds in the predicted direction | p (unpaired / paired, one-sided) |
|---|:---:|:---:|:---:|:---:|
| share of predator actions that are noop | 0.139 | 0.163 | 4/6 | 0.12 / 0.22 |
| share of predator-steps in which the predator moves | 0.793 | 0.765 | 5/6 | 0.12 / 0.078 |
| mean displacement per step (cells) | 0.964 | 0.931 | 4/6 | 0.24 / 0.16 |
| parent-offspring distance at offspring age 10 (cells) | 4.94 | 4.68 | 5/6 | 0.021 / 0.031 |
| ... at age 30 | 7.63 | 7.26 | 5/6 | 0.033 / 0.047 |
| ... at age 50 | 9.29 | 8.59 | 5/6 | 0.033 / 0.031 |
| births per 1000 predator-steps | 7.31 | 8.91 | 6/6 | 0.0011 / 0.016 |
| adjacent-neighbour share, minus random expectation at the same predator count | +0.090 | +0.132 | 6/6 | 0.0043 / 0.016 |

Mean parent-offspring distance by offspring age (ages 1, 2, 3, 5, 10, 20, 30, 50, 75, 100; mean over seeds): base 1.72, 2.29, 2.77, 3.57, 4.94, 6.53, 7.63, 9.29, 10.60, 11.27; `base_environment_step_energy` 1.76, 2.27, 2.72, 3.44, 4.68, 6.16, 7.26, 8.59, 9.66, 10.26. Newborns start equally close in both; `base_environment_step_energy`'s offspring fall behind gradually.

**A measurement pitfall caught along the way.** The *raw* adjacent-neighbour share is not higher in `base_environment_step_energy` (0.264 vs 0.285, 2/6 seeds), which at first looked like evidence against the mechanism. It is density-dependent: `base_environment_step_energy` has fewer predators, so fewer are adjacent by chance. Against random placement at each episode's own predator count (2,000 random draws per N), adjacent pairs are 2.0x the random expectation in `base_environment_step_energy` (0.264 vs 0.132) and 1.46x in base (0.285 vs 0.196); the excess in the table above uses this null.

**Reading**
- **Point 1 (births create adjacent pairs) is supported.** Adjacent pairs exceed random expectation in both configurations (base is also clustered, consistent with births alone), more so in `base_environment_step_energy`, and `base_environment_step_energy` predators reproduce more per capita (6/6 seeds, p = 0.001). Across the 12 policies, per-predator birth rate correlates with clustering R at Spearman -0.76 (p = 0.004), though this is confounded with configuration and within `base_environment_step_energy` alone (6 policies) it is not significant (-0.37, p = 0.47).
- **Point 2 (costly moving keeps clusters together) is supported only modestly.** Offspring end up 5-8% closer to the parent at ages 10-50 in `base_environment_step_energy` (5/6 seeds), and dispersal distance tracks clustering across the 12 policies (Spearman +0.62 at age 30, +0.75 at age 50; +0.83 within `base_environment_step_energy`, p = 0.042, n = 6). But overall movement barely differs: `base_environment_step_energy` predators move in 77% of steps versus 79%, not significant. This is not predators sitting still; the effect is a small slowing of dispersal that accumulates over the offspring's lifetime. Noop share and movement do not track R across the policies.
- **The two extra seed-42 policies fit "births and slow dispersal are both needed" (post hoc, one policy each).** The earlier free-resting run moves least (noop 0.22, displacement 0.85) but has the lowest birth rate (5.4 per 1000) and no extra clustering (R = 1.039); run A has the highest birth rate (10.5) but base-like movement (noop 0.11, displacement 0.99) and only moderate clustering (R = 0.983).

**Caveats**
- **Correlation, not causation.** Every measure differs between the two configurations together, so these data cannot say which one drives clustering. In particular the higher per-predator birth rate may itself be a consequence of the configuration (`base_environment_step_energy` has more prey per predator), i.e. downstream of the cost structure rather than independent of it. Density remains entangled with configuration (§17).
- **Multiple comparisons.** About a dozen base-vs-`base_environment_step_energy` comparisons were made. Only the birth-rate result (p = 0.0011) clearly survives a strict correction (0.05/12 = 0.004); the adjacency excess (p = 0.0043) is borderline; the dispersal results (nominal p = 0.02-0.05) do not survive one and should be read as suggestive.
- **Adjacency and R are related by construction.** Excess adjacency correlates with R at Spearman -0.92 across the policies, but adjacent pairs feed directly into R's nearest-neighbour distances, so this is largely definitional and is not independent evidence.
- **Small sample and post hoc adjustments.** Six seeds; the density adjustment of the adjacency measure was made after seeing the raw result (for a stated, mechanical reason above).

**Where this leaves it.** The leading explanation for `base_environment_step_energy`'s extra clustering is now births plus slower dispersal, not ambush waiting: the measurements point that way in direction, with modest effect sizes and no causal test yet. The cheapest causal test is listed first in the "Possible next steps" note at the end of §17: evaluate the same policies under a random-cell spawn rule and see how the clustering gap changes.

## 19. Populations across six seeds: fewer predators, more prey, and one seed that did not establish (2026-09-19)

§11-12 judged the default sustainable on one seed (42). The ten runs trained for §15 allow the same look across six seeds. The two configurations and their cost differences are defined in the table under "Naming" at the top of this file (in short: the default module has the same energy economy as `base_environment` except that moving costs extra). Numbers below are training-time ecology metrics for iterations 201-300 (averaged over the iterations that logged a completed episode), the same window for every seed; script: [`analyze_populations.py`](./analyze_populations.py), which reads the runs' Ray result logs.

| seed | base_environment | | | | base_environment_step_energy | | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **extinction predator** | **pred/episode** | **prey/episode** | **len** | **extinction predator** | **pred/episode** | **prey/episode** | **len** |
| 42 | 0% | 19.8 | 18.0 | 997 | 0% | 14.1 | 23.9 | 1000 |
| 43 | 0% | 19.3 | 19.2 | 981 | 4% | 11.7 | 30.1 | 970 |
| 44 | 0% | 19.8 | 18.0 | 984 | 0% | 11.9 | 28.3 | 1000 |
| 45 | 0% | 19.1 | 20.8 | 988 | **36%** | **3.8** | **49.1** | **815** |
| 46 | 0% | 19.0 | 20.7 | 1000 | 2% | 10.0 | 33.9 | 996 |
| 47 | 0% | 19.2 | 19.6 | 996 | 0% | 12.0 | 28.4 | 1000 |
| **mean** | 0% | 19.4 | 19.4 | 991 | 7% | 10.6 | 32.3 | 964 |

**Fewer predators and more prey, in every seed.** `base_environment_step_energy` has fewer predators than `base_environment` in 6/6 seeds (19.0-19.8 vs 3.8-14.1; unpaired one-sided p = 0.0011, the floor for six seeds; paired p = 0.016) and more prey in 6/6 (18.0-20.8 vs 23.9-49.1; same p-values). In the evaluation episodes of §16-18 (iteration-300 checkpoints) the mean predator count is 18.3 (18.2-18.5) vs 12.5 (8.6-14.4), again 6/6 and p = 0.0011. Prey are more numerous per predator even though predators are fewer, so predator numbers do not look food-limited; the higher energy cost of active hunting (0.18 per moving step against `base_environment`'s flat 0.15) is a consistent explanation but is not tested here.

**`base_environment` is uniformly robust; the default is not.** `base_environment` shows 0% predator extinction and ~19 predators in all six seeds. `base_environment_step_energy` shows 0-4% extinction in five seeds, but **seed 45 never established a stable predator population**: 47%, 48%, 34% and 38% predator extinction over the four training windows (1-100, 101-200, 201-250, 251-300) with only ~3-4 predators throughout. Every other seed recovered by iteration 100-200 (extinction 0-7% afterwards), so this is not a slow recovery that more of the same training would obviously fix, although 300 iterations is all that was run and a longer run might behave differently.

**What this changes.**
- The sustainability claim in §11-12 and the README, made on seed 42 alone, holds in 5 of 6 seeds and fails in 1. "Sustainable" should be read as "sustainable in most seeds", with a ~1-in-6 chance (from six seeds) of a failed predator population, until more seeds or a tuning change say otherwise. `base_environment` showed no such failures in the same six seeds.
- **The clustering result (§16) does not depend on seed 45**, even though it is the seed with the lowest predator count (8.6 in evaluation) and among the strongest clustering: without it, the five remaining pairs are all in the predicted direction (mean R 1.022 vs 0.935, difference 0.086 against 0.096 with all six; paired one-sided p = 0.031, the floor for five pairs; unpaired p = 0.008).
- Seed 45 also fits the pattern in §17 that clustering is strongest where predator numbers are lowest, which again cannot separate configuration from density (§17).

**Not investigated.** Why seed 45 failed (early-training luck in a policy that then never reaches a viable hunting strategy is the obvious candidate, but no analysis was done), and whether a slightly easier predator cost, more iterations, or more seeds change the failure rate.
