# base_environment_step_energy: investigation log and results

This documents the reasoning and results behind `base_environment_step_energy`, in the order they actually happened, including the mistakes and corrections along the way. See [`README.md`](./README.md) for the module's structure and how to run it; this file is the narrative behind the numbers.

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

| iters | base_environment | | | | run A (homeostatic 0.10/0.035, move 0.10/0.035) | | | | run B (homeostatic 0.10/0.035, move 0.08/0.035) | | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **extinction predator** | **pred/episode** | **prey/episode** | **len** | **extinction predator** | **pred/episode** | **prey/episode** | **len** | **extinction predator** | **pred/episode** | **prey/episode** | **len** |
| 1-100 | 22% | 12.3 | 35.2 | 818 | 58% | 2.3 | 46.7 | 558 | 27% | 8.0 | 33.6 | 784 |
| 101-200 | 0% | 18.6 | 19.5 | 1000 | 40% | 3.7 | 48.0 | 775 | 0% | 13.6 | 23.9 | 1000 |
| 201-300 | 0% | 19.8 | 18.0 | 997 | 17% | 7.0 | 39.3 | 908 | 0% | 14.1 | 23.9 | 1000 |
| 301-400 | 0% | 19.4 | 19.2 | 1000 | 7% | 8.0 | 36.1 | 960 | 1% | 13.3 | 25.2 | 994 |
| 401-500 | 0% | 19.1 | 18.8 | 992 | 15% | 6.1 | 41.6 | 914 | 0% | 12.2 | 28.9 | 1000 |

Run B tracks `base_environment` far more closely than run A at every matching range, and does so without run A's late-run reversal.

## 12. Decision: run B adopted as the shipped default (2026-09-17)

Easing only the predator's move cost (leaving homeostatic cost and both prey costs untouched) fixed the problem run A surfaced, without reopening the noop-must-cost-something requirement from §6-9 (homeostatic cost is still 0.10/0.035, never zero). `config_env.py` now ships run B's values as the default: `move_energy_cost_per_step_predator = 0.08` (was 0.10); `homeostatic_energy_cost_per_step_predator/prey` and `move_energy_cost_per_step_prey` unchanged at 0.10 / 0.035 / 0.035.

This closes the loop the module was built to investigate: `base_environment`'s flat, action-independent tax (§1) → a version where noop is completely free (§3, looked fine at 100 iterations, §5 showed it collapses at 500) → independent additive homeostatic+move costs so noop can never be free (§9) → still noticeably below-baseline predator population (§10) → predator move cost eased, now tracking baseline closely and stable for 400+ iterations (§11). Still open: only one seed (42) throughout: replication across 2-3 more seeds, and a run past 500 iterations to see how long the stability in §11 actually holds, are the natural next steps if this module is revisited.
