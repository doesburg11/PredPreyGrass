# Reward shaping: sparse vs. dense rewards

This folder holds one connected line of investigation: starting from a
single question ("is the base environment's sparse reward hurting
training?"), it grew into five trained sibling environments, an unplanned
discovery of two real RLlib-compliance bugs, and a headline result that
reverses the question it started with — **reward shaping should be
minimized here, not maximized.** This README is the full story: motivation,
methodology, the bug discovery, every module's result, the mechanistic
explanation, and what's still open.

Each module below has its own README with implementation-level detail
(exact reward mechanics, config). This file is the overview and the
results log — read this first.

## 1. Motivating hypothesis

[`base_environment`](../base_environment)'s only nonzero reward anywhere is
a flat `+10` bonus at the instant of successful reproduction — every other
reward hook (`reward_predator_catch_prey`, `reward_prey_eat_grass`,
`reward_predator_step`, `reward_prey_step`, `penalty_prey_caught`) is `0.0`.
That means every agent gets zero training signal for the ~50-200+ steps
between reproduction events.

This investigation started as the chosen next step after the Darwin/Baldwin
evolutionary project's Trial 7 came back null: rather than continuing to
retune/reseed the same shape of experiment, the plan was to first fix
reward sparsity as groundwork for a future, more progressive attempt at
combining Nature (heritable genome) and Nurture (lifetime RL learning). The
hypothesis: this sparsity was hurting training — slow, chaotic early
learning — documented across every module's `RESULTS.md` repo-wide.
Replacing it with a **dense, biologically literal reward** (each agent's
reward equal to its own net energy delta every single step — decay,
movement, eating, reproduction cost, no hand-designed shaping constants)
was expected to improve outcomes.

**This hypothesis is what the final result below falsifies.**

## 2. An unplanned discovery: two RLlib-compliance bugs

While implementing the first dense-reward variant, two real bugs surfaced
in `base_environment`'s output-assembly logic — present by code lineage in
most environments across this repo, not something specific to reward
design:

1. **Termination-reporting timing.** The environment's output filter used
   `self.agents` *after* a dying agent had already been removed from it, so
   that agent's `terminated=True`, final reward, and final observation were
   silently dropped before ever reaching RLlib — only
   `terminations["__all__"]` (full population collapse) was ever visible.
   Fix: defer removal from `self.agents` to the start of the *next* step,
   so a terminating agent stays listed through the step it dies in
   (matches what RLlib's `MultiAgentEpisode`/env-checker requires). The
   truncation branch (episode hits `max_steps`) had the same bug in a
   different form — it returned entries for every `possible_agent`
   (including ones never born that episode) instead of just `self.agents`.

2. **Agent-ID reuse within an episode.** Newborns recycled freed ID slots
   (`predator_0`..`49`, `prey_0`..`49`) — this collides with RLlib's
   per-episode agent-identity model, where one agent-ID string maps to
   exactly one continuous trajectory. Once bug #1 is fixed and terminations
   are correctly reported, RLlib hard-errors (`MultiAgentEnvError`) the
   instant a reused ID produces more data after being marked done.
   **Without fixing bug #1 first, bug #2 doesn't crash — it silently
   stitches two unrelated individuals' trajectories into one fabricated
   continuous episode object instead**, which is how `base_environment` has
   been running this whole time without ever erroring. Measured reuse rate
   under default config: **~75% of all births reuse a retired ID within the
   same episode** — not a rare edge case, the normal case. Fix: a
   monotonically increasing, never-reused per-species newborn-ID counter,
   reset only at `reset()`. This requires the ID pool
   (`n_possible_predators`/`n_possible_prey`) to be sized well above
   expected *cumulative* births per episode, not just concurrent
   population — bumped `50 → 2000` in every module below (cheap: RLlib only
   uses this list to build a per-episode space dict once per reset).

**Scope caveat, important**: both fixes are applied only in the five
modules in this folder. `base_environment` itself was left **untouched**,
kept as the historical original — not a comparison partner, not
retroactively fixed. Every `eco_evolutionary_*` module and everything else
in `predpreygrass/non_evolutionary/` (besides [`kick_back_rewards`](../kick_back_rewards)
and [`lineage_rewards`](../lineage_rewards), which were checked and already
had independent, correct fixes for the same bug class — see
[`lineage_rewards/PROPER_RLLIB_TERMINATION.md`](../lineage_rewards/PROPER_RLLIB_TERMINATION.md))
**still has both bugs, unverified and unfixed**. This is a real, not-yet-
acted-on implication for the Darwin/Baldwin evolutionary trial history — see
section 7.

A third, minor bug was also found and fixed in every module's training
script: the diagnostic `EpisodeReturn.on_episode_end` callback's
`episode.get_rewards()` call was observed to raise an `IndexError` on some
episodes under this env's dynamic population (agents born/dying mid-
episode), an apparent RLlib edge case in env-step↔agent-step index
translation for episodes chunked across `sample()` calls. This is
console-logging only — RLlib's own `env_runners/episode_return_mean`
metrics come from a separate internal path — so it was hardened with a
try/except rather than investigated further.

## 3. Methodology

Every module below was trained a full **1000 PPO iterations** under
identical hyperparameters and resource configuration, so any difference in
outcome is attributable to reward design, not infrastructure:

- `gamma=0.99`, `lr=0.0003`, `train_batch_size_per_learner=1024`,
  `minibatch_size=128`, `num_epochs=30`, `entropy_coeff=0.0`,
  `clip_param=0.3`, `kl_coeff=0.2`, `kl_target=0.01`
- `num_gpus_per_learner=1`, `num_learners=1`, `num_env_runners=20`
- Same conv/FC architecture for both predator and prey policies
  (`[16,32,64]` conv filters, `[256,256]` FC)

**Sequential, not concurrent, training.** Two runs were briefly trained
concurrently to save wall-clock time; this pushed combined GPU memory to
~92% early, before this environment's known memory-growth-with-episode-
length pattern even kicked in — a real OOM risk, and Ray does not
coordinate GPU memory across independent clusters (it's shared via normal
OS/CUDA time-slicing like any two unrelated processes). Since every run
uses identical batch size and hyperparameters, "iteration N of run A" vs.
"iteration N of run B" is a valid comparison regardless of wall-clock
simultaneity — concurrency was pure convenience, not a validity
requirement. Every run after the first two was trained solo,
full-resource, sequentially.

**Reproduction counts, not raw reward, as the comparison metric.** Raw
`episode_return_mean` is not comparable across these modules — the reward
*scales* are fundamentally different (discrete `+10` spikes vs. continuous
per-step deltas of very different magnitude). Every comparison below uses
real, reward-scheme-independent behavioral outcomes instead: births per
species (predator/prey reproduction counts) and final population size,
measured by running the final trained checkpoint through several seeded
episodes with deterministic actions.

## 4. Results table

| module | reward design | predator births (avg, 3 seeds) | prey births (avg) | % of sparse (pred / prey) | wall time |
|---|---|---|---|---|---|
| [`base_environment_sparse_rewards`](base_environment_sparse_rewards) | sparse, `+10` on reproduction only | **135.3** | **588.7** | 100% / 100% | **12.45h** |
| [`base_environment_sparse_rewards_plus_eating`](base_environment_sparse_rewards_plus_eating) | sparse + asymmetric eating bonus (`+1` predator / `+0.1` prey) | 111.3 | 552.0 | 82% / 94% | 11h22min |
| [`base_environment_dense_rewards_additive`](base_environment_dense_rewards_additive) | dense per-step energy delta **+** `+10` reproduction bonus | 85.0 | 445.0 | 63% / 76% | 16.22h |
| [`base_environment_dense_rewards`](base_environment_dense_rewards) | dense per-step energy delta only, no reproduction bonus | 56.7 | 311.0 | 42% / 53% | 18.68h |
| [`base_environment_sparse_rewards_plus_kickback`](base_environment_sparse_rewards_plus_kickback) | sparse + `+10` grandparent kick-back on grandchild birth | *training* | *training* | *training* | *training* |

Sparse wins on every axis measured against pure dense: highest reproduction
rate for both species, most balanced final predator:prey ratio, zero
extinction events across all tested seeds (pure dense had one predator
population go fully extinct even at its final, fully-trained checkpoint),
and the fastest wall-clock time despite supporting the largest population
(an anomaly not fully explained — see section 6).

## 5. Module by module

### `base_environment_sparse_rewards` — the baseline

Byte-for-byte the same sparse, reproduction-only reward as
`base_environment`, with only the two RLlib-compliance fixes applied.
Exists to be a fair, bug-fixed comparison partner for every other module
here. **Result**: 135.3 / 588.7 births, zero extinctions across tested
seeds, first reached ~1000-step episodes at iteration 20.

### `base_environment_dense_rewards` — pure dense replacement

Reward is pure per-step net energy delta:
`reward = energy_after - energy_before` (folds in decay, movement,
eating, reproduction cost). No reproduction bonus at all — the direct test
of the original hypothesis. **Result**: worst of all five — 56.7 / 311.0
births (42%/53% of sparse), one predator-extinction event observed in 3
tested seeds even at the final, fully-trained checkpoint. Also the slowest
wall-clock despite the smallest population.

### `base_environment_dense_rewards_additive` — dense + reproduction bonus

The dense-pure result raised an obvious question: is pure dense losing
because of *density*, or simply because it drops the reproduction
incentive entirely (reproduction here isn't an action an agent takes —
it fires automatically on crossing an energy threshold — so a pure energy
signal doesn't distinguish "accumulate energy" from "accumulate energy in
order to reproduce")? This module layers the sparse variant's `+10`
reproduction bonus on top of the dense per-step delta (additive, not
replacement). **Result**: recovers most but not all of the gap — 85.0 /
445.0 births (63%/76% of sparse), still short of the pure sparse baseline
and still slower wall-clock than sparse despite fewer agents.

### `base_environment_sparse_rewards_plus_eating` — isolating the noise vs. incentive question

The additive result sharpened the question further: was reward *density*
itself the problem, or specifically the continuous per-step signal's noise
sitting in the *same reward channel* as the reproduction event? (Concretely
demonstrated: the additive variant's reproduction-step rewards were
observed scattered across ~9.3–12.7 rather than sparse's exact, invariant
`10.0`, because the dense delta rides along underneath the flat bonus.)

This module tests that directly: the same clean, event-based sparse-reward
style as the baseline (zero continuous signal, zero decay/movement terms in
the reward at all), with one more *discrete* event type rewarded — eating —
alongside reproduction. The eating reward is deliberately **asymmetric**,
`+1` predator / `+0.1` prey, not a flat `+1`/`+1`: measured directly by
running the sparse baseline's final trained checkpoint (3 seeds, full
1000-step episodes, counting real eating events), predators catch prey
~4.4 times per reproduction on average, but prey eat grass ~60.5 times per
reproduction (grass regrows slowly — `energy_gain_per_step_grass=0.04`,
capped at `initial_energy_grass=2.0` — and gives little per visit, so prey
need many more, smaller meals to reach their reproduction threshold). A
flat `+1` for both would make prey's *total* eating reward per reproduction
cycle (`60.5 × 1 = 60.5`) six times larger than the reproduction reward
itself (`10.0`), swamping the primary incentive the same way the dense
signal's noise did. `+1`/`+0.1` keeps each species' total eating reward per
cycle clearly secondary to reproduction for both (predator: `4.4×1=4.4`;
prey: `60.5×0.1≈6.05`).

**Result**: recovered **82% (predator) / 94% (prey) of sparse's
reproduction rate** — a much better recovery than dense-additive ever
achieved, despite both adding a comparably-sized secondary incentive on top
of the same reproduction bonus. Stable from early in training (checkpoint
~290/1000 already showed 83%/93%) through the final checkpoint — not a late
fluke. Strong support for "clean discrete signals cost little, continuous
ones cost a lot" specifically, not just "less shaping is better" in
general.

### `base_environment_sparse_rewards_plus_kickback` — grandparent kick-back (in progress)

A further design discussion (after the eating-bonus result) landed on
testing kin-selection-style reward: keep the `+10` reproduction reward
unchanged, and add a second `+10` **kick-back** to a grandparent every time
its own child successfully reproduces (i.e. every time a grandchild is
born). Fires repeatably — once per grandchild, not capped at one per
lineage — and only if the grandparent is still alive to collect it (RLlib
cannot deliver a new reward to an agent already marked `terminated=True`).

This mechanism already exists elsewhere in this repo, in
[`kick_back_rewards`](../kick_back_rewards)
(`_reward_parent_for_child_reproduction`) — verified independently RLlib-
compliant, not affected by the two bugs in section 2. That module was
already tested at `kin_kick_back_reward = 4.0` (~0.4× the `10.0`
reproduction reward) and found no benefit. This module reimplements the
same mechanism in the single-predator/single-prey-type
`base_environment_*` family (directly comparable to the other four runs
here, rather than `kick_back_rewards`' more complex two-type structure) and
tests it at a full **1:1 weight** (`kickback_reward = 10.0`) instead — a
genuinely different, untested magnitude.

**Result: training in progress as of 2026-08-01 — not yet available. Do
not treat any number here as final until this section is updated.**

## 6. Cross-cutting findings

**Episode-length ramp-up is identical across every variant.** How quickly
agents learn to survive a full 1000-step episode at all was a dead heat —
every module tested so far first reached that point at the *identical*
training iteration: 20. Reward density and reward design have shown zero
measurable effect on this axis in this environment.

**Wall-clock ordering is not fully explained.** Sparse is the fastest of
all runs despite supporting the largest final population (more agents
should mean more per-step compute). This is a real, measured pattern, not
a mechanistically confirmed one — flagged as an open question, not a
settled explanation.

**Why sparse wins (best current explanation).** Classic "sparse reward is
hard" problems in RL are usually about *delayed* credit assignment —
reward arriving long after the actions that caused it. The reproduction
reward here isn't delayed; it fires immediately on the step reproduction
happens, it's just infrequent. PPO's value function is specifically built
to bootstrap across gaps like that (`gamma=0.99` gives an effective horizon
of ~100 steps, in range of the ~50-200 step gap the original hypothesis was
worried about). What the hypothesis didn't anticipate: layering a
continuous per-step signal into the *same reward channel* as the
reproduction event makes that one important signal noisier and harder for
PPO to cleanly attribute — even when the reproduction bonus is explicitly
restored on top (additive recovers 63-76%, doesn't fully close the gap).
The eating-bonus result reinforces this: a second *discrete* signal in a
separate, clean event channel costs comparatively little (82-94%
retention), even at a similar total secondary-incentive magnitude to
additive. In short: the sparsity itself wasn't the problem; adding density
introduced a different cost — signal noise — that outweighed the benefit it
was meant to provide.

**A biological-realism framing, not just an RL-engineering one.** Minimizing
hand-designed reward shaping also happens to align with modeling literal
Darwinian fitness: reproductive success, not a proxy signal for it. The
cleanest-performing designs so far (sparse baseline, sparse+eating) are also
the ones closest to "reward = did you reproduce" rather than "reward = a
continuous approximation of how well things are going."

## 7. Implications for the Darwin/Baldwin evolutionary project

This whole investigation started as groundwork for the project's evolutionary
trials (see `project_darwin_baldwin_experiment_goal` in memory). Two
implications follow from what was found here:

- The motivating hypothesis for that groundwork (fix sparsity, then retry) is
  now falsified — reward density is not the fix worth pursuing.
- More significantly: the RLlib-compliance bugs in section 2 are **not
  verified fixed anywhere except the five modules in this folder**. Every
  `eco_evolutionary_*` module used in the Darwin/Baldwin trials (Trials 1-7)
  still has both bugs, unverified. This means those trials may have been
  silently affected by agent-identity conflation (bug #2, when bug #1 is also
  present) throughout their entire history — an open, unconfirmed risk, not
  yet investigated or acted on.

## 8. Open questions and caveats

- **n=1 training run per condition.** Every result above comes from a single
  full training run per module (3 evaluation seeds at the end, not 3
  independent training seeds). Unlike the Darwin/Baldwin trials' established
  3-seed training practice, there's no statistical replication here yet —
  treat magnitudes as indicative, not definitive.
- **Findings may be specific to this environment's event frequencies**, not
  universal. The mechanistic explanation in section 6 rests on this
  environment's particular gap sizes (~50-200 steps) and `gamma=0.99`
  effective horizon (~100 steps) being in the same range; environments with
  much larger gaps or different discounting might behave differently.
  Untested here.
