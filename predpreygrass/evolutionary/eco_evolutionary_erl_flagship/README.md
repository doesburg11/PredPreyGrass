# ERL Flagship (Trial 13) — the proximate-vs-ultimate reward question, in the richer ecology

## Key message

**Caution beats recklessness, decisively, once they actually have to compete —
and it beats the project's own literal fitness signal too.** A reward that
makes prey reckless around predators (`anti_adaptive`) can look fine, even
thrive, when it never has to share space with more cautious prey — tested
alone, it produced *more* total offspring and far deeper lineages than a
predator-avoiding strategy (`avoider`), by benefiting from the extra food its
own predation losses freed up. But put the two side by side, competing for the
same food and space, and caution wins outright: in 9 of 10 evolutionary runs,
the cautious strategy completely excludes the reckless one within ~27
generations. `avoider` even beats flagship's own original "+10 on
reproduction, nothing else" reward — the fitness criterion itself, no proxy
involved — 27 times out of 30 (p=0.000008), because a sparse signal gives this
module's weak, one-step learning rule almost nothing to learn from between
rare reproduction events.

**One precision this whole arc earns, not a footnote: `avoider` was hand-designed
by a human, not evolved.** It was never discovered by this module's own
mutation-and-selection process — separately confirmed (via the Hunt-test drift
analysis below) that this trial's actual evolutionary mechanism does NOT
reliably converge on anything like it; it drifts instead. So the honest reading
is two separate claims, not one: a well-designed dense reward has real,
demonstrated power in this architecture (true) — but evolution here is shown
to be the mechanism that FINDS such a reward (not demonstrated; the opposite
was shown). Reward design matters enormously; the design came from the
experimenter, not from evolution.

So there is no single abstract "optimal reward" for evolution to converge on
in this ecology — but not because reward-shaping is arbitrary. It's because
"best" depends on what else is in the population, resolves only once you run
the actual competition, AND (per the precision above) isn't something this
trial's evolutionary process actually finds on its own even when a good answer
demonstrably exists. That's a more informative, more biologically realistic
conclusion than either of the two wrong turns taken en route to it in this
module's history below: not "it's all just random noise" (a measurement
problem, corrected), and not "the reward never reaches behavior at all" (also
wrong, corrected once tested properly at adequate statistical power). See
"Status" below for the full, evidence-by-evidence path to this conclusion,
including the overclaims caught and fixed along the way.

## Why this module exists

Trial 12 (`eco_evolutionary_erl_baldwin`) found, at n=30 confirmed, that evolution's
discovered reward function (`genome.eval_weights`) diverges from the fitness
criterion it's selected for (`offspring_count`): agents consistently over-weight
`health_norm`/`energy_norm` relative to their near-zero correlation with realized
reproduction. That result lives in a small, abstract World-AL-style ecology (a
custom pure-Python simulator, hand-designed 7-channel ray-cast observation).

This module asks the same question — does an evolved reward function diverge from
fitness? — inside the project's flagship module
(`predpreygrass/non_evolutionary/base_environment`) instead: a richer, 25×25
predator/prey/grass ecology with a spatial image observation, normally trained via
RLlib PPO against a fixed sparse reward. Only **prey** get an evolved reward
genome here — genuinely discovered, individually-inherited, the subject of the
whole trial. Predators are a real, adapting threat but not themselves a subject
of reward-discovery: they run a single, centrally-updated policy learned online
via REINFORCE (see "Predator strategy" in `config.py`, and deviation 4 below) —
after two earlier predator designs (a frozen PPO checkpoint, then a rule-based
hunter) were tried and diagnosed as insufficient. See "Status" for the full history.

## Architecture

Ports `eco_evolutionary_erl_baldwin`'s genome/REINFORCE machinery
(`genome.py`, `networks.py`) onto flagship's actual grid/energy/reproduction
mechanics, rather than reimplementing them:

- **Evaluation network** (`genome.eval_weights`, `eval_bias`): fixed for the
  agent's entire life, evolved only via reproduction. A linear map from an
  8-channel feature vector (`features.py`) to a scalar "goodness" value.
- **Action network** (`genome.action_weights`, `action_bias`): the genome's
  *initial* weights only. A live copy (`driver.PreyGenomeState.action_weights`)
  is adjusted every step by the same REINFORCE-style update as Trial 12
  (`networks.reinforce_update`), using the agent's own eval-network output as
  its intrinsic reward — no external reward signal.
- **Reproduction**: strictly Darwinian — always copies from `genome`, never from
  the live, post-learning `action_weights` (see
  `tests/test_genome_inheritance.py::test_offspring_genome_does_not_inherit_parents_learned_weights`).

flagship's own env (`PredPreyGrass`) is reused directly as a plain Python
simulator (`env.step(action_dict)` in a hand-written loop, `driver.py`) — no
`ray.init()`/`PPOConfig`/`Tuner` for Trial 13 itself. See "RLlib or not" below.

## Deviations from Trial 12 — read before trusting a result as directly comparable

1. **Asexual, mutation-only inheritance, not sexual crossover.** flagship's actual
   reproduction trigger (`predpreygrass_rllib_env.py:441-469`) is asexual/clonal —
   one parent's energy crosses a threshold, one child spawns adjacent. Trial 12's
   `_nearest_mate` + `genome.crossover` has no flagship equivalent and doesn't
   port; inventing new mate-search logic would mean reinventing flagship's
   reproduction trigger rather than hooking into it. `genome.mutate` (rate=0.05,
   std=0.05) is the only inheritance mechanism here.

2. **8 feature channels, not 7 — and two of Trial 12's channels are dropped.**
   `features.py`'s `[energy_norm, predator_dx, predator_dy, predator_proximity,
   food_dx, food_dy, food_proximity, local_grass_density]` echoes Trial 12's
   channel philosophy but isn't the same list: `in_tree` has no flagship
   equivalent (no shelter mechanic), and flagship prey have no `health` distinct
   from `energy` (no separate `health_norm`). This is a hand-reduction of the
   *same* raw window flagship's own CNN policy already receives (via
   `env._get_observation`, reused directly) — not additional information.

3. **Two required `config_env` overrides, both load-bearing, not cosmetic:**
   - `max_steps`: flagship's default (1000) truncates the "episode" and resets
     every agent to initial positions/counts. Overridden to an effectively
     unbounded value (10,000,000) so extinction is the only natural stopping
     condition, matching Trial 12's design.
   - `n_possible_prey`/`n_possible_predators`: flagship allocates agent IDs
     monotonically and **never recycles them on death**
     (`predpreygrass_rllib_env.py`'s `_next_prey_idx`/`_next_predator_idx`).
     Reproduction silently and permanently stops once cumulative births hit this
     cap (2000 by default) — invisible under flagship's own max_steps=1000
     (which resets the counter every episode), but a hard ceiling on TOTAL
     lifetime births once that reset is removed. Raised to 500,000 here.

4. **Predator handling: a centrally-learning shared policy, after two other
   designs were tried and diagnosed as insufficient.** `centralized_predator.
   CentralizedPredatorPolicy` is ONE small linear policy, shared by every
   predator, updated online via REINFORCE from every predator's own real
   experience (net energy change, baseline-subtracted -- see `driver.py`'s
   `_select_predator_action`). Two earlier designs are kept in the module
   (not deleted) as tested, working, documented alternatives:
   - `predator_policy.FrozenPredatorPolicy` -- a frozen, pretrained PPO
     checkpoint, inference-only. Reused a known-good, already-calibrated
     adversary rather than open-ended new tuning. Diagnosed as fundamentally
     mismatched to genome-driven prey (see "Status").
   - `rule_based_predator.RuleBasedPredatorPolicy` -- a simple FSA (move
     toward nearest visible prey, explore otherwise), like Trial 12's
     `Carnivore`. Fixed the mismatch but couldn't adapt, causing either
     predator or prey extinction depending on seed.
   See "Status" for the full diagnostic chain.

5. **No RLlib training loop for Trial 13 itself ("RLlib or not").** Concurrent
   population is small; an 8×9 linear genome (~90 params) is the same scale
   network Trial 12 already runs fast in pure Python. RLlib's env-runner/learner
   abstractions assume one stable shared-weight policy across a whole collected
   batch before a synchronized gradient step — there's no clean way to express
   "spawn a new module with mutated weights mid-episode, for exactly one agent
   id" inside `MultiRLModuleSpec`/`policy_mapping_fn`. `RLModule.from_checkpoint`
   is used for exactly one thing: the frozen predator.

6. **The run stops when EITHER population goes extinct, not just prey.**
   `run_trial13_simulation.py`'s main loop breaks on `prey_count == 0` (matching
   Trial 12) or on `predator_count == 0` — once the fixed predator threat is
   gone, the rest of the step budget would just watch prey grow unchecked with
   no predation pressure, which isn't what this trial tests and wastes compute.
   `eval_checkpoint.py`'s eval loop does the same.

7. **Founding-population sizes are tuned, from a real systematic sweep — not
   flagship's stock values, and not a guess.** `n_initial_active_prey=24` (was
   8), `n_initial_active_predator=8` (was 6), `initial_energy_predator=10.0`
   (was 5.0). Two earlier, narrower tuning attempts were tried and abandoned
   first (one compensated for an undertrained predator checkpoint and a
   since-fixed reproducibility bug; neither applies anymore). This tuning is
   different: run only after predator competence and reward design were both
   independently validated, via a real 15-config x 10-seed sweep, not
   incremental guessing. See `config.py`'s docstring and "Status" below for
   the full history and the important reframe about what this tuning does
   and doesn't achieve.

## Status (2026-09-13)

**Stage 0 (smoke test) passed all its mechanics checks**, both via the unit test
suite (`tests/`, 23/23 passing) and real live runs: `env.step()` runs continuously
past the 1000-step boundary with no premature reset; reproduction/mutation/
lineage-logging/checkpoint-round-trip all work correctly on real data (verified
via `eval_checkpoint.py` against a saved checkpoint — distinct evolving lineages,
generations incrementing, genomes mutating parent-to-child as expected); the
`n_possible_prey` override is nowhere near exhausted at this scale.

**Two real bugs found and fixed during calibration, both about reproducibility,
not simulation logic:**

1. **Predator action sampling was silently unseeded.** `FrozenPredatorPolicy.act()`
   used `torch.distributions.Categorical(...).sample()`, which draws from
   PyTorch's global RNG — never touched by `--seed`, which only seeds
   `driver.rng` (a NumPy `Generator`, for genome/mutation/prey action sampling).
   Found by running the identical `--seed 2` invocation twice and getting wildly
   different population trajectories. Fixed: `FrozenPredatorPolicy` now owns its
   own `torch.Generator`, seeded explicitly, and samples via `torch.multinomial`
   (which accepts a `generator`) instead of `Categorical` (which doesn't).

2. **Founder agent/grass placement was silently unseeded too, and was the
   dominant cause.** `driver.reset()` called `env.reset()` with no arguments;
   flagship's own `PredPreyGrass.reset(seed=None)` only (re)seeds its internal
   placement RNG when given an explicit seed — otherwise it draws fresh OS
   entropy every time (`predpreygrass_rllib_env.py:141-142`). So founder
   positions — which determine early predator/prey encounter geometry, the
   single biggest driver of whether predators find food in time — were never
   actually controlled by `--seed` at all. Fixed: `driver.reset()` now calls
   `env.reset(seed=self.cfg.get("seed"))`. Verified: the identical `--seed 2`
   invocation run twice now produces bit-for-bit identical `progress.csv` output.

Both bugs meant every "seed comparison" run before this fix (including an
in-progress Stage 1 pilot at the time) was not actually testing reproducible
seeds — each invocation was an independent random draw that happened to be
*labeled* with a seed. That data was discarded, not treated as a result.

**Predator checkpoint switched from iteration 110 to iteration 1000** (the same
tournament run's final, most-converged checkpoint). Iteration 110 was chosen
originally on the theory that an early, less-converged predator would be a
gentler adversary; in practice, once the above bugs were fixed and its
behavior could actually be observed reliably, it looked erratic rather than
gentle. Iteration 1000 produced clearly more legible dynamics — but see below:
the checkpoint route was abandoned entirely regardless of iteration.

**Root-caused why the frozen-checkpoint predator always went extinct (30/30
seeds), rather than accepting it as calibration noise** (prompted directly by
a sharp "this can't be right" from the user, comparing against
`base_environment`'s own tournament-matrix data — see its README's Master
Tournament section). Measuring action-distribution entropy directly: genome-
prey's randomly-initialized 8-feature linear policy is close to UNIFORM RANDOM
movement (entropy ~1.92 of a 2.197 maximum), while even the EARLIEST available
real `prey_policy` checkpoint (iteration 10, the very first save) is already
noticeably more structured/predictable (~1.76, individual agents sometimes
>85% probability on one action) — an architectural property of the CNN, not a
training-progress effect. Cross-checked directly against the real tournament
matrix data (`results_long.csv`): a mature predator vs. an early-iteration
`prey_policy` shows 3% predator extinction / 39% prey extinction over 165
real episodes — predators dominate. The resolution: "untrained" means
something different in each case. Tournament-matrix "untrained" prey are
unskilled but still confidently structured (a CNN property, present even at
iteration 10); genome-prey's "untrained" is genuinely close to random noise. A
predator's whole learned pursuit strategy is calibrated to exploit STRUCTURE
in movement — something it encountered at every stage of its own training,
skilled or not — and has no grip on real randomness, regardless of iteration.
Confirmed by two further controls: the frozen checkpoint thrives (6→13,
6→19-21) against its own co-trained `prey_policy` under identical code/RNG;
and disabling genome-prey's within-lifetime learning entirely (frozen random
weights, no REINFORCE) made predators die at the same rate — ruling out live
adaptation as the cause and isolating the initial weight-scale/entropy
mismatch as the actual mechanism.

**First fix attempted: `rule_based_predator.RuleBasedPredatorPolicy`** (move
toward nearest visible prey). Directly addresses the diagnosed mechanism — no
calibration against any particular prey movement distribution needed.
Confirmed working correctly (catches the nearest prey, explores when none
visible). But screening 15 seeds at 20,000 steps found it swings between two
failure modes depending on seed: predator extinction (10/15, 99-461 steps) or
prey extinction from too-efficient uncalibrated hunting (5/15) — zero of 15
seeds reached a stable, long coexistence. Being purely reactive, it can't
adapt its behavior to the actual population dynamics it's facing.

**Second fix, current default: `centralized_predator.CentralizedPredatorPolicy`**
— one small linear policy shared by every predator, updated online via
REINFORCE from real experience (net energy change), pooling every predator's
transitions into the same weights rather than each relearning independently
(proposed directly by the user as a way to get real adaptation without
per-agent training cost). Caught and fixed one real bug along the way: the
raw energy-change reward was dominated by the ambient per-step energy drain
(present on almost every step, since catches are rare), so training on it
directly mostly taught "whatever I just did was bad" uniformly rather than
"catching is good" — confirmed directly, `predator_action_weight_absmean` was
essentially flat (0.4037→0.4027) across a 400-step run. Fixed by
baseline-subtracting the ambient drain (`energy_change +
energy_loss_per_step_predator`), so an ordinary no-catch step nets to ~0
reinforcement (a no-op) and a catch is an isolated, clear positive signal.
Confirmed working after the fix: `predator_action_weight_absmean` now shows
real, sustained movement (e.g. 0.373→0.364→0.432→0.452 across one run), and
re-screening 15 seeds showed a real improvement in survival (mean ~231 steps,
vs. ~147 for the rule-based hunter; longest run 888 steps, vs. 461).

**A second, related reward bug found while answering a user question about
what exactly predators are rewarded for**, and fixed the same way: reproduction
ALSO costs the parent `initial_energy_predator` on top of the ambient drain
(`predpreygrass_rllib_env.py:423`), landing on the same step as whatever
action the predator happened to take — without correction, reproducing (a
*good* outcome, reflecting past hunting success) showed up as a large spurious
*negative* reinforcement for an essentially arbitrary action, since
reproduction is triggered by an energy threshold, not caused by that step's
action. Fixed by excluding the reproduction cost from the reinforcement
too (`driver.py`'s `_predators_reproduced_last_step`, detected from the env's
own `reproduction_reward_predator` signal). Re-screening the same 15 seeds
showed a further, real improvement: mean survival ~252 steps, longest run 811
steps (up from 261 for that seed pre-fix).

**Confirmed predator competence and reward design were never the remaining
cause, via a sharp user question about the base_environment reward-density
result.** The project's own `project_reward_shaping` study found sparse
reproduction-only reward beats every hand-crafted denser alternative for PPO
— reasonably raising the question of whether the predator's reward here
should mirror that (sparse, fitness-identical) instead of the catch-based
signal. Tested directly rather than assumed: literal `+10`-on-reproduction
sparse reward performed WORSE for this predator (mean 162.6 steps, vs. 252 for
the catch-based signal) — because PPO's sparse-reward result depends on
`gamma=0.99` + GAE value-function bootstrapping to propagate a distant reward
backward across the many steps that caused it, and this predator's 1-step
REINFORCE has no equivalent mechanism. A sparse reproduction-only reward here
mostly credits whatever arbitrary action a predator took on the exact step
its energy crossed the threshold, not the actual hunting that got it there.
Reproducing the reward-density result's *form* without its *mechanism*
doesn't reproduce its benefit.

**A systematic founder-population-size sweep (15 configs x 10 seeds = 150
runs) confirmed population scale as the real remaining lever, and found a
meaningfully better config** — see deviation 7 and `config.py`'s docstring
for the full sweep results and the config now in use
(`n_initial_active_prey=24`, `n_initial_active_predator=8`,
`initial_energy_predator=10.0`): mean survival 487 steps (vs. 310 baseline),
best seed 1629 (vs. 811). Key finding: prey ABUNDANCE is the dominant lever,
not predator count — more predators alone (holding prey fixed) made survival
*worse*, not better, since more mouths compete for the same scarce prey.

**Important reframe, not a remaining bug to keep chasing: none of the 150
swept configurations ever avoided eventual predator extinction.** A finite
population with no immigration/reseeding mechanism is, mathematically, an
absorbing Markov chain — extinction is the only steady state, and tuning
controls the EXPECTED TIME to reach it, not whether it happens. The practical
target is therefore "long enough for a real multi-seed pilot to accumulate
meaningful data," which the tuned config now clears meaningfully better than
the untuned baseline — not "extinction-proof," which isn't achievable by
founder-population tuning at all, at any population size. The
run-stops-on-predator-extinction behavior (deviation 6) means a pilot seed
that loses predators early just ends early and can be rerun or pooled with
others, rather than wasting budget on unchecked prey growth afterward.

**A real n=30 pooled batch at the population-tuned config still showed no
divergence signal — diagnosed as shallow generational depth, not absence of
divergence.** Evolved `eval_weights` across all 30 seeds (4,162 total lifetime
records) were statistically indistinguishable from random initialization,
clustering tightly around `Normal(0, founder_weight_std)`'s expected
magnitude. Checked directly rather than assumed: max `generation` reached
across the entire batch was 9, median 1 — 5%-per-site mutation simply hasn't
had enough successive rounds to move weights anywhere yet, regardless of how
many total lifetime records exist. Survival steps and generational depth are
related but distinct quantities.

**A second sweep targeting generational depth specifically found
`prey_creation_energy_threshold` (how much energy a prey needs to reproduce)
is the controlling lever**, independent of the population-size tuning above.
Lowering it from flagship's stock 8.0 to 4.0 raised mean generational depth
~4-5x in sweep testing (baseline mean 5.1, max 12 → tuned mean ~21-28, max up
to 143) and also increased mean survival.

**This sweep also led to a real bug fix in shared flagship code, not just a
Trial-13-local workaround.** A real CLI batch at the tuned config gave numbers
off by another 4-7x from the sweep's own predictions (mean 298 steps vs.
~1100-1900, max generation 20 vs. up to 143) — traced to flagship's
`_find_available_spawn_position` (`predpreygrass_rllib_env.py`) drawing its
fallback spawn position from the bare `np.random` module instead of the
environment's own seeded `self.rng`, silently breaking `--seed`
reproducibility whenever that fallback fires (rare at low population density,
constant at the densities this tuning induces). Fixed directly in flagship's
file — a one-line change, reviewed with Codex, verified fixed: identical
`--seed` CLI runs now produce bit-for-bit identical output even at this dense
config. See `config.py`'s docstring for the full history.

Ready for a real, pooled multi-seed batch run (Trial-12-style: n=1 → n=2 →
n=30, not a single long run) using the generation-tuned config — see the
Darwin/Baldwin Trial Log for status.

**n=30 at the tuned config still showed no stable "winning channel" across
independent batches, and pushing generational depth further (mean 222
generations, `prey_creation_energy_threshold=4.0` retuned harder) changed
which channel led rather than converging one.** Three independent scale/depth
attempts each showed a real, growing divergence of `eval_weights` from random
initialization (spread ratio vs. founder std: ~1.18x at ~9 generations, ~1.5x
at ~94 generations (n=15), ~1.6x at ~222 generations (n=30)) — but a
*different* channel led each time (energy_norm at n=15/94gen;
local_grass_density at n=30/94gen; food_proximity at n=30/222gen), while every
individual channel's correlation with realized `offspring_count` stayed
robustly ~0 throughout every batch. Reported at the time as a genuinely
unresolved, nuanced finding rather than forced into either a clean positive or
negative result.

**Resolved with the project's existing Hunt (2006)/Lande (1976) drift-vs-
selection model-fitting tool (`predpreygrass/evolutionary/model_selection.py`,
already used elsewhere in the project — see the Hunt model-selection tool
memory), applied directly to per-generation `eval_weights` trajectories from
`lineage_fitness.csv` (its core `fit_all_models` is file-format-agnostic; only
a small adapter was needed since the tool's convenience wrappers assume
RLlib's `result.json` format, which Trial 13 doesn't use).** A single deep
seed (520 generations) initially looked like a clean answer — Stasis
decisively rejected for all 8 channels, and `predator_dx` showing strong,
confident directional selection (GRW favored at 93.5% Akaike weight, a
cumulative directional shift ~2.7x larger than the diffusive noise expected
from drift alone). **That did not replicate.** Refit across all 30 seeds
reaching ≥50 generations (mean depth 439 generations, deepest batch run
against this config):

- **Stasis rejected 0/30 seeds, every one of the 8 channels** — genomes are
  never frozen at founder values; evolution is always moving them somewhere.
- **URW (drift) vs. GRW (directional selection) splits close to 50/50 for
  every channel** (URW favored in 13-22 of 30 seeds depending on channel;
  `predator_dx`'s apparently strong single-seed signal favored GRW in only
  14/30 seeds on refit).
- **Decisive: even in the seeds where GRW is favored, the *sign* of the
  fitted trend (`mstep`) is itself close to a coin flip across independent
  seeds** — 43-57% positive for every channel, no channel exceeding that. If
  any channel were under genuine directional selection, independent
  populations would agree on which direction it moved far more often than
  chance; none of the 8 do. This sign-inconsistency, not the raw URW/GRW
  model-fit counts alone, is what rules out selection: a random walk
  routinely *looks* directional over any one finite window (explaining the
  single-seed result and the shifting "winning channel" across earlier
  batches), but independent replicates of real drift don't agree on which way
  it went.

**Answer to "is there an optimal reward function to be found": no, not for
this feature set.** This sharpens rather than merely replicates Trial 12's
original divergence finding. Trial 12 showed the specific weights evolution
lands on don't correlate with realized fitness — the reward proxy is
imperfect. This result goes further: for essentially all 8 channels, the
*process* generating those weights across generations is statistically
indistinguishable from neutral genetic drift under mutation, not selection
homing in on an imperfect-but-real optimum. There is no reproducible
population-level "best" reward-channel weighting being converged upon here;
the apparent leaders in earlier, shallower batches were drift's leading edge
at that moment, not partial progress toward an answer.

**Checked directly, not assumed: is 439-generation mean depth actually
enough to detect a real but weak selection pressure, or just underpowered?**
A null result from a model-fit never proves absence of an effect on its own
— it can mean "no effect" or "too little data to see it." Ran a power
simulation using the exact same fitting code, the real 30-seed
generation-count/per-generation-sample-size structure, and the real observed
drift-noise scale (`vstep`≈5e-5): injected a known, population-invariant
`mstep` at increasing multiples of `sqrt(vstep)` and refit. A selection
pressure as small as 10-20% of the natural per-generation drift step already
produces 53-100% correct-sign detection power at n=30 seeds, and would show
up as 85-100% cross-seed sign agreement — far above the ~50% actually
observed for every channel. The null result is well-powered, not
underpowered: a universal (same-direction-in-every-population) selection
pressure of any magnitude worth calling "real" would have been visible at
this depth and seed count. One thing this specific test cannot rule out:
a *founder-contingent* landscape where each independently-founded population
converges on a different local optimum (no shared global answer, rather than
no answer at all) — a distinct claim the sign-consistency test isn't built
to detect, since population-specific trends of opposite sign would also
average to ~50% agreement.

**Follow-up, prompted directly by the user finding the drift verdict
unsatisfying ("this makes a more internal reward system, which humans
definitely have, no closer"): does reward-genome content have ANY fitness
leverage in this architecture at all, or is the search space genuinely flat
regardless of what evolution could find?** Two candidate explanations: (a)
behavior differs by genome but this ecology doesn't reward the difference, or
(b) genome never reaches realized behavior at all. Tested directly with
`positive_control.py`: 5 hand-picked, EXTREME, FIXED `eval_weights` vectors
(`forager`, `avoider`, `balanced`, `inert`, and a deliberately adversarial
`anti_adaptive` — rewarded for approaching predators and avoiding food),
`mutation_rate=0.0` so every descendant keeps the exact founder vector
forever (no evolution/drift in the loop at all), n=10 seeds each, 20,000-step
budget. **Zero significant differences anywhere**: Kruskal-Wallis across all
5 conditions gives p=0.95 (total births), p=0.33 (final population), p=0.59
(generational depth). Even the starkest contrast available —
`avoider` (punished for predator proximity) vs. `anti_adaptive` (rewarded for
it) — shows no difference in final population (Mann-Whitney p=0.91). None of
the 50 runs went extinct.

**Distinguished (a) from (b) directly with `behavior_diagnostic.py`**: measure,
every step, which direction a prey's CHOSEN action points relative to a
visible predator/food at the exact moment it was chosen (dot product of the
action's move vector with the threat/food offset) — same "measure behavior
directly, don't assume it" method that root-caused the earlier
predator-transfer puzzle via entropy. A Codex review of the first version
caught two real bugs before this result was trusted: measuring realized
*post-step* distance instead (a) excluded agents that died the same step from
the sample (survivorship bias -- silently dropping exactly the "approached a
predator and got caught" cases most relevant to the predator statistic), and
(b) confounded the prey's own choice with the predator's own simultaneous
movement and possible nearest-target changes. Fixed by measuring the chosen
action's direction against the pre-step observation instead, which needs no
post-step state at all (so dying agents are included) and isolates the one
thing under the genome's control. n=5 seeds/genome, 5000 steps, corrected
version: `avoider` (punished for predator proximity) had its actions point
toward a visible predator 36.0% of the time; `anti_adaptive` (rewarded for
the opposite) pointed toward one 36.6% of the time — statistically
indistinguishable (Kruskal-Wallis across 4 genomes, p=0.45). Food-approach
direction likewise indistinguishable across genomes (p=0.996) at this n and
depth. **Read at the time as "(b) confirmed, (a) ruled out: behavior itself
never differentiates by genome at all" — corrected below, this was too
strong a claim, itself a false negative from insufficient statistical power,
not a genuine null.**

**Correction, found while testing whether strengthening within-lifetime
learning would fix the (mis-diagnosed) problem:** `lr_sweep.py` tests
`avoider` vs. `anti_adaptive` at increasing learning-rate multipliers, using
a properly paired design (same seed range reused across both genomes --
matched environment/predator-policy draws per seed) with a Wilcoxon
signed-rank test and Holm-Bonferroni correction across multipliers tested (a
Codex review caught that an earlier version of this script used an unpaired
Mann-Whitney test, which does not use the pairing the design actually has,
and reported significance from a single uncorrected p-value across 4
multipliers -- both fixed). **At n=5 seeds, nothing reached significance
after correction, including at higher learning rates (p_holm=0.25 best
case) — consistent with the "genome never reaches behavior" claim above.
But n=5 is too few for a paired Wilcoxon test to ever reach significance
after Holm correction (its raw-p floor at n=5 is 0.0625, already above the
corrected threshold) — indistinguishable from "truly no effect" using that
test alone.** Rerun at n=30 seeds (properly powered from the start this
time, not incrementally): **the default learning rate (1x, the config used
for every result above) already shows a real, Holm-corrected-significant
difference** — `avoider` points toward a visible predator 37.1% of the time
vs. `anti_adaptive`'s 37.9% (p_holm=0.0155). The gap grows sharply with
learning rate: 34.8% vs. 40.2% at 5x (p_holm<0.0001), 29.6% vs. 41.5% at 20x
(p_holm<0.0001).

**Revised conclusion: the reward genome DOES reach behavior, even at the
architecture's default learning rate — the effect just starts small (about
1 percentage point) and needs n=30 with a properly paired test to detect
reliably, which none of this section's earlier diagnostics (n=5-10,
unpaired, testing 4-5 genomes at once rather than the single starkest
matched contrast) had the power to do.** This does not overturn the earlier
Hunt-test drift finding (that used real per-generation trajectories at much
larger effective sample sizes, not a small hand-picked contrast) or the
`positive_control.py` fitness-leverage null (still only tested at n=10,
default LR -- an open question given what's now known about power, not yet
rechecked at matching rigor). It does mean **"genome never reaches behavior
at all" was an overclaim** -- the honest, current picture is "a real but
small effect on behavior at baseline, growing substantially with learning
rate; whether that behavioral effect is large enough to matter for realized
fitness is the next open question, not yet answered at adequate power."

**Answered: rechecked `positive_control.py`'s fitness question at matching
rigor (n=30, paired Wilcoxon + Holm correction, `--lr-multiplier` support
added) -- does the confirmed behavioral effect actually move fitness?** At
baseline (1x) learning rate, still no significant fitness difference between
`avoider` and `anti_adaptive` (total offspring p=0.13, final population
p=0.89) -- the small baseline behavioral gap doesn't reach a detectable
fitness signal at this n. **At 20x learning rate, where the behavioral gap
is large, it does move fitness -- decisively, but not as a simple "good
genome wins":**

| metric | avoider | anti_adaptive | p (Holm) |
|---|---|---|---|
| total offspring | 8,615 | 12,070 (+53%) | <0.0001 |
| final population | 75.4 | 68.5 (-10%) | 0.0009 |
| mean lifespan | 175.5 steps | 108.8 steps (-38%) | (direct lineage check) |
| generational depth | 214 | 544 (+154%) | (direct lineage check) |

`anti_adaptive` prey (rewarded for approaching predators, avoiding food) die
38% faster and produce 53% MORE total offspring and 2.5x deeper lineages,
while ending with a SMALLER standing population than `avoider`. Confirmed
mechanism, not just inference: higher predation mortality culls the
population faster, which relieves food/grass competition among survivors,
letting them cross the reproduction threshold more often per unit time --
classic density-release/compensatory population dynamics, not a defect in
the measurement. **Whether reward genome "wins" depends on which fitness
currency is used** -- standing population size (avoider ahead) or total
reproductive throughput and generational depth (anti_adaptive ahead, by a
lot). This ambiguity is plausibly part of why the Hunt test found no clean
directional selection signal on `eval_weights` in the first place: if
different fitness measures point in opposite directions for the same
genome, a single scalar notion of "selection" can look driftlike even when
the genome is doing something real and measurable.

**Pushed one step further: does a REAL, competing, evolving population
(not two segregated single-genotype populations) maintain BOTH strategies
at once, rather than one driving the other extinct?** `avoider` and
`anti_adaptive` were each tested in its own isolated population -- neither
had to compete with the other for the same food/space. Tested directly with
a new `polymorphism_check.py`: run real evolution (mutation on, standard
neutral random founder init, no fixed genomes) at 20x learning rate for
~260 mean generations (n=10 seeds), and test at each of 10 checkpoints
whether the live population's `predator_proximity` eval_weight -- the axis
that separated `avoider` (-2.0) from `anti_adaptive` (+2.0) -- becomes and
stays bimodal, via a 1- vs 2-component Gaussian mixture model compared by
BIC (same "fit competing models, compare by information criterion" logic as
the Hunt test, applied to population structure here). Rigor added after a
Codex review of the first version (mirroring the `lr_sweep.py` false-positive
lesson earlier in this investigation): a null-calibrated false-positive
check (the identical procedure applied to a matched-size, matched-mean/std
synthetic unimodal sample, at every checkpoint) and `n_init=10` for the
2-component fit (not sklearn's single-init default, to avoid reporting a
local optimum).

**Result: partially confirmed, but much more modest than the headline
hypothesis.** The population does maintain real, non-noise internal
structure rather than collapsing to a single point -- the null-calibrated
false-positive rate was 0% at every checkpoint (so this isn't the test
firing spuriously), and 6/10 seeds showed a persisting two-component signal
across their last 3 checkpoints. But the gap between the two detected modes
is small (mean ~0.10-0.17 units) and, checked directly across the whole
run, does NOT grow over ~260 generations -- it stays flat or even shrinks
slightly. **The two dramatic, fitness-consequential strategies from the
positive-control experiment (separated by 4.0 units) do not spontaneously
emerge and stably coexist within a real, competing population at this
depth.**

Likely mechanism, not just a negative result: the "density release"
advantage that made `anti_adaptive` look strong in `positive_control.py`
was measured with it as the ONLY genotype present, monopolizing the whole
population's resource release from its own predation losses. A rare
`anti_adaptive`-type mutant inside a mostly-`avoider` population wouldn't
get that same benefit -- its own predation losses barely dent food
competition for the much larger surrounding population. Whether a
polymorphism is evolutionarily stable depends on RARE-STRATEGY (invasion)
fitness, not on how each strategy performs in isolation -- a different
question from what the segregated fixed-genome comparison tested, and one
this real-population check answers directly: at this generational depth, no.

**Tested that different question directly: if the split is ALREADY
established (not asked to spontaneously emerge), is it then maintained?**
`polymorphism_maintenance_check.py` seeds founders 50/50 directly from
`avoider` and `anti_adaptive` (`driver.py`'s new `mixed_founder_weights`
cfg option, each founder randomly assigned to one cluster; mutation stays
on, same 20x LR, n=10 seeds, 20,000 steps). **Answer: no -- and decisively,
not a borderline result.** At every one of 10 checkpoints per seed, every
live agent in 9/10 seeds was classified to `avoider`'s side and every live
agent in the remaining 1/10 was classified to `anti_adaptive`'s side --
zero intermediate values observed anywhere, in 100 checkpoint-seed
combinations. The split resolves to complete exclusion by the FIRST
checkpoint (~27 generations) and never reverses afterward. Verified this
isn't a founder-count artifact: founder composition varied naturally around
50/50 (binomial noise, e.g. seed 5 started 7/24 `avoider` vs. 17/24
`anti_adaptive` -- a clear minority), yet `avoider` still won completely in
8 of 9 seeds where it wasn't the founder majority, including that 7-vs-17
minority start. Only the single most extreme minority start (5/24) went the
other way. **`avoider` has a real, substantial competitive edge once both
strategies share the same food/space** -- strong enough to overcome a
sizeable founder disadvantage in most cases -- consistent with the
mechanism proposed above: `anti_adaptive`'s density-release advantage in
the segregated comparison depended on monopolizing an entire population's
predation losses, a benefit a competing `avoider` population, which itself
suffers far less predation, ends up capturing instead.

**Closing picture for this whole line of investigation:** the reward
genome is real, and moves both behavior and fitness once learning is strong
enough -- but that doesn't make an ecological trade-off automatically into
a stable population-level polymorphism. `avoider` and `anti_adaptive` are
not two co-viable strategies waiting to be discovered together; once they
actually compete, one wins, consistently, for an identifiable ecological
reason. The earlier "what does an optimal reward function even mean" framing
undersold this: there IS a real answer here, for THIS pair of strategies in
THIS ecology -- it's just not visible from segregated single-genotype
comparisons alone, and required actually running the competition to find.

**One more question this raised directly: does that mean the project's own
original reward design (flagship's `+10` on reproduction, nothing else --
literally the fitness criterion itself, no proxy involved) can't be beaten?**
Tested directly with a new `sparse_reward_check.py`, against exactly the same
mixed-competition methodology as the maintenance check above (real evolution,
50/50 founders, 20x LR, n=30 seeds). Two build notes first: (1) `eval_weights`
is a linear function of the 8 OBSERVED features, and "did I just reproduce"
isn't one of them -- a genome literally cannot represent this reward. The
faithful implementation bypasses `eval_weights` entirely for this condition
(`PreyGenomeState.sparse_mode`, a lineage tag inherited unchanged, never
mutated) and feeds the environment's own reproduction signal directly into
the REINFORCE update -- not a proxy genome, the actual thing. (2) This isn't
"REINFORCE can't handle sparse reward" in general -- textbook REINFORCE uses
full-trajectory Monte Carlo returns and handles delayed reward fine, just
with high variance. It's specifically this module's one-step simplification
(`reinforcement = e_now - prev_eval`, a documented simplification of Ackley &
Littman 1991, not a full-trajectory return) that has no multi-step credit
assignment at all.

**Result: no, it is beaten -- decisively.** `avoider` (dense, hand-shaped,
genome-driven) wins 27/30 seeds against the sparse reproduction-only signal
(binomial p=0.000008), reaching 90% mean population share, and -- unlike the
`anti_adaptive` comparison, where the loser at least produced more total
offspring -- `avoider` wins BOTH metrics cleanly here (7,402 vs. 1,595 total
offspring). Mechanistically consistent with everything above: sparse
reproduction-only reward gives this architecture's one-step update almost
nothing to learn from between the rare reproduction events, so a
sparse-reward prey behaves close to its random initial policy for its whole
life -- more like the earlier `inert` genome (which also received ~zero
reinforcement every step) than like a competently-trained agent. (Smaller,
n=10 first pass gave a weaker, non-significant 7/10 split -- scaled to n=30
for the properly-powered result above, the same lesson this investigation
has repeated at every stage: check power before trusting a margin.)

**This completes the algorithm x reward-density comparison as a real 2x2,
not two studies stitched together (a real confound flagged directly by the
user -- comparing "sparse+PPO" against "dense+this module's REINFORCE"
conflates reward density with algorithm, since both differ at once):**

| | sparse reward | dense/shaped reward |
|---|---|---|
| **PPO (gamma+GAE)** | wins (project's reward-density initiative) | loses |
| **this module's one-step reinforcement** | loses (above, n=30, p<0.0001) | wins (`avoider` beats `anti_adaptive`, n=30) |

Reward density's effect isn't context-free -- it flips depending on whether
the learning algorithm can do multi-step credit assignment. Sparse, truthful
reward needs a capable learner to turn into behavior; a capable learner (PPO)
doesn't need reward shaping and can be hurt by it (reward hacking, misalignment
risk). A weak learner (this module's one-step rule) needs the denser signal
regardless of how well-aligned the sparse alternative is with fitness. Neither
"sparse beats dense" nor "dense beats sparse" is the general truth -- which
one wins is a property of the algorithm x reward-density pair, not either
factor alone.

**One more precision, worth stating plainly: BOTH "dense/shaped" cells in this
table are HAND-DESIGNED, not evolved.** `project_reward_shaping`'s dense
reward (`energy_after - energy_before`, tested against sparse under PPO) was
a single a priori formula an experimenter wrote, never tuned or searched
(n=1 per variant, per that module's own README). `avoider` was, the same way,
hand-picked by the experimenter as a deliberately extreme "obviously good"
predator-avoidance vector for this comparison -- not a genome this module's
own evolutionary process discovered (see the Key message above and the Hunt
test's drift finding: real evolution here does not reliably converge on
anything like `avoider`). So this table is a genuine, fair 2x2 on the
algorithm x reward-density axes -- but the two sides aren't symmetric in one
respect worth being explicit about: PPO has no mechanism to search over
reward functions at all -- a human necessarily supplies its reward, dense or
sparse, by construction, so "would PPO find a good reward on its own" isn't
even a meaningful question to ask about it. This module is different: it HAS
a built-in reward-discovery mechanism (mutation and selection over
`eval_weights`), and that mechanism was tested directly and found not to
discover a reward like `avoider` on its own. So "dense/shaped wins" in this
table demonstrates that a good dense reward has real power here once
supplied -- for the PPO side, that's the whole, unremarkable story; for this
module's side, it leaves open (and this trial separately answers, in the
negative) whether the system's own evolutionary process is what supplies it.

```bash
# Stage 0: smoke test, mechanics only
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation \
    --steps 5000 --seed 1 --log-every 200

# Inspect a checkpoint standalone (no training loop)
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.eval_checkpoint \
    <out_dir>/checkpoints/checkpoint_step_5000.pkl --steps 0

# Proximate-vs-ultimate reward analysis, once a real lineage_fitness.csv exists
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.analyze_proximate_reward \
    <out_dir>/lineage_fitness.csv
```

Run the test suite: `pytest predpreygrass/evolutionary/eco_evolutionary_erl_flagship/tests/`
