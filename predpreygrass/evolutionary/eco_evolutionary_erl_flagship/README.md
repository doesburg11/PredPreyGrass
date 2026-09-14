# ERL Flagship (Trial 13) — the proximate-vs-ultimate reward question, in the richer ecology

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

7. **Founding-population sizes are flagship's stock values, not tuned.** An
   earlier version of this module raised `initial_energy_predator`/
   `n_initial_active_prey`/`n_initial_active_predator` to compensate for
   predators starving out fast — but that was compensating for an undertrained
   predator checkpoint and a since-fixed reproducibility bug (see "Status"
   below), and re-tested against the corrected setup, the tuning showed no
   clear benefit over stock. Reverted; see `config.py`'s docstring for the full
   history.

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

**Open, accepted limitation: even a confirmed-adapting, correctly-rewarded
predator still eventually goes extinct in every seed tested (15/15).** With
predator competence and reward design now both ruled out as the cause across
three different predator designs and two reward-signal iterations, this
cleanly isolates the remaining blocker as population SCALE, not behavior: a
founding cohort of 4-6 predators is small enough that one unlucky stretch —
regardless of how well they hunt — can wipe it out before it recovers, since
flagship has no immigration/reseeding mechanism. This is the founder-
population-sizing question (deviation 7) again, now unconfounded by any
remaining predator-competence or reward-design question. Not yet addressed.
The run-stops-on-predator-extinction behavior (deviation 6) means a pilot seed
that loses predators early just ends early and can be rerun, rather than
wasting budget on unchecked prey growth.

Ready for the founder-population-sizing calibration pass, now well-isolated as
the real remaining blocker — see the Darwin/Baldwin Trial Log for status.

## Usage

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
