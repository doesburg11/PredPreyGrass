# ERL Baldwin — a structurally different architecture, not a new trait

## Why this module exists

Trials 1-9 (see `predpreygrass/evolutionary/RESULTS.md`) all share one architecture:
**a single PPO policy shared across every agent of a species, with genome as a
side-channel scalar** (`metabolic_rate`, `plasticity`, ...) that modulates
physics or a bonus term, but never touches the policy itself. Nine-plus trials
under that architecture came back flat or null. A deliberately extreme
positive control (`eco_evolutionary_metabolic_rate_positive_control`) showed
the pipeline *can* detect a weak selection signal given a 16x fitness
gradient, and ruled out mutation rate as the dominant bottleneck (Pilot 2) --
pointing more at population size. But one architectural question was never
tested at all: **can genome causally shape behavior, individually, given the
shared-policy design has no mechanism for that?**

This module tests that question directly, using a real, working precedent
rather than a new invention: **Ackley & Littman (1991), "Interactions Between
Learning and Evolution,"** *Artificial Life II* -- "Evolutionary
Reinforcement Learning" (ERL). Each agent has its **own** genome-initialized
network and its **own** lifetime of local reinforcement learning, combined
with GA-style reproduction. They demonstrated the Baldwin Effect concretely
in this architecture and measured it with a method that doesn't fight the
population-size noise floor that's dogged every trial so far.

This is a new, parallel avenue -- not a replacement for the PPO-based trial
family, and not a claim that PPO or the earlier trials were wrong. It's a
different, cheaper, literature-validated bet on the one thing those trials
structurally cannot test.

## The mechanism (from the paper, read in full 2026-08-09)

Each agent carries two single-layer networks:
- **Evaluation network** (`genome.eval_weights`, `eval_bias`): fixed for the
  agent's entire life. Maps observation -> a scalar "goodness" value. This
  *is* the genetically inherited goal -- never touched by learning.
- **Action network** (`genome.action_weights`, `action_bias`): only the
  *initial* weights are genetic. A live copy is made at birth
  (`Agent.action_weights`/`action_bias` in `world.py`) and adjusted every
  step by reinforcement learning.

Reinforcement signal: `R_t = E_t - E_{t-1}` -- literally "am I better off now
than a moment ago, by my own inherited sense of good." No externally supplied
reward function.

**Reproduction copies the genome record, never the live action network.**
Whatever an agent learned during its life is discarded at reproduction; only
the pre-learning genome (plus mutation/crossover) is passed on. This is the
whole reason it's Darwinian, not Lamarckian -- not a convention, a hard
architectural fact: reproduction reads from a record that learning never
writes to. `world.py`'s `_handle_reproduction` enforces this explicitly
(`child_genome = agent.genome.copy()` / `crossover(agent.genome, mate.genome, ...)`
-- always `.genome`, never `.action_weights`), and
`tests/test_erl_baldwin.py::test_offspring_genome_does_not_inherit_parents_learned_weights`
asserts it directly.

Their headline finding: combined evolution+learning (ERL) produced far more
long-surviving populations than evolution alone, learning alone, or no
adaptation -- and evolution alone did *surprisingly badly*. Their
explanation: it's much easier to genetically specify a compact **goal** (one
evaluation-network weight: "food is good") than to specify the full
**behavior** needed to act on it (many action-network weights). Genes encode
*what*; learning fills in *how*.

They detected genetic assimilation (the actual Baldwin Effect) via
**functional-constraint analysis**: track, per genome site, how much that
site's value changes across a lineage over generations. Sites that matter for
survival get purged of mutations (low change rate = "constrained"); sites
that don't matter drift freely. Early in a run, evaluation-network sites were
constrained (the learned goal was doing the work); by ~3 million steps,
action-network sites became constrained instead (the behavior had been
assimilated -- agents approached food instinctively, no learning required).
This module's `metrics.py::FunctionalConstraintTracker` implements the same
method, tracked separately per species for eval-weight sites vs.
action-weight sites.

They also found a second, unsettling phenomenon not in Hinton & Nowlan's
original (1987) theoretical version of the Baldwin Effect: **shielding**.
When an innate ability (e.g. instinctive predator-avoidance) is
survival-critical enough that agents must be born with it, the corresponding
*evaluation*-network genes for that domain stop mattering for fitness and can
drift freely -- to the point some agents genuinely evolved to prefer the
sight of danger, while remaining fit because their action network avoided it
reflexively regardless. Worth watching for if this module's danger-sensing
channels (prey's predator-detection) ever show the same pattern.

## What's reused vs. adapted from the original paper

**Reused as-is:** the ERL mechanism itself (two networks, genome/live-network
split, `R_t = E_t - E_{t-1}`, GA reproduction), and the functional-constraint
detection method.

**Adapted, documented here so it isn't mistaken for the original:**
- **World:** this module reuses PredPreyGrass's simpler predator-prey-grass
  ecology (energy-based survival/reproduction, no separate carnivores, trees,
  or walls) rather than Ackley & Littman's exact World AL. Their world is
  richer; this project's ecology is already validated elsewhere in this
  repo, and the point of this module is to test the ERL *mechanism*, not
  reproduce their world byte-for-byte.
- **Observation:** 9 features -- nearest food/danger in each of 4 compass
  directions (closer = larger value) plus own normalized energy -- matching
  their *design principle* (small, semantically pre-processed input suited to
  a single-layer network) rather than their exact visual-appearance encoding.
- **Action network output:** a softmax categorical policy (5 actions:
  stay/N/S/E/W) rather than their specific 2-bit stochastic-threshold
  encoding. Functionally equivalent (stochastic, genome/learning-influenced
  action selection); simpler to implement and verify correctly.
- **Learning rule:** `networks.py::reinforce_update` is a standard REINFORCE
  policy-gradient step with separate positive/negative learning rates
  (mirroring their `eta_+`/`eta_-`), not a bit-for-bit replica of their CRBP
  backprop-through-stochastic-threshold algorithm (Figure 3 of the paper).
  Same complementary-reinforcement logic (reward increases the taken action's
  probability, punishment decreases it); different implementation.
- **Genome encoding:** real-valued weight vectors with Gaussian mutation,
  not their redundant 4-bit-per-weight bit-string encoding (a specific 1991
  error-correction choice, not essential to the scientific claim).

**Why this counts as testing the same hypothesis despite the adaptations:**
none of the deviations touch the two things actually being tested here --
individually-owned, genome-initialized networks, and genome/live-network
separation preventing Lamarckian inheritance. Those are reproduced exactly.

## No RLlib, no PPO, no GPU

Deliberately: each agent's network is tiny (single-layer, ~50-90 weights) and
learns locally with plain NumPy, not backprop-through-time or a centralized
PPO training loop. `run_erl_simulation.py` runs as a plain Python script --
no Ray, no gymnasium multi-agent API, no GPU. Roughly 240 steps/sec observed
on this machine (single-threaded); a run of the length Ackley & Littman used
for their clearest genetic-assimilation results (~1-9 million steps) is
estimated at 1-10 hours, not the multi-GPU-hour cost of a PPO pilot.

## What to watch for

- `{species}_action_site_change_rate` becoming lower than
  `{species}_eval_site_change_rate` over generations (action genes becoming
  *more* constrained than eval genes) is the direct genetic-assimilation
  signature -- the thing Trials 1-9 could never measure this way, since they
  have no per-agent genome-initialized network for a "site" to even mean
  anything.
- `{species}_eval_weight_absmean` / `action_weight_absmean` drifting from the
  founder distribution is the coarser, population-mean-level signal (same
  category as everything tried so far, kept for comparison).
- Population survival itself: per Ackley & Littman, most initial random
  populations die out quickly; a handful survive far longer. Extinction on a
  short run (see `RESULTS.md`) is expected, not a bug -- multiple
  seeds/longer runs are needed before reading anything into it.
