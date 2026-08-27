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

## World AL rebuild (2026-08-09) -- what's reused vs. adapted

An earlier version of this module ran the ERL mechanism on top of this
project's own simpler predator-prey-grass ecology instead of Ackley &
Littman's actual World AL. After a comparative-study run (see RESULTS.md)
came out only partially consistent with the paper and left an open question
about whether that was scale or a world/mechanics difference, the world was
rebuilt to match their described mechanics directly, not this project's ecology.

**Reused/matched from the paper, including every exact number it actually
publishes:**
- 100×100 grid, non-toroidal (`grid_size=100`).
- **Two distinct populations**, not two adaptive ones: a single ADAPTIVE
  species (`Agent` -- genome + learning, omnivorous: eats plants, dead
  agents, dead carnivores) and a permanently NON-adaptive species
  (`Carnivore` -- no genome, no network, no learning, hard-coded "seek
  nearest visible agent" rule, *regardless of `strategy`*). This is a real
  structural correction from the earlier version, which had two adaptive
  species (predator+prey) -- the paper has exactly one experimental subject.
- Agents sense 4 cells in each compass direction, carnivores 6
  (`agent_sense_range=4`, `carnivore_sense_range=6` -- exact paper values).
- A new carnivore spawns every 200 steps (`carnivore_spawn_interval=200` --
  exact paper value, Figure 4).
- `min_plants=50` reseed floor (exact paper value).
- Trees (shelter, one occupant, carnivores can't climb or attack a sheltered
  agent), walls (permanent, damage on collision), corpses (persistent,
  partially edible over multiple bites, decay over time) -- all present per
  Figure 4/5, mechanics implemented as described.
- Action semantics exactly matching Figure 5's table: 4 directions (no
  "stay"), effect determined by target-cell contents (Enter / Eat all /
  Climb / Damage self / Damage other / Eat some), including that carnivores
  structurally cannot target a wall or occupied tree ("as programmed").
- Observation vector matches Figure 4's input panel: visual appearance in
  4 directions + in-tree binary + health + energy (`OBS_DIM=7`; the paper's
  explicit "bias" input unit is instead a standard network bias term --
  behaviorally equivalent, not an extra input feature).

**Still not the same, and can't be, because the paper doesn't say:** damage
amounts, energy thresholds, growth/birth/death probabilities, wall density,
and reproduction costs are never published as numbers -- only described
qualitatively ("minor damage", "geometric growth", "sufficiently
nourished"). Every such constant in `config.py` is my own chosen value,
clearly marked there. No amount of rebuilding recovers numbers the paper
never printed.

**Still a deliberate simplification, not yet revisited in this rebuild:**
the learning rule (`networks.py::reinforce_update`) is a standard REINFORCE
policy-gradient step, not their exact CRBP backprop-through-stochastic-
threshold algorithm (Figure 3) -- same complementary-reinforcement logic,
different implementation. Genome encoding is real-valued weights with
Gaussian mutation, not their redundant 4-bit-per-weight bit-string.

## No RLlib, no PPO, no GPU -- but slower than the earlier ecology

Each agent's network is still tiny (single-layer) and learns locally with
plain NumPy -- no Ray, no gymnasium multi-agent API, no GPU. But the richer
World AL mechanics (100×100 grid, carnivores, trees, walls, corpses) run
at roughly **30 steps/sec** single-threaded on this machine, down from the
~240 steps/sec the simpler ecology managed. A run to the paper's own
1,000,000-step comparative-study ceiling is now estimated at **~9 hours per
seed** that actually survives that long, not 1-10 hours as the earlier
(simpler-world) estimate said. Worth knowing before launching another
full-scale comparative study on this rebuilt world.

## What to watch for

- `action_site_change_rate` becoming lower than `eval_site_change_rate` over
  generations (action genes becoming *more* constrained than eval genes) is
  the direct genetic-assimilation signature -- the thing Trials 1-9 could
  never measure this way, since they have no per-agent genome-initialized
  network for a "site" to even mean anything.
- `eval_weight_absmean` / `action_weight_absmean` drifting from the founder
  distribution is the coarser, population-mean-level signal (same category
  as everything tried so far, kept for comparison).
- Population survival itself: per Ackley & Littman, most initial random
  populations die out quickly; a handful survive far longer. Extinction on a
  short run (see `RESULTS.md`) is expected, not a bug -- multiple
  seeds/longer runs are needed before reading anything into it.

## Cooperation (C / ERLC) -- a new question, not from Ackley & Littman

Houghton (2024), a commentary on Hinton & Nowlan (1987) -- the paper this
whole module's Baldwin-Effect framing traces back to -- proposes that
*cooperation*, not just learning, can guide evolution toward a complex
multi-gene target. His toy model: agents in fixed groups of four, a group
"fit" once its members collectively cover all 20 needed sub-traits (blind
to which member supplies which), with breeding biased toward fit-group
members. That mechanism doesn't port literally -- this world's genome is
continuous weights producing one composite survival behavior, with no
decomposable sub-traits the way his 20 binary genes are. What's adapted
instead: three competencies this world already produces as events
(foraging, carnivore evasion, reproduction) stand in for his sub-traits. A
local group (an agent + living agents within `cooperation_radius`) gets a
reproduction-energy-threshold discount once it has collectively
demonstrated all three within a recent window, by any member -- the same
group-blind-to-which-individual credit assignment, adapted from his
synchronous single-generation toy model to this world's continuous,
spatially-local, energy-gated reproduction.

Two new strategies, added alongside the original five without changing
them (`ErlWorld`'s docstring has the full mechanism and code pointers):
  - **"C"**: like "E" (evolution alone, no learning) plus the group-fitness
    breeding bonus. Isolates cooperation's marginal effect over evolution
    alone, the way "L" isolates learning's effect over "F".
  - **"ERLC"**: like "ERL", plus the same bonus. Tests whether cooperation
    adds anything on top of learning+evolution combined.

**Status:** mechanism implemented and unit-tested (`tests/test_cooperation.py`,
including a direct test of the Houghton-style credit assignment -- three
agents, each missing two of three competencies individually, register as a
fit group because *between* them all three are covered). Smoke-tested at
`grid_size=40` for population-scale behavior: the group-fitness check fired
in ~28% of evaluations, a real, non-trivial rate. **A pilot comparative
study and a follow-up sensitivity check (RESULTS.md §12-13) since found
this mechanism design is a dead end**: no detectable benefit at default
strength, and actively *worse* survival when strengthened -- most likely
because its reproduction-threshold-discount lever re-triggers the same
boom-bust failure mode (§7) the base world needed retuning to avoid. Not
pursued further as designed; see §13 for what a genuinely different
mechanism (one that doesn't touch reproduction thresholds) would need to
look like instead.

## Kin selection (K / ERLK) -- a second, independent cooperation question

Nowak's five mechanisms for the evolution of cooperation include both group
selection (what C/ERLC above approximates) and kin selection (Hamilton's
rule: an act that costs the actor is favored if it benefits a relative
enough, weighted by relatedness). This adds kin selection as a second,
separate mechanism -- not combined with C/ERLC, kept independently testable.

Rather than inventing new machinery, it reuses two things already in this
world: the agent-on-agent aggression branch in `_resolve_agent_action`
(an agent already deals `agent_attack_damage` to another agent it moves
onto), and the genome itself as a relatedness proxy -- `genome.
genome_similarity` is an RBF-kernel distance over each agent's behavioral
genes (eval + action weights), which correlates with true kinship because
agents mate locally (`mate_search_radius`) and reproduce via
crossover+mutation, without needing separate parent/lineage bookkeeping.
A new evolvable trait, `genome.kinship_sensitivity` (sigmoid-transformed),
lets the population itself evolve toward or away from kin-biased leniency,
rather than hard-coding a fixed discount -- under "K"/"ERLK", an attacker's
damage is discounted by `sigmoid(kinship_sensitivity) *
kinship_discount_cap * genome_similarity(attacker, victim)`.

Two new strategies, independent of C/ERLC and of each other's mechanism:
  - **"K"**: like "E" (evolution alone, no learning) plus the kinship
    discount.
  - **"ERLK"**: like "ERL", plus the same discount.

**Status:** mechanism implemented and unit-tested (`tests/test_kin_selection.py`,
11 tests -- including a direct check that near-identical genomes get the
full discount and very different ones get none, and a regression guard that
ERL/E/L/F/B/C/ERLC never enter the kinship code path at all). Smoke-tested
at `grid_size=100` (paper default) over 3,000 steps: surviving-population
pairwise genome-similarity ranged from 0.120 to 1.000 (mean 0.568) -- real
spread, not degenerate, confirming actual encounters in a live population
span the full spectrum from close-kin-large-discount to
unrelated-no-discount. **A pilot comparative study and a follow-up
sensitivity check (RESULTS.md §12-13) since found this mechanism design is
a dead end**: no detectable benefit at default strength, and actively
*worse* survival when strengthened -- most likely because reducing
aggression damage this broadly re-triggers the same reproduction/survival-
easing dynamic that caused the base world's boom-bust problem (§7) before
its retune. Not pursued further as designed.

## Communication / alarm calls (S / ERLS) -- a third mechanism, deliberately different lever

C/ERLC and K/ERLK both work by DISCOUNTING a survival/reproduction cost --
which is most likely why both hit the same boom-bust failure mode this
world already needed retuning to avoid. This third mechanism is built on a
structurally different lever: pure information, no discount on anything.

Directly modeled on Ackley & Littman's OWN 1994 follow-up to their 1991
paper -- "Altruism in the Evolution of Communication" (Artificial Life IV)
-- which extended World AL with evolved alarm/food signaling and found
predator-warning calls reliably evolve when predators significantly affect
survival and the signal can interfere with predator success, despite being
costly (it can attract the predator to the caller).

Mechanism: agents under "S"/"ERLS" get one extra observation input (line-
of-sight alarm signal, same blocking semantics as the existing visual
channels), fed by a new evolvable trait `genome.alarm_call_propensity`
(sigmoid-transformed) that determines the probability of calling when a
carnivore is nearby -- deliberately evolvable, not hard-coded, since
whether a costly signal is worth emitting at all is the actual question.
The cost is real: a calling agent becomes measurably more conspicuous to
carnivore targeting (`call_conspicuousness_multiplier`) for a few steps.
Critically, there is NO hard-coded benefit on the receiving end -- whether
the existing evolved eval network learns to treat the alarm as "danger"
and the existing learned/evolved action network learns to move away from
it is left entirely to the same machinery that drives every other
behavior. No new reflex, on purpose: otherwise this would just be a third
bespoke bonus, not a fair test of whether general learning+evolution can
exploit an information channel.

Two new strategies, independent of C/ERLC, K/ERLK, and each other:
  - **"S"**: like "E" (evolution alone, no learning) plus the alarm
    mechanism.
  - **"ERLS"**: like "ERL", plus the same mechanism.

**Status: closed as a documented dead end (RESULTS.md §16), but for a
different, better-supported reason than C/K.** A pilot (n=20, 300k steps)
found `S` statistically indistinguishable from `E` (p=1.0000 -- about as
clean a null as a test produces) and `ERLS` from `ERL` (p=0.36). A
follow-up long-budget diagnostic (n=8, the full 1,000,000-step ceiling,
mechanism parameters unchanged) ruled out "just needs more generations"
directly -- `ERLS`'s cap-reach rate trended *below* `ERL`'s even with 3x
the steps. Reading Ackley & Littman's actual 1994 paper in full (not just
secondary summaries) explains why: their own conclusion is that costly
signaling only evolves and stabilizes when the beneficiaries of a call are
disproportionately the caller's own kin -- otherwise "information
parasites" erode it. This design broadcasts to any nearby agent
uniformly, with no kin-bias at all -- the one ingredient their own theory
says is necessary is the one thing never built here. A kin-biased version
(restricting the signal's benefit toward genetically similar listeners,
reusing K/ERLK's `genome_similarity`) would be a structurally different
second attempt, not ruled out by anything found here -- just not built.
