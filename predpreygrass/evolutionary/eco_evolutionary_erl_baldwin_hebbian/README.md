# ERL Baldwin Hebbian — same question, a different learning algorithm

## Why this module exists

This is a fork of `eco_evolutionary_erl_baldwin` (see that module's README for
the full original motivation, world-rebuild history, and its own trial log --
not repeated here). That module's headline result: **ERL (per-agent,
genome-initialized networks + per-life reinforcement learning + GA-style
reproduction) significantly outperforms the prior shared-policy trial family**
-- the project's strongest confirmed result (p<0.00001). That result was
produced with one specific learning algorithm: a REINFORCE-style,
reward-modulated policy-gradient update on the live action network
(`reinforce_update`).

The open question this fork tests: **does the result depend on that specific
algorithm, or does it generalize to per-agent independent learning as a
category?** If a structurally different learning rule -- one that doesn't do
policy-gradient descent at all -- reproduces the same genetic-assimilation
signature using the exact same world, genome/reproduction mechanics, and
evaluation methodology as ERL Baldwin, that's much stronger evidence the
effect is about "genome-conditioned per-agent lifetime learning" in general,
not an artifact of one particular update rule. A null result here wouldn't
undo ERL Baldwin's own finding, but it would narrow the claim to that specific
algorithm rather than the broader category.

The algorithm substituted in: **reward-modulated Hebbian plasticity**, per
Miconi, Clune & Stanley (2018, "Differentiable Plasticity") for the
correlation-trace mechanism, and Miconi, Rawal, Clune & Stanley (2019,
"Backpropamine") for making that trace's update reward-modulated rather than
purely unsupervised -- see "Why reward-modulated, not plain Hebbian" below for
why the unsupervised version was explicitly rejected for this test. Everything
else -- world mechanics, genome/reproduction structure, the eval network's
role, the C/ERLC, K/ERLK, S/ERLS cooperation/kin-selection/communication
add-ons, the functional-constraint analysis methodology -- is unchanged from
`eco_evolutionary_erl_baldwin`; only `genome.py`'s action-network fields,
`networks.py`'s learning rule, and the corresponding `Agent`/`World` plumbing
in `world.py` differ. See that diff, not a rewrite, as the honest description
of what this module is.

## The mechanism

Each agent carries two single-layer networks:
- **Evaluation network** (`genome.eval_weights`, `eval_bias`): fixed for the
  agent's entire life. Maps observation -> a scalar "goodness" value. This
  *is* the genetically inherited goal -- never touched by learning. Unchanged
  from ERL Baldwin.
- **Action network** (`genome.action_weights`, `action_bias`): fixed BASE
  weights for the agent's entire life -- unlike ERL Baldwin, these are never
  modified in place during life. Alongside them, a NEW genome field,
  `genome.action_alpha_weights`/`action_alpha_bias` (same shape), scales how
  much a per-life **Hebbian trace** (`Agent.hebb_trace`/`hebb_bias_trace`,
  starts at zero at birth) can shift the network actually used to act. The
  effective network each step is `action_weights + action_alpha_weights *
  hebb_trace` (`networks.py::effective_action_weights`) -- base plus
  genome-scaled plasticity, not a live-updated copy of the base weights
  themselves.

Reinforcement signal (unchanged from ERL Baldwin): `R_t = E_t - E_{t-1}` --
literally "am I better off now than a moment ago, by my own inherited sense
of good." No externally supplied reward function. What changed is how that
signal is used: instead of a policy-gradient step applied directly to live
weights, it modulates a decaying per-life trace
(`networks.py::hebbian_trace_update`), currently (second revision -- see
"Diagnostic history" below for why):

```
credit = one_hot(prev_action, n_actions) - prev_action_probs
hebb_trace(t) = clip((1 - eta) * hebb_trace(t-1)
                      + eta * reinforcement * outer(obs, credit),
                      -trace_clip, trace_clip)
```

`obs` is the presynaptic input. `credit` is the SAME quantity ERL Baldwin's
`reinforce_update` calls `grad_logits` -- positive for the action actually
taken, negative for the others, scaled by how confident the policy already
was -- not a plain correlation against the whole output distribution. This
makes the trace update **action-specific**: it credits the action taken for
the reinforcement that followed, the same credit assignment REINFORCE uses,
just accumulated into a bounded, decaying, genome-scaled trace instead of
applied directly and unboundedly to live weights. `eta`/`trace_clip` are
config hyperparameters (`hebbian_eta`, `hebbian_trace_clip`), free choices
like ERL Baldwin's own `lr_positive`/`lr_negative` before them, not values
from the literature.

### Diagnostic history: why the update rule changed

The first version of this mechanism correlated `obs` against the full
`prev_action_probs` vector (plain reward-modulated Hebbian plasticity, per
Miconi et al.'s "Backpropamine" -- see the subsection below for why that
specific variant, not unmodulated Miconi 2018, was chosen first). A
calibration sweep found no `hebbian_eta`/`hebbian_trace_clip` combination
produced genuinely sustainable population dynamics -- low eta caused fast
extinction, higher eta merely delayed the same outcome (a 6000-step
extension of the best-looking run showed boom-then-bust: 60->628->240
agents, carnivores crashing to 1), and the "best" eta (0.3) failed on 2 of 3
seeds tried.

To find out why, a controlled diagnostic isolated one variable at a time:
a single immortal agent (huge health/energy, no reproduction), no
carnivore, no competition -- just its own lifetime trajectory. Both a
failing eta (0.1) and a nominally-surviving one (0.2) produced
*identical*-looking behavior: `argmax` action fixed on the same choice for
the entire 3000-step lifetime, entropy oscillating mildly, effective
weights barely moving from the founder baseline. Reintroducing a single
carnivore to evade didn't change this -- the agent's chosen action stayed
fixed regardless of the carnivore's distance (1 to 27 cells), "evasive"
only 2/29 sampled steps, for both etas.

The decisive comparison: the SAME founder genome, SAME agent/carnivore
placement, SAME seed, run instead under ERL Baldwin's original
`reinforce_update`. That mechanism visibly adapted -- `argmax` shifted N
(->W at step ~300) and stayed there, "evasive" 11/31 sampled steps
(~35%), and `|live weights - genome weights|` grew steadily and without
bound to 1.68 over 3000 steps, vs. the Hebbian trace's norm staying under
0.09 throughout. Scaling the plasticity coefficients (`action_alpha_*`) up
to 20x on the identical scenario barely moved the needle (0% -> 10% evasive,
non-monotonically) -- ruling out "alpha too small" as the explanation.

The actual difference: REINFORCE's update credits the SPECIFIC action
taken; the first Hebbian version credited the whole output distribution,
carrying no information about which action was actually responsible for
what happened next. That's a credit-assignment gap, not a scale problem --
confirmed by the fact that no amount of rescaling closed it. The current,
action-credited version above is the fix, and is no longer *plain* Hebbian
plasticity in Miconi et al.'s sense -- see "Why reward-modulated, not plain
unsupervised Hebbian" below for the related (but distinct) decision this
revision also preserves, and REFERENCES.md for where this now sits in the
literature (closer to reward-modulated/three-factor Hebbian learning, or
REINFORCE with an eligibility trace, than to Miconi's original formulation).

### Why reward-modulated, not plain unsupervised Hebbian

Miconi et al.'s original 2018 "Differentiable Plasticity" trace has no reward
term at all -- it's pure pre×post correlation, with plasticity coefficients
trained by backpropagation through many trials (a batch procedure this
project's per-agent lifetime loop has no equivalent of). Implementing that
version literally would make the trace update depend only on correlation, not
on the eval network's output at all -- which would make `genome.eval_weights`
**causally inert**: still inherited, mutated, and tracked in the stats, but no
longer connected to behavior or fitness through any code path. That would
quietly change the experiment from "does ERL's result hold with a different
learning algorithm" to "does an unsupervised, non-goal-directed plasticity
mechanism also show a selection signal" -- a different and much weaker
hypothesis, since it drops the one architectural feature (genome specifies a
goal, goal drives learning) that ERL Baldwin's own result is actually about.
Scaling the Hebbian update by `reinforcement` (closer to Miconi, Rawal, Clune
& Stanley's 2019 "Backpropamine" neuromodulated-plasticity follow-up) keeps
that causal chain intact, so this module stays a fair test of the actual
question it exists to answer. This was a deliberate choice, not the only
possible design -- a plain unsupervised variant would test a different,
narrower hypothesis (see above).

**Reproduction copies the genome record, never the live trace.**
Whatever a trace accumulated during an agent's life is discarded at
reproduction; only the genome (base weights, alpha, plus mutation/crossover)
is passed on. This is the whole reason it's Darwinian, not Lamarckian --
enforced the same way as ERL Baldwin, just via a different piece of state:
`world.py`'s `_handle_agent_reproduction` always copies from `.genome`, never
from an agent's live `.hebb_trace`, and
`tests/test_erl_baldwin.py::test_offspring_genome_does_not_inherit_parents_hebbian_trace`
asserts it directly (every child's trace starts at exactly zero regardless of
the parent's accumulated trace at the moment of reproduction).

The rest of this section is background from Ackley & Littman's original paper
(carried over from `eco_evolutionary_erl_baldwin`'s README) -- it describes
their result and the measurement methodology this fork reuses unchanged, not
a result this fork has produced itself (see "Status" at the end of this file).

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
method, tracked separately for eval-weight sites vs. action-weight sites --
in this fork, the "action" portion includes `action_alpha_weights`/
`action_alpha_bias` alongside the base weights (see `genome.py::Genome.flatten`),
since the plasticity coefficients are as much a part of the action network's
genome as the base weights are, and are exactly where a Hebbian-specific
assimilation signature (if one exists) would show up.

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
Littman's actual World AL. After a comparative-study run (see ../eco_evolutionary_erl_baldwin/RESULTS.md)
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

**Still a deliberate simplification, inherited from ERL Baldwin and not
revisited here:** genome encoding is real-valued weights with Gaussian
mutation, not the paper's redundant 4-bit-per-weight bit-string. The learning
rule itself is this fork's whole point and is no longer a REINFORCE
approximation of the paper's CRBP algorithm at all -- it's reward-modulated
Hebbian plasticity (`networks.py::hebbian_trace_update`), a different
algorithm family entirely, not a closer or looser approximation of Ackley &
Littman's Figure 3 mechanism in either direction. See "Why reward-modulated,
not plain Hebbian" above.

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
  generations (action genes -- base weights AND alpha -- becoming *more*
  constrained than eval genes) is the direct genetic-assimilation signature,
  same interpretation as ERL Baldwin.
- `eval_weight_absmean` / `action_weight_absmean` drifting from the founder
  distribution is the coarser, population-mean-level signal (same category
  as everything tried so far, kept for comparison).
- NEW in this fork, and the two numbers its own headline question is actually
  about: `action_alpha_absmean` (does the plasticity coefficient itself drift
  from its random founder scale under selection?) and `hebb_trace_absmean`
  (does the per-life trace it scales ever move meaningfully away from zero?
  If this stays ~0 across a whole run, the Hebbian channel never does
  anything -- Baldwin-relevant or not -- and any result would be
  uninterpretable regardless of which way it points).
- Population survival itself: per Ackley & Littman, most initial random
  populations die out quickly; a handful survive far longer. Extinction on a
  short run (see `RESULTS.md`) is expected, not a bug -- multiple
  seeds/longer runs are needed before reading anything into it.

## C/ERLC, K/ERLK, S/ERLS: inherited mechanisms, inherited (not re-verified) verdicts

The three sections below are carried over unchanged from `eco_evolutionary_erl_baldwin`
-- the code, tests, and "dead end" verdicts all describe runs made under
ERL's original reward-modulated policy-gradient learning rule, not this
fork's Hebbian plasticity. They have NOT been independently re-run here. The
reasons given for each dead end are about the world's reproduction-threshold
dynamics (C/K) and Ackley & Littman's own signaling theory (S), not about the
learning algorithm specifically, so there's no particular reason to expect
Hebbian plasticity to change any of these verdicts -- but "no particular
reason to expect a change" is not the same as "verified," and these three are
not part of this fork's own open question (see "Why this module exists").

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
study and a follow-up sensitivity check (eco_evolutionary_erl_baldwin/RESULTS.md §12-13) since found
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
sensitivity check (eco_evolutionary_erl_baldwin/RESULTS.md §12-13) since found this mechanism design is
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

**Status: closed as a documented dead end (eco_evolutionary_erl_baldwin/RESULTS.md §16), but for a
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

## Status (this fork specifically)

Built 2026-10-01 by forking `eco_evolutionary_erl_baldwin` and replacing its
learning rule as described above. Current state:
- `genome.py`, `networks.py`, `world.py`, `metrics.py`, `config.py`,
  `run_erl_simulation.py` updated for the new action-network/trace split.
- Full test suite ported and extended (92 tests passing): every existing
  ERL/E/L/F/B/C/ERLC/K/ERLK/S/ERLS test rewritten against `hebb_trace`
  instead of a live-updated `action_weights`, plus new tests specific to the
  Hebbian mechanism itself (`effective_action_weights`, `hebbian_trace_update`
  direction/clipping/zero-reinforcement behavior, zero-alpha inertness).
- Smoke-tested: all 11 strategies run without crashing at small scale;
  checkpoint save/load round-trips correctly; a 1500-step run at the paper's
  default 100x100 config showed the population surviving with fluctuation and
  `hebb_trace_absmean` moving measurably away from zero under "ERL" (the
  trace is actually doing something, not dead code).
- **Not yet run at the scale needed to answer this module's own question.**
  No comparative study (ERL-Hebbian vs. E/L/F/B, or vs. ERL Baldwin's own
  numbers) has been launched. `hebbian_eta`/`hebbian_trace_clip`/the alpha
  init scale are untuned first guesses. Nothing in this file above the
  inherited-history sections describes a result yet -- only a working,
  tested mechanism ready to be run.
