# Darwin/Baldwin Search — Cross-Module Trial Log

Cross-module trial log for the search described in **[README.md](README.md)** — see
that file for the goal, the three success criteria, and the module catalog. This file
tracks the sequence of attempts against that goal: what was tried, why each pivot
happened, and the current state of the search. Each module below (`eco_evolutionary_*`)
also has its own detailed RESULTS.md with full data. Read top to bottom.

## Where things stand (2026-08-27)

One thing is established with real statistical power: **nature and nurture combined
beats either alone.** Trial 11 (`eco_evolutionary_erl_baldwin`) found ERL agents (genome
+ individual lifetime learning) significantly outsurvive evolution-alone, learning-alone,
and no-adaptation controls (p<0.00001, n=100/condition) — the strongest result in this
project's history, and the one still standing.

Everything built on top of or alongside that result has come back null or unresolved:

- **Single-continuous-scalar traits** (`metabolic_rate`, `offspring_investment_fraction`,
  `cooperation_rate` — Trials 2/3/5/6): real fitness landscapes exist, but no
  selection-driven drift beyond neutral noise in properly-replicated tests. One partial
  exception: Trial 6's population-scaling pilot showed a directional (but n=3-ceiling)
  signal for prey investment — inconclusive, not followed up.
- **Combinatorial genome** (`loci`, Trial 7): also null, reversed on the headline metric.
- **Dual-inheritance / cultural learning** (Trials 8-9): flat so far, both the
  static-coordination and seasonal-target versions.
- **Positive control** (Trial 10): confirms the pipeline *can* detect a real signal given
  an extreme (16x) gradient — population size, not mutation rate, is the likelier reason
  subtler traits above show nothing.
- **Cooperation, kin selection, and alarm-call communication** (built on top of Trial
  11's ERL architecture — see
  `predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/RESULTS.md` §10-16): all
  three closed as dead ends, each for a specific, understood reason, not an ambiguous
  null.
- **Genetic-assimilation timing**
  (`predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/RESULTS.md` §15): the
  paper's specific crossover signature wasn't found; genuinely open whether that's a
  real absence or a metric problem.

Read the trial-by-trial log below for how each conclusion was reached.

---

## Trial 1 — `eco_evolutionary_cadence` — rejected

**Trait:** movement-frequency ("speed") genome — an agent only gets a real move on roughly
1-in-6 steps depending on its evolved cadence value.

**Result:** structurally prevents predators from sustaining a population regardless of policy
quality — confirmed directly, predators went extinct in 30/30 sampled seeds under a trained
policy. The speed-as-movement-frequency mechanic itself was the problem, not tunable away.

**Verdict:** abandoned. Moved to `eco_evolutionary_metabolic_rate`, which already had partial
documented evidence of a working loop, as the more promising base.

---

## Trial 2 — `eco_evolutionary_investment` (R1-R3) — early signal, paused on an unrelated bug

**Trait:** `offspring_investment_fraction` — the share of a parent's energy transferred to each
offspring at birth.

**Result (R1, 59 iterations):** clean Baldwinian (RL) learning, and directional genome drift in
both species (predator −0.0055, prey −0.0061) self-reported as "confirmed" — but **never checked
against a neutral-drift control**. R2 and R3 (checkpoint-resume runs) revealed a severe,
unrelated engineering problem: the trained policy collapses catastrophically when resumed under
new random seeds, because training never encountered cold-start population states.

**Verdict:** paused, not concluded. The genome-drift claim is unverified by the standards later
established in Trial 3 (see below) — it's exactly the kind of premature "confirmed" read that
turned out to be noise for `metabolic_rate`. Mitigations were drafted (randomized initial
population, entropy warmup on resume) but never carried through to a full run. See
`eco_evolutionary_investment/RESULTS.md`.

---

## Trial 3 — `eco_evolutionary_metabolic_rate` (Iterations 0-6) — methodology built here, null result

**Trait:** `metabolic_rate` — sub-linear energy gain (`food^alpha`) vs. linear energy cost,
creating a policy-dependent interior optimum.

This is where the actual rigorous methodology got built, iteration by iteration:
- **Iterations 0-1:** baseline crashes constantly; a population-ratio reproduction cap fixes
  sustainability but isn't biologically motivated (no individual can sense a population ratio)
  and dilutes the selection signal — rejected despite better raw numbers.
- **Iteration 2-3:** individual-level satiation throttle (cooldown + per-catch energy cap, a
  Holling-type handling-time mechanism) — biologically grounded, sustainability much improved,
  and produces a real-looking predator genome cycle, replicated (with noisy magnitude) across two
  seeds.
- **Iteration 4:** neutral-drift control introduced — same config, genome inheritance severed
  from reproductive success — to test whether that drift exceeds what pure mutation +
  finite-population sampling produces on its own. Single-run result: ambiguous for predator,
  encouraging for prey.
- **Iteration 5:** the "encouraging" single-run prey read doesn't survive a proper 3-seed-each
  real-vs-control replication (Mann-Whitney U) — **null for both species**, direction doesn't
  even consistently favor real over control.
- **Iteration 6:** sharpened the fitness gradient (`metabolic_rate_alpha` 0.7 → 0.4) to test
  whether the signal was real-but-weak. Still null. Pulled the individual-level
  `mr_repro_spearman` metric (a more direct test than population-mean drift) — also flat and
  indistinguishable between real and control.

**Verdict:** null for criterion 3 (selection-driven drift), specifically for this trait's
implementation. Criteria 1 and 2 (sustainability, coexistence) are solved and unaffected. Not
proof no trait could show the loop here — but two independent lines of evidence (population-mean
drift, individual-level correlation) across two fitness-gradient steepnesses both came back flat,
which points at the trait's fitness leverage being too indirect rather than a tuning problem.
Full detail, data, and statistics in `eco_evolutionary_metabolic_rate/RESULTS.md`.

---

## Trial 4 — `eco_evolutionary_investment`, resumed (R4+) — in progress

**Why here:** `offspring_investment_fraction`'s R1 showed a *bigger* raw drift signal in far
fewer iterations (59) than `metabolic_rate` ever did, and — unlike `metabolic_rate` — it was
never actually tested rigorously; it was abandoned for an unrelated bug, not disproven. It also
plausibly has more direct fitness leverage (investment directly affects offspring survival odds
in one step, vs. `metabolic_rate`'s multi-step indirect energy-accounting chain).

**Numbering:** this trial continues `eco_evolutionary_investment`'s own R-numbering (R1-R3 are
the original 2026-06-27 runs; R4 onward is the resumed work), one R-number per distinct
trial/run/config — same flat convention as `metabolic_rate`'s "Iteration N". See
`eco_evolutionary_investment/RESULTS.md` for the live, detailed version of everything below.

**Plan and progress:**
1. Port the satiation-throttle sustainability fix from `metabolic_rate`. **Done** — validated by
   R4 (100-iter pilot, inconclusive) and R5 (400-iter pilot, confirmed: predator:prey ratio
   oscillates in a healthy band rather than climbing toward collapse).
2. **Test the reverse leg early and cheap:** freeze the genome at several fixed values
   (`genome_enabled: False`, varying `founder_genome` mean), no new instrumentation needed, and
   check whether fitness outcomes vary across values at all. This is the check `metabolic_rate`
   skipped from Iteration 0 onward (its own original next-steps list named it "Priority 1" and it
   was never done, only ever approximated by the `mr_repro_spearman` proxy — which turned out to
   carry no signal anyway). If outcomes are flat here too, stop before spending the compute on a
   full replication. **R6, in progress** (5 fixed values × 100 iterations).
3. If R6 shows a real gradient (→ R7, planned): port the neutral-drift control setup, run the
   same 3-seed real-vs-control replication that just ran for `metabolic_rate`.
4. If that looks promising (→ R8, planned): the expensive step — train under different frozen
   genome regimes and
   compare *learned behavior*, not just outcomes, for the strongest version of the reverse-leg
   claim.

**Status:** R4-R7 complete. R4/R5: throttle validated. R6: fitness outcomes are not flat across
fixed `offspring_investment_fraction` values — a real landscape exists. **R7 (the actual
selection test): null** — real vs. neutral-control drift magnitude statistically
indistinguishable for both species (Mann-Whitney U ≈ chance midpoint, p=0.5-0.65), same pattern
as `metabolic_rate`. R8 (contingent on a real R7 signal) does not apply. See
`eco_evolutionary_investment/RESULTS.md` for full data.

## Where this leaves the search

Two independent traits (`metabolic_rate`, `offspring_investment_fraction`), two independent
properly-powered multi-seed replications (Mann-Whitney U, real vs. neutral-drift control), both
null on criterion 3 (selection-driven genome drift beyond neutral noise). Criteria 1
(sustainability) and 2 (coexistence) are solved for both traits and unaffected by this — the
loop's forward-and-back mechanics work, the population dynamics are stable and realistic, but
neither trait shows detectable selection acting on it.

Notably, `offspring_investment_fraction` was the more promising candidate going in — a bigger
raw drift signal in far fewer iterations than `metabolic_rate` (R1, pre-rigor), a more direct
fitness mechanism (one-step energy transfer vs. `metabolic_rate`'s multi-step accounting chain),
and R6 confirmed real fitness leverage exists. It still came back null on the actual selection
test. That two differently-mechanised traits both show a real-fitness-landscape-but-no-detectable-
selection pattern is more informative than either result alone — it points at something more
structural than "wrong trait," which is exactly what the Hinton & Nowlan theoretical note above
anticipated: a single continuous scalar trait with a smooth fitness landscape may not be the
right shape of problem for the Baldwin effect to produce a detectable signal, almost regardless
of which scalar is chosen — and PPO's population-level policy optimization is a looser fit to
the classical individual-lifetime-learning mechanism than assumed going in.

**Next step is a design decision, not another replication of the same shape:** either (a) scale
population size to shrink the neutral-drift noise floor (previously deprioritized for
`metabolic_rate`, now worth reconsidering given two traits agree), or (b) attempt the
combinatorial/multi-gene trait design sketched in the theoretical note — a bigger pivot, but one
now supported by two converging null results rather than one. Not scoped or started; needs its
own discussion before committing more compute.

---

## Trial 5 — `eco_evolutionary_cooperation` — paused after Pilot 1 (likely null)

**Trait:** `cooperation_rate` — fraction of *this step's* catch/graze energy donated to
same-species neighbors within `cooperation_range` (meal-sharing, not a tax on standing
energy). Founder mean is 0.0 (identical to no-genome baseline), so any positive drift is a
direct selection signal.

**Why here:** unlike `metabolic_rate` and `offspring_investment_fraction` (both single
scalars with a smooth interior optimum reachable by ordinary selection alone — see the
Hinton & Nowlan note below), `cooperation_rate`'s payoff structure depends on an emergent,
policy-dependent quantity: local relatedness, set by RL-learned dispersal behavior
(offspring spawn adjacent to parent → population viscosity → kin-biased donation without
explicit kin recognition). This is also a direct empirical test of the literature link
between the Baldwin effect and cooperation (Suzuki & Arita; Taylor 1992's kin-competition
cancellation result), which motivated this module in the first place. It's a cleaner,
single-mechanism instrument for the same question an earlier, richer environment
(`stag_hunt_forward_view_nature_nurture`) left unresolved — that module got a solid
*behavioral* (nurture-side) cooperation result, but its heritable `coop_trait` (nature side)
was wired in and never actually evaluated (the nature-weight ablation script was written but
never run).

**Design note (documented in `eco_evolutionary_cooperation/README.md`):** `cooperation_rate`
is currently invisible to the policy's observation space (own or neighbors') — donation is
a mechanical multiplier, not a learned action. This means the current design tests only
**unconditional, viscosity-based kin selection**, not a plasticity/reputation-based
conditional-cooperation mechanism. A follow-up making a neighbor's trait observable would
open a second pathway — trait-based assortment / the **green-beard effect** (Dawkins;
Riolo, Cohen & Axelrod 2001) — but that's scoped as a separate variant, not a change to this
run, to keep this one a clean single-mechanism test.

**Status:** Pilot 1 complete — a real run and a `genome_neutral_drift_control` run (250
iterations each, single seed, GPU config), the latter added 2026-07-17 using the same
neutral-control mechanism ported from `investment`. **Preliminary result: likely null.**
The control drifted as much as or more than the real run in both species (predator: real
↓ to 0.010 vs. control ↑ to 0.023; prey: real ↑ to 0.023 vs. control ↑ to 0.035, an even
bigger move) — the same red-flag pattern that turned out to be noise for `metabolic_rate`
and `investment` before proper replication. Not yet confirmed: this is a single-seed
pilot, not the 3-seed Mann-Whitney replication those traits were held to, and the
`local_relatedness_proxy`/`coop_repro_spearman` metrics needed to distinguish "no
selection pressure" from "Taylor's-cancellation-cancelled selection pressure" are not yet
implemented. See `eco_evolutionary_cooperation/RESULTS.md` for full detail.

**Decision (2026-07-18): paused, not replicated further.** With `metabolic_rate` and
`investment` both already confirmed null via proper 3-seed replication, a full
replication of `cooperation`'s pilot would most likely just be a third data point
confirming a pattern already reasonably well established, not new information — see
"Where this leaves the search" below. Compute is better spent on the structural
decision that pattern points to (Trial 6) than on replicating a third single-scalar
trait. The missing-metrics instrumentation work is still worth doing at some point
(cheap, and needed for interpretability if this module is revisited), but is not
currently prioritized.

---

## Where this leaves the search, updated after Trial 5

Three single-continuous-scalar traits (`metabolic_rate`, `offspring_investment_fraction`,
now `cooperation_rate`) have each shown sustainability/coexistence solved but no
detectable selection-driven drift — two confirmed by proper replication, the third
(`cooperation_rate`) only at pilot/preliminary strength but pointing the same way. Three
different fitness mechanisms (energy-accounting asymmetry, one-step offspring transfer,
emergent-relatedness-mediated donation) converging on the same pattern is a stronger
signal than any one result alone that the shared property — *smooth, continuous, single-
scalar fitness landscape* — is the structural issue, not the specific trait chosen. See
the Hinton & Nowlan note below.

**Decision: pursue (a) before (b).** Two directions were on the table — (a) scale
population size to shrink the neutral-drift noise floor, or (b) the combinatorial/
multi-gene trait design the Hinton & Nowlan paper actually demonstrates. Going with (a)
first: it's a config-level change to an already-validated trait/pipeline, not a new
mechanism design, so it's cheap to falsify or confirm before committing to (b)'s much
larger scoping effort. If (a) still comes back null at larger scale, that rules out
"just noise" more convincingly and makes the case for (b) much stronger.

## Trial 6 — population scaling on `offspring_investment_fraction` — complete, mixed/inconclusive

**Why `investment`, not `metabolic_rate` or `cooperation`:** R6 already confirmed a real
fitness landscape exists for `offspring_investment_fraction` (fitness outcomes are not
flat across fixed values) — of the three traits tried, it's the one where "a real signal
exists but selection can't detect it at this population scale" is the most plausible
reading of the null R7 result, rather than "there is no signal to detect." Re-testing at
larger population scale validates directly against a trait already known to have real
fitness leverage.

**Plan:** increase population size (~2x grid/agent scale — an initial 4x attempt was
abandoned as too expensive, ~2.75 days/seed and GPU memory at the edge) and re-run the
same 3-seed real-vs-neutral-control replication methodology used for R7. Executed as R9
in `eco_evolutionary_investment` — see that module's RESULTS.md for full config, timing,
and per-seed data.

**Result:** species-asymmetric. **Predator: still null**, direction if anything reversed
from R7 (real final |dev| 0.0187 vs. control 0.0290, p=0.900). **Prey: directionally
positive, at the maximum separation a 3-vs-3 Mann-Whitney U can show** — all three real
seeds' final deviation from founder exceed all three control seeds' (U=9, p=0.050,
mean real=0.0233 vs. control=0.0084) — a materially different outcome than R7's flat
unscaled prey result (real=0.0397 vs. control=0.0351, p=0.500). p=0.050 at n=3 is exactly
the statistical floor of this design, not a result that clears conventional significance
with room to spare.

**Verdict:** first data point in the whole search pointing toward "a real selection
signal exists but was below the unscaled noise floor" — but only for one species, at the
edge of what n=3 can demonstrate, and not the clean both-species confirmation that would
validate the noise-floor hypothesis outright. Neither a clean win (predator gives no
support, and prey hasn't cleared the n=3 ceiling) nor a third flat null (prey's separation
is the strongest directional result seen across three traits and two scales so far).
**Genuinely inconclusive — the honest next step is more prey-focused seeds (e.g. 45/46/47)
to see if the separation holds past n=3, not a verdict either way on criterion 3.** Full
data, per-seed table, and timing in `eco_evolutionary_investment/RESULTS.md` R9.

**Status:** R9 complete (2026-07-24). Decision on next step (extend prey replication vs.
proceed to the combinatorial-trait pivot below) not yet made.

---

## Trial 7 — `eco_evolutionary_metabolic_code` — complete, null (reversed on headline metric)

**Trait:** `loci` — a length-10 combinatorial genome, each locus CORRECT/WRONG/
PLASTIC relative to an implicit fixed target (Hinton & Nowlan, 1987
needle-in-haystack design), rather than another smooth continuous scalar. An
agent can only achieve a full match if it carries zero WRONG loci (permanently
unfixable within a lifetime); PLASTIC loci are searched fresh every step
within that agent's own lifetime via a joint guess across all of them at once.
A solved genome multiplies energy gain from that step onward.

**Why this trait, why now:** Trial 6 was the last data point on the
single-continuous-scalar line of attack — inconclusive rather than a clean
confirmation (see above), and not worth pursuing further before trying the
structurally different design this file's theoretical note (below) already
pointed at. Per that note, the Baldwin effect specifically needs a landscape
"hard to search without an adaptive process to restructure the space" — a
smooth 1-D scalar, which is what all three traits tried so far (`metabolic_rate`,
`offspring_investment_fraction`, `cooperation_rate`) are, doesn't qualify.
This trait is shaped like the one the source paper actually demonstrates the
effect with.

**What's new here relative to every prior module:** a genuine **per-individual
lifetime search**, decoupled from the shared PPO policy. PPO continues to
learn movement/hunting/foraging behavior exactly as before; a separate, cheap,
per-agent stochastic process (one joint guess per step over that agent's
unresolved loci) resolves this trait within that same individual's lifetime.
This directly addresses structural gap #2 in the theoretical note below (PPO
being population-level optimization, not individual-lifetime search) — for
this one trait, not as a change to how PPO itself works elsewhere.

**Implementation:** its own directory (`eco_evolutionary_metabolic_code/`),
smoke-tested and covered by a 32-test unit suite (genome sampling/mutation,
the per-step solving mechanism, fixed-genome-independent offspring investment,
neutral-drift-control template selection, live/episode metrics builders)
before any real run. No separate throwaway pilot was run — a real seed
(42, real+control) was launched directly and monitored through its early
iterations instead, on the reasoning that a dedicated pilot brings nothing a
real run's early iterations don't also show, while staying usable data if
healthy rather than being discarded.

**Result: 3-seed real-vs-control replication (42/43/44), same Mann-Whitney
methodology as `investment`'s R7 — null, and reversed on the headline metric,
in both species.**

| species | metric | real (n=3) | control (n=3) | p |
|---|---|---|---|---|
| predator | mean_wrong_loci | 3.12 | 2.92 | p(real<control)=0.900 |
| predator | fraction_solved | 0.0080 | 0.0070 | p(real>control)=0.350 |
| prey | mean_wrong_loci | 3.92 | 3.27 | p(real<control)=0.800 |
| prey | fraction_solved | 0.0087 | 0.0059 | p(real>control)=0.500 |

Both species show real *higher* than control on the headline metric
(mean WRONG-loci count) — the opposite of the hypothesized direction, since
selection should push this down, not up. The secondary metric
(fraction_solved) trends the predicted direction in both species but far too
weakly to matter at n=3. One data-quality caveat: seed 44's prey values are a
clear outlier in both groups simultaneously (real and control both ~4-5.6 vs.
~2.7-4.0 for every other seed/species), and that same seed also showed
meaningfully more predator-reproduction capacity-blocking than seeds 42/43 —
both point at seed 44 having produced atypical population dynamics that add
noise to the n=3 aggregate, not at a mechanism bug. Full data, per-seed table,
and timing in `eco_evolutionary_metabolic_code/RESULTS.md`.

**Verdict:** the fourth trait design in this project to fail this test, and
the cleanest null yet in one sense — unlike Trial 6's species-split result,
both predator and prey agree here. Purpose-built to fix both gaps the
theoretical note below identifies (a true needle-in-a-haystack landscape, and
a genuine per-individual lifetime search independent of PPO), and it still
didn't produce a detectable selection signal beyond neutral drift at this
scale. Not proof the combinatorial-genome category can't work — the founder/
mutation/bonus parameters were calculated starting guesses never tuned
against real data, and the seed-44 outlier suggests more noise than ideal in
this particular pass — but it's a real, disappointing data point, not
grounds to declare victory prematurely.

**Status:** replication complete (2026-07-26). Decision on next step (more
seeds, parameter retuning, or stepping back to reconsider the search
direction) not yet made.

---

## Trial 8 — `eco_evolutionary_cultural_plasticity` — stopped early, likely null

**Mechanism, not just another trait.** Trials 1-7 all share one structural
property regardless of trait shape: a single heritable channel feeding a
shared per-species PPO policy. This trial adds a **second, independent
inheritance channel** instead — gene-culture coevolution / dual inheritance
(Boyd & Richerson, 1985; Cavalli-Sforza & Feldman, 1981), not a variant of
the same single-channel design tried seven times so far.

**The two channels:** `dialect` (categorical, one of 4 arbitrary same-species
coordination codes) is a *live, mutable* per-agent state — seeded from a
heritable founder value at birth, but free to change many times within one
agent's own lifetime via social learning from neighbors within
`culture_range`. `plasticity` (continuous, `[0,1]`, standard Gaussian-mutation
genome trait) is the actual gene under test: it sets how readily an agent's
live dialect updates toward the local same-species majority. The gene does
not encode behavior directly — it encodes capacity to adopt culture, which is
the literal Baldwin/dual-inheritance mechanism.

**Why this targets the diagnosis in the theoretical note below more directly
than Trial 7 did.** A catch/graze event grants an energy bonus when an
agent's live dialect matches its local majority at that moment — a
coordination game, not a gradient. Unlike Trial 7's combinatorial
needle-in-a-haystack (still null, both species, reversed on the headline
metric), which fixes gap #1 (a rugged landscape) but only partially addresses
gap #2 (genuine individual-lifetime search), this design's social-learning
update is frequency-dependent and history-dependent by construction, and the
gene being tested (`plasticity`) has direct, continuously-measurable leverage
over how fast that individual-level process runs — a tighter version of gap
#2 than Trial 7's per-step joint-guess mechanism, which the genome could not
influence at all (only zero-vs-nonzero WRONG-loci viability gated it).

**Reverse leg, addressed directly instead of left unconfirmed.** Every prior
module kept the genome invisible to the policy's observation space. This
module deliberately breaks that convention (`include_culture_in_obs`): the
policy observes its own live dialect and local-majority-match status, so PPO
can learn to condition movement/hunting on cultural state — a plausible route
to the "genome/culture shift feeds back into learned behavior" leg no prior
trial confirmed.

**What's ported unchanged:** satiation-throttle sustainability mechanism,
reproduction/energy dynamics, and the `genome_neutral_drift_control` /
Mann-Whitney multi-seed replication methodology — all directly from
`eco_evolutionary_metabolic_code`. `plasticity` uses the |deviation from
founder| test (à la `metabolic_rate`/`investment`), not Trial 7's directional
one, since plasticity has no a-priori predicted drift direction.

**Status:** implemented, 27 unit tests passing, 300-step random-policy smoke
run clean. A 300-iteration single-seed pilot (seed=1) confirmed
sustainability/coexistence under real PPO training and a genuinely active
cultural-learning mechanism (`dialect_match_rate` up to 0.82, far above the
0.25 chance baseline). The full replication was launched but **stopped
early, after all 3 real seeds finished and before any neutral-control
seed**: `plasticity_mean` stayed within roughly one founder-std of its 0.1
starting value in all 3 real seeds with no consistent direction, and
individual-level plasticity-vs-reproduction correlation was essentially
zero in every seed, both species — the same flat, no-fitness-correlation
pattern as Trials 1-7, despite this trial's dual-inheritance,
frequency-dependent design being specifically intended to route around
that failure mode. No control-seed data was collected, so this isn't a
Mann-Whitney-confirmed null, but the descriptive real-seed signal was
judged too weak to justify the remaining ~21h of control-seed compute.
See `eco_evolutionary_cultural_plasticity/RESULTS.md` (§3-4) for the full
real-seed numbers, the stop rationale, and candidate next steps.

---

## Trial 9 — `eco_evolutionary_cultural_plasticity_seasonal` — replication in progress

**Targets the actual diagnosis behind Trial 8's null result, not just the trait shape again.**
Trial 8's postmortem pointed to Rogers' Paradox (Rogers, 1988): a gene for social-vs-individual
learning has no fitness advantage in a *static* environment, because Trial 8's coordination bonus
rewards matching the *local majority* dialect — a self-referential game with no external
"correct answer" that ever changes. A separate exploration this session
(`base_environment_seasonal`, a 6-regime resource-abundance sweep on the plain, non-evolutionary
base env) confirmed a simple on/off timer keyed on the env's own per-episode step counter works
cleanly, but scaling *how much* food exists doesn't test the right thing for Rogers' Paradox
either — what's needed is a change to *which behavior is correct*.

**The change, on top of Trial 8's dual-inheritance design (unchanged):** a new
`_current_target_dialect()` (same square-wave-on-`current_step` pattern as
`base_environment_seasonal`'s `_current_season_multiplier`, cycling through all `n_dialects`
every `dialect_season_length_steps` instead of high/low) replaces `_local_majority_dialect` as
what `_dialect_match_bonus` checks against. `_apply_cultural_learning` (how dialects spread via
copying) is untouched. After each flip, the population is briefly stuck matching the old target;
only fast adopters (high `plasticity`) recover the bonus quickly — a concrete, repeated
opportunity for `plasticity` to pay off that Trial 8's static coordination game never had.

**Status:** implemented, 29 unit tests passing (27 ported from Trial 8, 2 new for
`_current_target_dialect()`'s phase-cycling, and Trial 8's 2 coordination-bonus tests rewritten
as the regression guard for this module's one behavioral change). Smoke run, an extra short run,
a single-seed pilot, and the first real replication seed (42) have all completed
(2026-08-08, full 1000/1000 iterations). `plasticity_mean` and `plasticity_repro_spearman`
bounced with no consistent direction the whole run, the same flat shape as Trial 8 — not yet
a formal real-vs-control comparison (seeds 43/44 and the neutral-control replication not
started), but discouraging on this one seed. (Earlier progress notes here cited "833/1000
iterations" — that number came from a CSV-parsing bug on the reporting side, not an actual
pause; corrected.) See `eco_evolutionary_cultural_plasticity_seasonal/RESULTS.md` for full run
inventory.

---

## Trial 10 — `eco_evolutionary_metabolic_rate_positive_control` — pipeline sanity check, weak-but-real signal, mutation ruled out as bottleneck

**Not a new trait — a diagnostic.** After Trials 1-9, the project had never confirmed the
training pipeline (genome inheritance + mutation + differential reproduction + population-level
metric aggregation) could detect selection *at all*, independent of any trait's design. This
module clones Trial 3 (`eco_evolutionary_metabolic_rate`) and pushes `metabolic_rate_alpha`
from the parent's subtle sub-linear 0.4-0.7 range to a deliberately extreme super-linear 3.0 —
a 16x efficiency gap between the trait's bounds, an overwhelming advantage with no interior
optimum and no dependence on policy quality (unlike the parent module's design).

**Pilot 1** (mutation std=0.04, same as everything else): weak partial signal, not a clean
pass. `metabolic_rate_mean` climbed from 1.0 to only ~1.05-1.06 then plateaued (bound is 2.0).
`predator_mr_repro_spearman` (individual-level: does higher trait value actually predict
reproduction?) stayed consistently positive through the run's second half (last-quarter mean
+0.046) — real, but small. `prey_mr_repro_spearman` showed no direction. Rules out "the
pipeline can't detect selection at all" but also rules out "effect size alone was the
bottleneck," since a 16x gradient should dominate fast and cleanly if effect size were the only
constraint.

**Pilot 2** (mutation std lowered 0.04→0.01, alpha unchanged): isolated the mutation-rate
hypothesis directly. No improvement over Pilot 1 — predator's correlation was weaker (+0.021 vs.
+0.046), population mean did not climb further (1.041 vs. 1.054), std did not narrow. Mutation
rate is not the dominant bottleneck, at least across this range.

**Verdict:** population size, not mutation rate, is the better-supported explanation for why
Trials 1-9's much subtler trait effects never showed up — consistent with Trial 6's own earlier
finding (above) that scaling population produced the strongest directional signal seen anywhere
in this project, for a different trait. A population-size isolation pilot is the logical next
step for this module, not yet run. Also surfaced and partially fixed a real infrastructure bug
along the way: Ray Tune's CSV logger silently drops any metric key not present in the first
reported iteration's result, which had hidden `mr_repro_spearman` from `progress.csv` in this
module (recovered via the TensorBoard event file) and, checked retroactively, in Trial 3's
original real-seed runs too (though Trial 3's documented "flat" conclusion held up against the
recovered data — nothing to revise there). Full detail in
`eco_evolutionary_metabolic_rate_positive_control/RESULTS.md`.

---

## Trial 11 — `eco_evolutionary_erl_baldwin` — different architecture, not a new trait or a retune

**The one thing Trials 1-10 structurally cannot test.** Every trial so far uses a single PPO
policy shared across each species, with genome as a side-channel scalar that never touches the
policy itself. This module replaces that architecture entirely for a parallel experiment: each
agent gets its own genome-initialized network and its own lifetime of local reinforcement
learning, following **Ackley & Littman (1991), "Interactions Between Learning and Evolution,"**
*Artificial Life II* — the paper that coined "Evolutionary Reinforcement Learning" (ERL) and
computationally demonstrated the Baldwin Effect in almost exactly this architecture (predator-
prey-food agents, GA-evolved genome, individually-learning behavior). No RLlib, no PPO, no GPU —
plain Python/NumPy, ~240 steps/sec single-threaded, since each agent's network is a single
layer.

**Critical design property, explicitly tested:** reproduction copies an agent's *genome*
record, never its live, post-learning network weights — Darwinian, not Lamarckian, by
construction, not convention. Unit-tested directly
(`test_offspring_genome_does_not_inherit_parents_learned_weights`).

Detects genetic assimilation via **functional-constraint analysis** (tracking how much each
genome site's value survives mutation across generations) rather than the population-
mean-drift / individual-correlation approach used in every other trial — a method that doesn't
compete with the population-size noise floor Trial 10 diagnosed, since it measures whether
mutations at a site are purged by selection at all, at any population size.

**Status: the comparative-study claim is confirmed, with real statistical power.** The world
was rebuilt to match the paper's actual World AL mechanics (100×100 grid, carnivores as a
separate non-adaptive hard-coded species, trees/walls/corpses/health — not the simpler
predator-prey-grass ecology this project uses elsewhere), retuned once to fix a boom-bust
collapse, then run as a full 5-condition × 100-seed × 1,000,000-step comparative study,
matching the paper's own scale. Result: **ERL (nature + nurture combined) significantly beats
evolution alone, learning alone, neither, and pure luck — p<0.00001 against all four** — the
strongest, most statistically legitimate result in this project's entire trial history (1-11).
The internal structure substantially reproduces the paper's own findings too (learning-alone
significantly beats evolution-alone; no-adaptation is statistically indistinguishable from
luck). One honest discrepancy remains (evolution-alone beats luck here; the paper found the
reverse), and overall survival difficulty isn't calibrated to their reported rate (83% of ERL
runs reach the step ceiling here vs. their ~7%) — the *ranking* matches, the *absolute
difficulty* doesn't. This is a new, parallel avenue, not a replacement for the PPO-based trial
family — it tests a different architectural hypothesis than Trial 10 (which pointed at
population size for the *existing* shared-policy architecture): whether genome can causally
shape behavior at all via individually-owned, genome-initialized networks. It can, and
combining it with learning measurably outperforms either alone. Full detail, significance
tables, and what's still not attempted (the paper's deeper single-population longitudinal
genetic-assimilation study) in `eco_evolutionary_erl_baldwin/RESULTS.md` §9.

---

## Theoretical note — Hinton & Nowlan (1987), a candidate future trait direction

This motivated the Trial 7 pivot above (`eco_evolutionary_metabolic_code`) — recorded here in full since it's the theoretical basis for that module's design, not just a historical note anymore.

Hinton & Nowlan, "How Learning Can Guide Evolution" (1987) — the paper that formalized the
Baldwin effect computationally — offers a plausible theoretical account for why both traits
tried so far (`metabolic_rate`, `offspring_investment_fraction`, `cooperation_rate`) have shown
weak-to-null Darwin/Baldwin coupling, and a concrete direction for a trait design that might not.

**1. The Baldwin effect needs a "needle in a haystack," and none of the traits tried so far is one.**
The paper is explicit about its own limitation: *"The main limitation of the Baldwin effect is
that it is only effective in spaces that would be hard to search without an adaptive process to
restructure the space."* Their demonstration uses a combinatorial genome (20 genes × 3 alleles,
2²⁰ combinations) with a single narrow fitness spike that pure evolution can't find unassisted —
learning's role is to carve out a detectable "zone of increased fitness" around near-miss
genotypes. `metabolic_rate`, `offspring_investment_fraction`, and `cooperation_rate` are all
single continuous scalars with a smooth interior optimum (sub-linear gain vs. linear cost; an
investment tradeoff curve; a donation-rate tradeoff mediated by relatedness). Ordinary selection
can climb a smooth 1-D gradient without any help from learning — there is no haystack for
learning to rescue you from. By the paper's own logic, a strong, clearly measurable Baldwin
effect isn't expected in any of the three traits as designed.

**Plain-language version:** a smooth hill vs. a combination lock. On a smooth hill, wherever
you stand you can feel which direction is slightly better — a mutant that's a little closer to
the optimum is a little fitter, every generation, so blind mutation-and-selection climbs it
fine on its own. Add learning on top and you learn nothing new: evolution would have gotten
there anyway, so you can't tell learning's contribution apart from plain selection's. A
combination lock is different: with 10 dials, 9-out-of-10 correct pays off exactly as badly as
0-out-of-10 — there's no "getting warmer," so blind mutation-and-selection can wander forever
without a signal to climb. Individual lifetime learning changes that: an organism born with
9 correct dials can search nearby combinations within its own lifetime and often find the 10th,
while one born with only 3 correct can't search far enough to compensate. That converts "close
genotype" into "usually successful phenotype" — manufacturing a slope where genetically none
existed — and now evolution has something to climb: individuals close to the answer
out-reproduce individuals far away, generation after generation fixing a few more correct
dials, needing less learning each time (genetic assimilation). The Baldwin effect is this
learning-manufactures-a-gradient phenomenon specifically — it's only visible on a
needle-in-a-haystack landscape, not a smooth one, which is exactly what all three traits tried
so far lack.

**2. The paper's "learning" is individual-lifetime search; PPO here is population-level policy
optimization.** In the simulation, each of 1000 organisms performs its own random-search learning
trials within its own lifetime, and that individual's discovery determines that same
individual's fitness. In this project's environments, "learning" is a shared PPO policy trained
across the whole population over many iterations — an agent born late in training inherits an
already-mostly-trained policy; it doesn't independently search anything itself. This is a real
structural gap from the classical Baldwin effect, and plausibly part of why the reverse leg
specifically (genome shaping what gets learned) has been hard to detect — there isn't genuine
individual-lifetime search for the genome to interact with.

**Option (b), now executed as Trial 7 (`eco_evolutionary_metabolic_code`):** a trait design
closer to what the paper actually demonstrates — a combinatorial locus code with a narrow
joint fitness optimum (zero-WRONG-loci requirement) rather than a single smooth scalar, plus
a genuine per-individual lifetime search decoupled from PPO (addressing gap #2 above directly
for this one trait). Option (a) — population scaling on `investment` (Trial 6 / R9) — came back
inconclusive rather than cleanly null or cleanly confirmed (predator showed no signal at 2x
scale, prey showed the strongest possible directional real-vs-control separation for n=3, but
not enough to settle the question either way). Rather than spend more compute extending that
single-scalar line further, the decision was to move to option (b) — see the Trial 7 entry
above for its design and current status.
