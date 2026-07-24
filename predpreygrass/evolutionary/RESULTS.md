# Darwin/Baldwin Search — Cross-Module Trial Log

Cross-module trial log for the search described in **[README.md](README.md)** — see
that file for the goal, the three success criteria, and the module catalog. This file
tracks the sequence of attempts against that goal: what was tried, why each pivot
happened, and the current state of the search. Each module below (`eco_evolutionary_*`)
also has its own detailed RESULTS.md with full data. Read top to bottom.

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

## Trial 6 — population scaling on `offspring_investment_fraction` — planned, not started

**Why `investment`, not `metabolic_rate` or `cooperation`:** R6 already confirmed a real
fitness landscape exists for `offspring_investment_fraction` (fitness outcomes are not
flat across fixed values) — of the three traits tried, it's the one where "a real signal
exists but selection can't detect it at this population scale" is the most plausible
reading of the null R7 result, rather than "there is no signal to detect." Re-testing at
larger population scale validates directly against a trait already known to have real
fitness leverage.

**Plan (not yet scoped in detail):** increase population size (raise
`n_possible_predators`/`n_possible_prey` and/or `n_initial_active_*`, and/or grid size to
match) for `offspring_investment_fraction`, and re-run the same real-vs-neutral-control
replication methodology used for R7. If a real signal emerges that R7 missed, this
validates the "noise floor" hypothesis and the population-scaling direction generally.
If still null, that's a strong case for the combinatorial trait-design pivot (b) below.

---

## Theoretical note — Hinton & Nowlan (1987), a candidate future trait direction

Not yet acted on; recorded here so it isn't lost before Trial 6 (or a Trial 7 combinatorial-design pivot) wrap up.

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

**Candidate future direction, i.e. option (b) above (not scoped, not started):** a trait design
closer to what the paper actually demonstrates — multiple interacting genetic parameters with a
narrow joint fitness optimum (a small combinatorial co-adaptation problem) rather than a single
smooth scalar. Would need its own scoping (what parameters, what joint-optimum structure, how
"learning" maps onto per-individual behavior within an episode) before it's a real R-number in
any module. Per the Trial 6 decision above, option (a) — population scaling on `investment` — is
being tried first since it's cheaper to falsify; revisit this pivot if Trial 6 also comes back
null.
