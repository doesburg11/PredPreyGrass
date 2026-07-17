# Darwin/Baldwin Search — Cross-Module Trial Log

Top-level index for the search for a **sustainable Darwin/Baldwin evolutionary loop** in
predator-prey coevolution. Each module below (`eco_evolutionary_*`) has its own detailed
RESULTS.md with full data; this file exists because that detail is scattered across modules and
there was no single place showing the overall trial sequence and why each pivot happened. Read
top to bottom.

**The goal, unchanged throughout:** genetic evolution (Darwin) and within-lifetime RL learning
(Baldwin) feeding back into each other, demonstrated via three criteria — sustainability,
predator-prey coexistence, and genuine *selection-driven* genome drift (checked against a
neutral-drift control, not just eyeballed) — ideally with the reverse leg too: genome drift
measurably changing the RL fitness landscape, not just RL learning driving genome drift.

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

**Status:** R4 and R5 complete (throttle validated). R6 (fixed-genome fitness sweep) complete —
fitness outcomes are not flat across fixed `offspring_investment_fraction` values (clearest
effect between the lowest value and everything above it). R7 (neutral-control replication, 3
real + 3 control seeds, 1000 iterations each) launched 2026-07-16, in progress. R8 planned. See
`eco_evolutionary_investment/RESULTS.md` for live detail.

---

## Theoretical note — Hinton & Nowlan (1987), a candidate future trait direction

Not yet acted on; recorded here so it isn't lost before R6-R8 wrap up.

Hinton & Nowlan, "How Learning Can Guide Evolution" (1987) — the paper that formalized the
Baldwin effect computationally — offers a plausible theoretical account for why both traits
tried so far (`metabolic_rate`, `offspring_investment_fraction`) have shown weak-to-null
Darwin/Baldwin coupling, and a concrete direction for a trait design that might not.

**1. The Baldwin effect needs a "needle in a haystack," and neither trait tried so far is one.**
The paper is explicit about its own limitation: *"The main limitation of the Baldwin effect is
that it is only effective in spaces that would be hard to search without an adaptive process to
restructure the space."* Their demonstration uses a combinatorial genome (20 genes × 3 alleles,
2²⁰ combinations) with a single narrow fitness spike that pure evolution can't find unassisted —
learning's role is to carve out a detectable "zone of increased fitness" around near-miss
genotypes. `metabolic_rate` and `offspring_investment_fraction` are both single continuous
scalars with a smooth interior optimum (sub-linear gain vs. linear cost; an investment tradeoff
curve). Ordinary selection can climb a smooth 1-D gradient without any help from learning — there
is no haystack for learning to rescue you from. By the paper's own logic, a strong, clearly
measurable Baldwin effect isn't expected in either trait as designed.

**2. The paper's "learning" is individual-lifetime search; PPO here is population-level policy
optimization.** In the simulation, each of 1000 organisms performs its own random-search learning
trials within its own lifetime, and that individual's discovery determines that same
individual's fitness. In this project's environments, "learning" is a shared PPO policy trained
across the whole population over many iterations — an agent born late in training inherits an
already-mostly-trained policy; it doesn't independently search anything itself. This is a real
structural gap from the classical Baldwin effect, and plausibly part of why the reverse leg
specifically (genome shaping what gets learned) has been hard to detect — there isn't genuine
individual-lifetime search for the genome to interact with.

**Candidate future direction (not scoped, not started):** a trait design closer to what the paper
actually demonstrates — multiple interacting genetic parameters with a narrow joint fitness
optimum (a small combinatorial co-adaptation problem) rather than a single smooth scalar. Would
need its own scoping (what parameters, what joint-optimum structure, how "learning" maps onto
per-individual behavior within an episode) before it's a real R-number in any module. Revisit
after R6-R8 (`eco_evolutionary_investment`) conclude, and only if that line also comes back null
— not a replacement for it.
