# Nuptial Gift-Giving: Sexed Predators with Obligate Male Provisioning

## The mechanism

Predators are split into two sexes with separate learned policies:

- **predator_male**: hunts prey exactly like a normal predator. Never reproduces.
- **predator_female**: never hunts. Grazes grass instead, but each graze event is capped
  by `female_grass_energy_gain_cap` (default `0.10`), set *below*
  `basal_energy_cost_predator_female` (default `0.15`). Even in the impossible best case
  of grazing successfully on every single step, a female's net energy income from grazing
  alone cannot exceed zero — a mathematical guarantee, not just a statistical tendency,
  that grazing alone can never reach `predator_female_creation_energy_threshold`. Only
  `predator_female` ever reproduces.

Each step, a `predator_male` that successfully hunts donates
`male_donation_rate * (that step's hunting gain)` — a **heritable, mechanically-executed
genome trait**, not a learned action — to `predator_female` neighbors within
`cooperation_range` (Chebyshev distance). This is a nuptial gift (Vahed 1998): real gift-
giving species show males provisioning females with resources tied to reproductive
access. Donation is meal-sharing, not a tax on standing energy: a male who doesn't hunt
this step donates nothing, and a male with no eligible female neighbor keeps the full
catch.

Because offspring spawn adjacent to their mother, spatial neighbors are more likely to be
kin than a random draw from the population (population viscosity) — so gifts are
kin-biased without any explicit kin-recognition mechanism, the same substrate used by
`eco_evolutionary_cooperation`'s kin-selection test.

Episode termination is three-way: the episode ends when `prey`, `predator_male`, or
`predator_female` counts hit zero (in addition to the normal step-limit truncation).

## Why donation is a genome trait, not a learned action

This came out of a specific credit-assignment argument, not an arbitrary design choice.
If "share or not" were a per-step PPO action, an independent policy-gradient male would
have zero incentive gradient toward sharing: his own reward stream never reflects whether
the female he fed later reproduces — that outcome lives entirely in a different agent's
trajectory. Standard multi-agent PPO (even with policies literally sharing weights across
male instances) computes each agent's advantage from its own subsequent rewards, so a
male's "give energy" action looks like pure loss with no compensating signal anywhere in
his own returns, regardless of the population-level consequences.

The alternative that actually creates selection pressure is genotype-level (Darwinian)
selection: the donation itself is executed mechanically by the environment based on the
genome value (exactly like `cooperation_rate` in `eco_evolutionary_cooperation`), and only
the *rate* is inherited with mutation and selected across generations by whether it leads
to more surviving/reproducing lineages. This sidesteps the RL credit-assignment problem
entirely — no policy gradient is involved in the giving decision at all, only in each
agent's own movement/foraging/hunting behavior.

## Why this is a structurally different test than prior traits

Three prior heritable traits in this search — `metabolic_rate`, `offspring_investment_fraction`,
`cooperation_rate` (see `predpreygrass/evolutionary/RESULTS.md`) — all came back null on
criterion 3 (selection-driven genome drift beyond neutral noise). In every case the trait's
fitness effect was a smooth, soft tradeoff: a scalar that helps a little either way, with no
sharp landscape for selection to grip. The cross-module RESULTS.md explicitly flags (citing
Hinton & Nowlan 1987) that a single continuous scalar trait with a smooth fitness landscape
may be the wrong shape of problem for a detectable Baldwin effect.

`male_donation_rate` under this design is deliberately the opposite: predator-females have
an **obligate**, not smooth, dependency — `rate=0` should be close to lethal for the female
lineage (no reproduction without gifts), not merely suboptimal. This is intentionally the
sharpest, least-smooth fitness landscape attempted yet in this search.

## Sex-limited genome expression

`male_donation_rate` is autosomal: offspring inherit and can pass on the trait value from
their mother regardless of which sex they are assigned at birth (see
`_handle_female_reproduction`), but the value is only ever mechanically *expressed* for
`predator_male` agents (see `_apply_nuptial_gift`, only ever called from
`_handle_male_hunting`). A daughter silently carries her mother's donation-rate value and
can pass it to a son of her own; the trait's phenotypic effect skips a generation whenever
it passes through a female. This is standard sex-limited-expression genetics (e.g. milk
yield genes carried but not expressed in bulls) and is why `founder_genome` configures both
`predator_male` and `predator_female` entries identically — the initial population is
genetically uniform across sexes even though only males ever act on the value.

Offspring sex itself is an independent, unbiased coin flip at each birth
(`rng.random() < 0.5`), not part of the genome and not subject to mutation/selection.

## Omnivory side effect

Because `predator_female` grazes the same grass patches prey do (with a partial-browsing
cap, rather than fully consuming the patch like prey), predator-females compete with prey
for grass. This is an intended emergent ecological interaction (omnivory), not a bug — see
`_handle_female_grazing`.

## The open measurement problem: male fitness is not directly observable

A male's own reproductive success is never directly observable — he never reproduces
himself. Unlike `cooperation_rate`'s `coop_repro_spearman` (a same-agent correlation between
an individual's trait value and whether *that individual* reproduced), there is no
individual-level correlation available for `male_donation_rate`: the reproduction event
belongs to a different agent (the female he fed, if any). The primary signal this module can
actually deliver is:

- **Population-level drift**: `live_genome/predator_male_male_donation_rate_mean` moving away
  from the founder value of `0.0` over training iterations — the direct Darwin signal.
- **`female_reproduction_gift_share_mean`**: at each female reproduction event, what fraction
  of her lifetime energy intake (`lifetime_energy_from_gifts / (lifetime_energy_from_gifts +
  lifetime_energy_from_grazing)`) came from gifts vs. her own grazing. This is the direct
  mechanistic test of the obligate-gate claim — expected to sit near `1.0` if
  `female_grass_energy_gain_cap` is tuned correctly (i.e., grazing alone truly cannot get a
  female to threshold).

A lineage-forward "did this specific male's gifts precede a birth" efficacy metric is a
plausible stretch goal but is not implemented — it would require tracing forward from each
donation event to the recipient's subsequent reproduction, which is a meaningfully bigger
instrumentation lift than the metrics above.

## Config reference

Key parameters in `config/config_env_eco_evolutionary.py`:

```python
"basal_energy_cost_predator_male": 0.15,
"basal_energy_cost_predator_female": 0.15,
"female_grass_energy_gain_cap": 0.10,        # < basal decay: obligate-gate guarantee
"predator_female_creation_energy_threshold": 12.0,
"initial_energy_predator_female": 8.0,       # bumped from 5.0, see "Retuning" below
"founder_genome": {
    "predator_male":   {"male_donation_rate_mean": 0.5, "male_donation_rate_std": 0.1},
    "predator_female": {"male_donation_rate_mean": 0.5, "male_donation_rate_std": 0.1},
    "prey":            {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
},
"genome_mutation": {"rate": 0.05, "std": 0.04},
"trait_bounds": {"male_donation_rate": (0.0, 1.0)},
"cooperation_range": 4,                      # widened from 2, see "Retuning" below
```

### Retuning (2026-08-01/02): initial energy, cooperation range, and founder mean

The first fixed-genome pilot (40 iterations, original config: `initial_energy_predator_female:
5.0`, `cooperation_range: 2`) showed episode length capped at ~37-44 steps across *every*
donation rate tested, including `1.0` — suspiciously close to a female's own pure-starvation
survival time (`5.0 / 0.15 ≈ 33` steps). Two parameters were bumped in response:

- `initial_energy_predator_female`: `5.0 -> 8.0` — buys more real time for a gift to arrive
  before starvation. Does **not** weaken the obligate-gate guarantee: that guarantee is about
  net grazing income (`female_grass_energy_gain_cap < basal_energy_cost_predator_female`),
  which is independent of initial energy — she still trends toward zero without gifts, just
  more slowly.
- `cooperation_range`: `2 -> 4` — under an early, high-entropy, largely-random movement policy,
  "a male's catch happens to be near a female" is a rare joint event at range 2; widening it
  raises that probability independent of dispersal skill.

Validated directly: at `male_donation_rate=1.0` with the retuned config, 60 iterations produced
**34.5 total reproduction events and 508 total gift energy donated**, with episode length
growing from ~34 to 80-130 steps and the female population visibly growing past its 3-agent
founding cohort. At `male_donation_rate=0.0`, the same 60 iterations produced **zero**
reproduction events and **zero** gifts, throughout. See "Fixed-genome sweep results" below.

**A second, later retune was needed on the founder mean itself.** The neutral-control
replication (real evolutionary mode, `genome_enabled: True`) originally kept the founder mean at
`0.0` (matching the "neutral start" convention used by every other trait in this project, so any
drift is attributable to selection). That put the *entire starting population* deep in the
confirmed-null regime from the sweep — and mutation only fires *at* a reproduction event, so a
population that never reproduces once can never mutate toward a higher, viable rate either. This
is a structural deadlock, not a slow start: confirmed directly when real seed 42 showed zero
`female_reproduction_events_total` through 530/1000 iterations, genome completely static the
whole way. Founder mean bumped `0.0 -> 0.5`, std `0.05 -> 0.1`, and the run restarted. This
sacrifices the "founder mean=0 is a clean neutral baseline" framing, but that framing is moot for
an experiment that can't leave the starting gate — the real-vs-control comparison logic is
unaffected by where the shared founder mean sits, since both conditions use the same one. A
15-iteration smoke test with the new mean confirmed gift energy flowing immediately (up to
~2.1/iteration by iteration 10, vs. exactly `0.0` throughout under the old mean=0.0 config).

## Metrics to watch in training

- `eco_evolution/predator_male_male_donation_rate_mean/std/p25/p50/p75` — the primary Darwin
  signal (population of live males).
- `eco_evolution/female_reproduction_gift_share_mean/p50` — the obligate-gate mechanistic check.
- `eco_evolution/female_reproduction_events_total` — sanity check that reproduction is
  actually happening at all.
- `eco_evolution/gift_energy_donated_total` / `gift_energy_received_total` — population-level
  flow sanity check (should be equal).
- `eco_evolution/gift_local_relatedness_proxy` — fraction of donated energy that went to
  genuine kin (mother/sisters, from `agent_parents`) vs. unrelated females; the Hamilton's-rule
  moderator, same interpretation as `cooperation_rate`'s equivalent metric.
- `eco_evolution/predator_male_count`, `predator_female_count`, `prey_count`,
  `peak_active_predator_male/female/prey` — sustainability/coexistence checks (criteria 1, 2).

## Fixed-genome sweep results (partial: 2 of 5 values valid)

Staged-rollout step 2: freeze `male_donation_rate` at fixed values (`genome_enabled: False`,
mechanically executed, no inheritance/mutation — see `_apply_nuptial_gift`) and check whether
the obligate gate actually creates a real fitness landscape before spending compute on a full
replication.

**Intended design**: 5 values (`0.0, 0.25, 0.5, 0.75, 1.0`), 60 iterations each, retuned config.

**What actually has valid data**: only the two extremes.

| `male_donation_rate` | reproduction events (total over 60 iters) | gift energy donated | episode length trend |
|---|---|---|---|
| `0.0` | **0** | **0** | flat, never grows |
| `1.0` | **34.5** | **508** | grows 34 -> 80-130 steps; female count grows past founders |

A dramatic, clean contrast — exactly the obligate, non-smooth fitness landscape this module was
designed to produce (contrast with `cooperation_rate`/`metabolic_rate`/`offspring_investment_fraction`,
all of which had smooth landscapes and came back null on selection).

**`0.25`, `0.5`, `0.75` are missing/invalid**, not because of anything about the mechanism —
an operational mistake during this session: the underlying training script was edited (to add
`--seed` support for the replication, below) *while the sweep's shell loop was still calling it*
for these three values. `0.25`'s invocation silently ignored all its CLI flags (an even-earlier
edit had briefly removed argument parsing entirely) and ran an unplanned, unrelated 250-iteration
real-mode run instead; `0.5` and `0.75` hit the next edit (which restored argument parsing but
had dropped the `--fixed-donation-rate` flag) and errored out instantly, producing no data at
all. The two extremes were run earlier and are unaffected and valid. Not re-run as of this
writing — the two-point contrast was judged sufficient to justify moving to the replication
before spending more compute on the full dose-response shape. Re-running the missing three
values (`run_fixed_genome_sweep.sh`-style, once it exists again) is a reasonable follow-up if
the dose-response *shape* (not just "does it matter at all") becomes interesting later.

**Operational gotcha found along the way, relevant to any future high-parallelism run of this
module**: `progress.csv`'s column schema locks in from an early iteration's reported keys and
silently drops any metric (including RLlib's own built-in `episode_len_mean`) that doesn't exist
yet at that point. With enough env-runner parallelism relative to episode length, iteration 1 can
have zero completed episodes, permanently losing those columns from the CSV for the whole run
even though later iterations have valid values. `result.json` (JSONL, one full result per line,
no fixed schema) is unaffected and is what `analyze_replication_seeds.py` reads — use it, not
`progress.csv`, for any programmatic analysis of this module's training runs.

## Staged rollout plan

Following the same discipline used for `offspring_investment_fraction` (R4-R7) and
`metabolic_rate` in `predpreygrass/evolutionary/RESULTS.md`:

1. **Smoke test** — done.
2. **Fixed-genome fitness sweep** — partially done, see above. Two-point contrast (`0.0` vs
   `1.0`) confirms a real, dramatic fitness landscape exists; the middle three values are an
   open gap, not a null result.
3. **Neutral-control replication** — **stopped early, 2026-08-03, not completed**. 3 real seeds
   (42, 43, 44) + 3 neutral-control seeds (42, 43, 44) planned, 1000 iterations each, sequential
   (GPU + 28 env runners), via `run_replication_seeds.sh`, auto-launched by a chain watcher once
   the fixed-genome sweep's process exited. Founder mean was bumped `0.0 -> 0.5` (std `0.1`)
   first, after an earlier attempt at founder mean `0.0` deadlocked completely (see "Retuning"
   above) — that fix worked mechanically (gifts flowed, reproduction was no longer literally
   impossible), but the resulting run still wasn't informative enough to justify the remaining
   ~4 days of compute:

   - **Real seed 42 ran to completion** (1000/1000 iterations, 15.9h). Result: **too few
     reproductions to be meaningful** — 18.59 total reproduction events summed across the whole
     run, ~80% of iterations (796/998) had *zero* reproduction events, and no growth trend
     (first-100-iteration mean 0.0164/iter vs. last-100 mean 0.0142/iter — flat to slightly
     down, not rising). **No meaningful episode-length increase** — oscillated in the 50-70 step
     range the entire run (overall mean 61.4, first-100 mean 56.8 vs. last-100 mean 61.7), never
     showing the sustained climb into the 80-130 range that the `rate=1.0` fixed-genome pilot
     showed. **No meaningful directional drift** — `male_donation_rate_mean` oscillated in a
     narrow ~0.53-0.58 band around the founder value of `0.5` for the entire run (started
     0.575, ended 0.571), no visible upward or downward trend.
   - Real seed 43 was killed partway (~52%, iteration ~519/1000) when the decision was made to
     stop; no complete data from it. No control seeds were started.
   - **Addendum (2026-08-26):** independently checked seed 42's `male_donation_rate_mean`
     trajectory with `predpreygrass/evolutionary/model_selection.py`'s AICc model selection
     (Stasis/URW/GRW). Predator-male fits **Stasis** with Akaike weight 1.000 — a formal
     confirmation of the "oscillated in a narrow band, no visible trend" read above, not a new
     or surprising finding once seen alongside it: with only ~18.6 reproduction events across
     the whole 1000-iteration run, there are too few mutation/inheritance events for the trait
     to accumulate any real random-walk variance, so a near-zero-process-noise (Stasis-like) fit
     is the mechanistically expected result, not an anomaly. Predator-female's own fit (URW,
     Akaike weight 0.421) is close enough to Stasis' fit for the same seed to not be a
     meaningful qualitative difference — both point at the same "too little reproduction for
     drift, real or neutral, to say anything" conclusion the paragraph above already reaches by
     a different route.
   - **Verdict**: this specific config (founder mean `0.5`, real evolutionary mode) produces a
     population that survives but doesn't reproduce often enough, or drift its genome enough,
     for the real-vs-control comparison to plausibly say anything — continuing to the full 6-run
     replication was judged not worth ~4 more days of compute given this trajectory.

   **Candidate next steps, not yet started**: (a) further retune the energy economy (e.g.
   another bump to `initial_energy_predator_female` or `cooperation_range`, or lowering
   `predator_female_creation_energy_threshold`) to push reproduction rate up before attempting
   the replication again; or (b) sidestep the evolutionary bootstrap problem entirely for a first
   look by training with genome evolution disabled and `male_donation_rate` **fixed at `0.8`**
   (i.e. extending the fixed-genome sweep, which only directly tested `0.0` and `1.0` — see
   "Fixed-genome sweep results" above) to check whether a real run (not just the 60-iteration
   pilot) sustains healthy reproduction and longer episodes at a high-but-not-maximal rate,
   before reinvesting in the harder evolutionary-drift question.

## What This Is Not

This module does not copy parent PPO weights into offspring. Learned policy weights remain
shared per policy group (`predator_male`, `predator_female`, `prey`) unless a future experiment
explicitly adds individual policy inheritance — see `predpreygrass/evolutionary/eco_evolutionary/README.md`'s
"What This Is Not" for the same argument made for the base eco-evolutionary module.

## Key Files

- `predpreygrass_rllib_env.py`: environment with sexed predators, foraging-niche split,
  nuptial-gift donation, three-way termination, and genome inheritance.
- `config/config_env_eco_evolutionary.py`: per-sex energy/threshold parameters, the grazing
  cap, founder distributions, and mutation settings.
- `config/config_env_eco_evolutionary_neutral_control.py` /
  `tune_ppo_nuptial_gift_neutral_control.py`: scaffolded but not yet run — see staged rollout
  plan above.
- `utils/genome.py`: `Genome` dataclass (`male_donation_rate`) plus founder/mutation helpers.
- `tests/test_eco_evolutionary_validation.py`: regression tests for sex assignment, the
  hunting/grazing behavioral split, gift directionality, three-way termination, and genome
  inheritance from the mother regardless of offspring sex.

## Quick Test

```bash
python -m pytest -q predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/tests/test_eco_evolutionary_validation.py
```

## References

**Nuptial gift-giving:**
- Vahed, K. (1998). *The function of nuptial feeding in insects: a review of empirical
  studies.* Biological Reviews, 73(1), 43-78 — the real biological mechanism this module is
  named for and modeled on.

**Kin selection substrate (shared with `eco_evolutionary_cooperation`):**
- Hamilton, W. D. (1964). *The genetical evolution of social behaviour.* Journal of
  Theoretical Biology, 7(1), 1-52.

**Baldwin Effect / Darwin-Baldwin search framing:**
- Hinton, G. E. & Nowlan, S. J. (1987). *How Learning Can Guide Evolution.* Complex Systems,
  1(3), 495-502 — see `predpreygrass/evolutionary/RESULTS.md` for how this motivates the
  obligate-vs-smooth design choice made here.
