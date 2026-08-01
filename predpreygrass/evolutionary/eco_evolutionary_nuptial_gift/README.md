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
"founder_genome": {
    "predator_male":   {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.05},
    "predator_female": {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.05},
    "prey":            {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
},
"genome_mutation": {"rate": 0.05, "std": 0.04},
"trait_bounds": {"male_donation_rate": (0.0, 1.0)},
"cooperation_range": 2,
```

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

## Staged rollout plan

Following the same discipline used for `offspring_investment_fraction` (R4-R7) and
`metabolic_rate` in `predpreygrass/evolutionary/RESULTS.md`:

1. **Smoke test** (done as part of building this module): a handful of training iterations,
   confirming no crashes and sane metric output.
2. **Fixed-genome fitness sweep** (not yet run): freeze `male_donation_rate` at several values
   including `0.0` via `genome_enabled: False`-style founder override, and confirm `rate=0`
   causes female/predator collapse while higher values sustain the population — this is the
   check that the obligate gate actually behaves as designed, before spending compute on a full
   replication. `metabolic_rate` and `offspring_investment_fraction` both confirmed a real
   fitness landscape at this stage before their (ultimately null) selection tests; this module's
   landscape is expected to be far starker (near-lethal at `rate=0`, not merely worse).
3. **Neutral-control replication** (not yet run): only after step 2 confirms a real landscape,
   port the `genome_neutral_drift_control` flag (already scaffolded — see
   `config/config_env_eco_evolutionary_neutral_control.py` and
   `tune_ppo_nuptial_gift_neutral_control.py`) into a proper 3-real + 3-control-seed replication,
   compared via Mann-Whitney U, mirroring `offspring_investment_fraction`'s R7.

Steps 2 and 3 are expensive, separate follow-on runs — deliberately out of scope for the initial
build of this module.

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
