# Eco-Evolutionary Cooperation — Training Results

## Status: Pilot 1 — preliminary, likely null (not yet a final verdict)

Two 250-iteration, single-seed runs (GPU config, seed 41) completed 2026-07-17/18:
a real run and a `genome_neutral_drift_control` run (genome inheritance severed from
reproductive success; population/spatial dynamics unchanged — see
`config/config_env_eco_evolutionary_neutral_control.py`). This is a pilot comparison,
**not** the properly-powered 3-seed-real-vs-3-seed-control replication (Mann-Whitney U)
that `metabolic_rate` and `offspring_investment_fraction` were held to before either was
called null — see `predpreygrass/evolutionary/RESULTS.md`. Treat this as a first look
only.

## Environment recap

- Founder `cooperation_rate`: mean 0.0, std 0.05 for both species (identical to the
  no-genome baseline; any drift is attributable to mutation and/or selection).
- Mutation: rate 0.05, std 0.04 per reproduction event.
- Trait bounds: `[0.0, 1.0]`.
- `cooperation_range`: 2 (Chebyshev).
- Donation: meal-sharing model — `cooperation_rate * (this step's catch/graze gain)`
  split among same-species neighbors in range; no donation on non-eating steps or when
  no eligible neighbor is present.

## Ecology (sustainability, coexistence)

Both runs sustained both species for the full 250 iterations, no extinction:

| | Real: iter 1 | Real: iter 250 | Control: iter 1 | Control: iter 250 |
|---|---|---|---|---|
| `predator_count` | 5.0 | 17.1 | 5.0 | 17.2 |
| `prey_count` | 8.3 | 21.8 | 8.4 | 12.0 |

Criteria 1 (sustainability) and 2 (coexistence) look satisfied in both runs — population
trajectories are broadly comparable between real and control, as expected (the control
only severs the genome-inheritance link, not population/energy dynamics).

## Genome evolution — the Darwin signal

`cooperation_rate_mean`, iteration 1 → 250:

| Species | Real | Control |
|---|---|---|
| Predator | 0.019 → **0.010** (↓ 44%) | 0.019 → **0.023** (↑ 21%) |
| Prey | 0.014 → **0.023** (↑ 68%) | 0.014 → **0.035** (↑ 152%) |

`cooperation_rate_std` narrowed in both runs and both species (real: predator
0.021→0.012, prey 0.022→0.014; control: predator 0.021→0.020, prey 0.023→0.014) —
narrowing on its own is not diagnostic here since it happens in the control too.

**This is the same red flag that turned out to be noise for both prior traits.** The
control — where selection on `cooperation_rate` is structurally impossible — drifted as
much as or more than the real run in both species. For predators, the real run's
direction (down) is even the *opposite* of the control's (up). Nothing here clears the
bar of "the real run's drift exceeds what the neutral control alone produces."

## The Baldwin signal — not measurable in this pilot

The README's "what to watch" list specifies `local_relatedness_proxy`,
`energy_donated_total`/`energy_received_total`, `coop_repro_spearman`, and
`coop_repro_rate_q1`-`q4` as the metrics needed to interpret *why* any drift happens
(kin-biased vs. not) and to test the reverse leg. **None of these are implemented in
`utils/episode_return_callback.py`** — only the population-level
`cooperation_rate_mean/std/p25/p50/p75` and `predator_count`/`prey_count` are logged.
This means the current data cannot distinguish "no selection pressure" from "selection
pressure existed but was cancelled by kin competition" (Taylor 1992) — the central
scientific question this module was built to test is not yet instrumented.

## Darwin/Baldwin loop verdict

**Preliminary, not confirmed:** the single-seed real-vs-control comparison shows no
signal that clearly exceeds neutral noise, consistent with the null pattern already
established for `metabolic_rate` and `offspring_investment_fraction`
(`predpreygrass/evolutionary/RESULTS.md`). This would make three-for-three null results
on criterion 3 across three differently-mechanised traits — but this particular result
should not be treated as confirmed until (a) it survives a proper multi-seed replication,
and (b) the relatedness/reverse-leg metrics exist to rule in or out Taylor's cancellation
as the explanation, rather than "no selection pressure at all."

## Next steps

1. Implement the missing metrics (`local_relatedness_proxy`,
   `energy_donated_total`/`_received_total`, `coop_repro_spearman`,
   `coop_repro_rate_q1`-`q4`) in `episode_return_callback.py` before spending compute on
   a full replication — without them, even a positive replication result would be hard
   to interpret mechanistically.
2. If a real signal still looks plausible after instrumenting those metrics, run the
   same 3-seed-real-vs-3-seed-control Mann-Whitney U replication used for `investment`
   R7.
3. If this also comes back null, `cooperation_rate` joins `metabolic_rate` and
   `offspring_investment_fraction` as a third null result — worth folding into the
   top-level design-decision discussion (population scaling vs. combinatorial/multi-gene
   trait design) already flagged in `predpreygrass/evolutionary/RESULTS.md`.
