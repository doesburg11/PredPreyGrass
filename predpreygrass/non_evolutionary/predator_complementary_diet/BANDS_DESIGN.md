# predator_bands: design draft (not built; for approval)

Status: draft, 2026-09-25. Nothing here is implemented. It would be a clone of `predator_complementary_diet` (itself a
clone of `predator_sexual_reproduction`), so earlier modules and results are untouched. Non-evolutionary; PPO only.

## Question

With sexual reproduction and a complementary diet in place, does *band living* (shared food, cohesion, marriage between
bands) let a viable, differentiated population exist, and do members learn to stay together and divide foraging? The
sharing rule itself is designed in (like the existing mate gifts); what is tested is what the payoffs then produce.

## 1. Initial structure

- About 5 bands of 6-8 predators each on the 25x25 grid, placed in separate areas (start spatially clustered).
- Each band: a few families (male + female + juveniles) plus unpaired adults of both sexes. Every predator has a `band_id`.
- Scale: 30-40 learning predators (the base module starts with 6), so scripted prey (`scripted_prey`) are assumed.
- Children = offspring of living parents that have not yet reproduced (already tracked: `agent_parents`, `has_reproduced`).
  No age variable; "grown up" stays reproduction-based.

## 2. Food sharing within a band (mechanical, like the gifts)

- When a member eats (prey or fruit), a fraction `band_share_rate` goes to band members within `band_share_range`
  (Chebyshev), split equally; meat stays meat and fruit stays fruit (stores are preserved, as in the diet module).
- Replaces the recorded-mate gift and parental care (a mate/child in the band is covered by the band rule); the recorded-mate
  machinery is kept only for marriage bookkeeping.
- No reward term for grouping. Cohesion, if it appears, is learned because staying within range pays in energy.
- Open: equal split vs need-based split; free-riding (a member who only stays home) is expected to be possible.

## 3. Membership changes

- **Marriage:** when a male and a female of *different* bands reproduce, one joins the other's band. Default: the female joins
  the male's band; the reverse is a variant to test.
- **Newborns** join their parents' band.
- **Death** removes the member; a band with no adults left dissolves.
- **Kin exclusion:** no mating between parent and child or between siblings (lineage is tracked), so after one generation
  within-band partners run out and exogamy becomes necessary.
- **Cost of leaving:** a member outside band range receives no shares; that is the trade-off, not a reward.

## 4. Observation

Predators still cannot see sex. To make bands learnable, add channels: same-band vs other-band predators (two layers, or
one layer with band-relative sign), so a predator can tell who shares food with it. Open: whether to expose the band's
total energy.

## 5. Later addition: big game ("mammoths")

A rarer, high-energy prey (15-20) that one hunter rarely kills; success and death odds depend on the number of predators
within range of it, from any band. The kill is split among bands in proportion to (sum of member energy)^r; r = 1
proportional, larger r approaches winner-takes-all; the winning band then applies the sharing rule. Only meaningful once
bands exist; needs an "adjacent hunters count" attack rule (predators cannot share a cell).

## 6. Decision defaults (to confirm)

5 bands of about 6-8; equal split; female joins the male's band; kin exclusion on; r = 1; scripted prey on.

## 7. Controls and comparisons

1. Bands vs no bands (`band_share_rate = 0`) under the same diet requirement.
2. With vs without kin exclusion (does exogamy appear?).
3. Sharing on vs off for the diet requirement (does band sharing rescue females, as the diet pilots suggest it must?).

## 8. Risks

- Learning: 30-40 agents and joint behaviour are hard for independent PPO learners; needs a calibration pilot first.
- Sharing is a designed rule, so results say what payoffs produce, not that sharing emerges.
- Scripted prey change the ecology relative to the learned-prey modules.
- Population, band and lineage bookkeeping are new state (snapshots, cleanup); needs tests and a Codex review before any run.
