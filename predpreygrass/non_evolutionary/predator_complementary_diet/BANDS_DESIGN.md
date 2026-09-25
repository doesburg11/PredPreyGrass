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

## 9. Draft addition: band fission and fusion (not built; 2026-09-25)

Status quo: the number of bands can only fall (a band ends when its last member dies; band IDs are created only at reset).
Real hunter-gatherer bands split and merge with the seasons and the food supply, so the count would rise and fall. Proposed as
default-off options so all existing runs stay reproducible.

**Rules (mechanical, no reward terms)**
1. **Fission by size** (`band_fission`, `band_max_size`, default 12): when a band has more than `band_max_size` living members, it
   splits into two spatial clusters (2-means on member positions, each part at least `band_min_split_size` = 3). The
   part with the higher-energy centroid keeps the ID; the other gets a NEW ID from a counter that never reuses IDs. Children stay with
   the parent they are near (a dependent child moves with the part containing its living mother, else its father).
2. **Fusion by contact** (`band_fusion`, `band_fuse_distance` = 3, `band_fuse_steps` = 30): two bands fuse when their centroids stay
   within `band_fuse_distance` for `band_fuse_steps` consecutive checks AND the combined size is at most 0.75 x `band_max_size`
   (hysteresis: this prevents an immediate re-split). The lower ID survives.
3. **Optional: individual drift-out.** A member with no same-band member within `band_share_range` for `band_drift_steps` (50)
   leaves and becomes a one-member band; it can be absorbed later by fusion. Off by default.
4. Checked every `band_check_interval` (10) steps, not every step, to limit churn.

**What stays unchanged:** sharing, kin exclusion (lineage is by agent, not by band), marriage (moves an individual between current
bands), and the same-band / other-band observation channels (they always reflect current membership).

**New state and metrics:** `_next_band_id`; per-episode `band_fissions`, `band_fusions`, and the time series of `bands_alive` and
mean band size (already partly logged as `bands_alive` at the end of an episode). Snapshot and removal bookkeeping as for the other
band state.

**Design consequences and risks**
- Bands become close to spatial clusters, so the within-range sharing rule and band membership largely coincide; that is realistic
  (camp = co-residence) but reduces how much "band" adds beyond proximity. A comparison with fission/fusion off shows how much.
- Policies see membership only through the two band channels, so frequent re-labelling changes their inputs; the hysteresis
  (`band_fuse_steps`, the 0.75 factor, the check interval) is there to keep membership stable enough to learn from.
- Fission at a size cap is a designed rule, so what it tests is what the payoffs do to bands that can split, not that bands would
  choose to split.

**Tests to write first:** the split geometry (two clear clusters, minimum part size, ID never reused, children follow the mother),
fusion hysteresis (no immediate re-split), determinism under the env seed, snapshot round trip, and that both options off give
bit-identical behaviour to the current module.

**First experiment:** 3 bands (18 predators) with `band_max_size` = 8 so that growth forces fission, 150 iterations, sharing
0.6 / 0.3, meat share 0.10, fruit regrowth ~0.05; measure `bands_alive` over time, female survival, and the behaviour analysis
(does cohesion or the foraging split change?).
